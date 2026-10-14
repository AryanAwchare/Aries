"""Operations control-plane router.

Exposes the platform's *actions* to the terminal UI: runtime configuration,
research/CLI jobs (backtest, walk-forward, ablation, replay, leader polling),
leader-roster management, audit + risk-event logs, model-registry info, and a
risk-gated manual order ticket.

Design notes
------------
* Jobs are process-local and asynchronous. Heavy CPU work is dispatched with
  ``asyncio.to_thread`` so the event loop stays responsive.
* Nothing here changes kill-switch semantics: the risk engine is authoritative
  (``enabled=True`` == trading allowed). The UI labels the switch truthfully.
* Config edits mutate the shared ``app.state.settings`` so long-running jobs and
  the dashboard observe them immediately; they are optionally persisted to
  ``.env`` for restarts.
"""
from __future__ import annotations

import asyncio
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Coroutine

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from ... import services
from ...config import BASE_DIR, DATA_DIR, Settings, TradingMode
from ...db import audit
from ...db.base import Database
from ...db.models import OrderRecord, RiskEvent, SystemEvent, TradeSide
from ...execution.base import OrderRequest
from ...execution.killswitch import KillSwitch
from ...logging import get_logger
from ...risk.engine import AccountState, OrderCandidate, RiskEngine
from ...risk.limiter import RateLimiter
from ...risk.prop_profiles import PropProfile, load_profile, load_profile_dir
from .dashboard import (
    MT5_AVAILABLE,
    _get_mt5_account_sync,
    _get_mt5_candles_sync,
    _get_mt5_positions_sync,
)

log = get_logger(__name__)

router = APIRouter(prefix="/api/ops", tags=["ops"])

_PROFILE_DIR = BASE_DIR / "app" / "risk" / "prop_profiles"
_MODEL_DIR = DATA_DIR / "models" / "wf_main"
_RUNS_DIR = DATA_DIR / "models" / "runs"

_ALLOWED_TIMEFRAMES = {"M1", "M5", "M15", "M30", "H1", "H4", "D1"}
_ALLOWED_DRIVERS = {"paper", "ea", "metaapi"}
_DEFAULT_STOP_POINTS = 5.0

_CLI_REFERENCE: list[dict[str, Any]] = [
    {"command": "setup-db", "description": "Create the database schema", "args": []},
    {"command": "poll-leaders", "description": "Poll Myfxbook live-tier leaders once", "args": ["--once"]},
    {
        "command": "scan-leaders",
        "description": "Discover, rank and rotate leaders (two-tier)",
        "args": ["scan|list|add-check", "--apply", "--add <id>", "--name", "--url", "--remove <id>", "--live <ids>", "--id <id>"],
    },
    {"command": "backtest", "description": "Run the hybrid-signal backtest", "args": ["--synthetic", "--days 730", "--seed 42"]},
    {"command": "ablation", "description": "Run the 6-way ablation comparison", "args": ["--synthetic", "--days 1800", "--seed 42"]},
    {"command": "walk-forward", "description": "Walk-forward train/validate/test", "args": ["--synthetic", "--calibrate", "--days 800", "--seed 42"]},
    {"command": "replay", "description": "Replay bars through the live stack", "args": ["--synthetic", "--days 180", "--seed 42"]},
]


# ---------------------------------------------------------------------- helpers
async def _session(request: Request) -> AsyncSession:
    db: Database = request.app.state.db
    async with db.session_factory() as session:
        yield session


def _active_profile(request: Request) -> PropProfile:
    profile = getattr(request.app.state, "profile", None)
    if profile is not None:
        return profile
    settings: Settings = request.app.state.settings
    return load_profile(_PROFILE_DIR / f"{settings.active_prop_profile}.yaml")


def _active_kill_switch(request: Request) -> KillSwitch:
    return getattr(request.app.state, "kill_switch", None) or services.get_kill_switch()


def _copy_trading(request: Request):
    return getattr(request.app.state, "copy_trading", None) or services.get_copy_trading_toggle()


_LIMITERS: dict[str, RateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def _get_limiter(profile: PropProfile) -> RateLimiter:
    key = f"{profile.name}:{profile.max_orders_per_minute}:{profile.max_orders_per_hour}"
    with _LIMITERS_LOCK:
        limiter = _LIMITERS.get(key)
        if limiter is None:
            limiter = RateLimiter(profile.max_orders_per_minute, profile.max_orders_per_hour)
            _LIMITERS[key] = limiter
        return limiter


def _risk_engine(profile: PropProfile, kill_switch: KillSwitch) -> RiskEngine:
    return RiskEngine(profile, kill_switch, _get_limiter(profile))


def _build_client(settings: Settings, db: Database):
    driver = settings.execution_driver
    if driver == "ea":
        from ...execution.ea_bridge import EaBroker

        return EaBroker(settings, db.session_factory)
    if driver == "metaapi":
        from ...execution.base import MetaApiClient

        account_id = (
            settings.metas_api_prop_account_id
            if settings.trading_mode in (TradingMode.PROP_EVAL, TradingMode.LIVE)
            else settings.metas_api_demo_account_id
        )
        return MetaApiClient(settings.metaapi_token, account_id)
    from ...execution.base import PaperBroker

    return PaperBroker()


def _mask_db_url(url: str) -> str:
    return re.sub(r"://([^:/@]+):([^@/]+)@", r"://\1:***@", url)


def _persist_env(updates: dict[str, str]) -> bool:
    env_path = BASE_DIR / ".env"
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    except OSError:
        return False
    remaining = dict(updates)
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in remaining:
                out.append(f"{key}={remaining.pop(key)}")
                continue
        out.append(line)
    for key, value in remaining.items():
        out.append(f"{key}={value}")
    try:
        env_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    except OSError:
        return False
    return True


# ---------------------------------------------------------------------- jobs
@dataclass
class _Job:
    id: str
    kind: str
    params: dict
    status: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: str | None = None
    finished_at: str | None = None
    result: dict | None = None
    error: str | None = None
    log: list[str] = field(default_factory=list)
    task: asyncio.Task | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "params": self.params,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
            "log": self.log[-200:],
        }


_JOBS: dict[str, _Job] = {}
_JOBS_LOCK = threading.Lock()


@dataclass
class _JobContext:
    settings: Settings
    db: Database
    job_id: str


async def _load_ohlcv(settings: Settings, params: dict, emit: Callable[[str], None]):
    from ...data.ohlcv import CSVProvider, OHLCVRequest, SyntheticProvider

    days = int(params.get("days", 800))
    seed = int(params.get("seed", 42))
    synthetic = bool(params.get("synthetic", False))
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    if synthetic:
        provider = SyntheticProvider(seed=seed)
    else:
        provider = CSVProvider(str(DATA_DIR / "raw"))

    try:
        df = await provider.get_ohlcv(
            OHLCVRequest(symbol=settings.market_symbol, timeframe=settings.market_timeframe, start=start, end=end)
        )
    except FileNotFoundError as exc:
        raise ValueError(
            f"market data not found for {settings.market_symbol}_{settings.market_timeframe} "
            f"(use synthetic=true): {exc}"
        ) from exc
    if df is None or df.empty:
        raise ValueError("no OHLCV rows for the requested window")
    df.attrs["symbol"] = settings.market_symbol
    emit(f"loaded {len(df)} bars {df.index[0]} .. {df.index[-1]}")
    return df


async def _job_backtest(ctx: _JobContext, params: dict, emit: Callable[[str], None]) -> dict:
    from ...backtest.runner import BacktestRunner, ProbScorer, TransactionCostModel

    settings = ctx.settings
    ohlcv = await _load_ohlcv(settings, params, emit)
    profile = load_profile(settings.prop_profile_path)
    risk = RiskEngine(profile, KillSwitch(True), RateLimiter())
    runner = BacktestRunner(risk, scorer=ProbScorer(), cost_model=TransactionCostModel())
    emit("running backtest…")
    result = await asyncio.to_thread(runner.run, ohlcv, profile.capital)
    curve = result.equity_curve
    emit(f"backtest done: {result.n_trades} trades, return {result.total_return_pct:.2f}%")
    return {
        "n_trades": result.n_trades,
        "wins": result.wins,
        "losses": result.losses,
        "win_rate": result.win_rate,
        "total_return_pct": result.total_return_pct,
        "max_drawdown_pct": result.max_drawdown_pct,
        "sharpe": result.sharpe,
        "profit_factor": result.profit_factor,
        "brier_score": result.brier_score,
        "total_commission_paid": result.total_commission_paid,
        "total_slippage_paid": result.total_slippage_paid,
        "total_swap_paid": result.total_swap_paid,
        "equity_curve": [
            {"time": str(ts), "equity": round(float(eq), 2)} for ts, eq in curve.items()
        ][-2000:],
    }


async def _job_walk_forward(ctx: _JobContext, params: dict, emit: Callable[[str], None]) -> dict:
    from ...backtest.leakage import run_leakage_report
    from ...backtest.walkforward import walk_forward, write_fold_report

    settings = ctx.settings
    ohlcv = await _load_ohlcv(settings, params, emit)
    promote = bool(params.get("promote", False))
    out_dir = DATA_DIR / "models" if promote else (_RUNS_DIR / ctx.job_id)

    emit("running leakage checks…")
    leakage = await asyncio.to_thread(run_leakage_report, ohlcv, settings.test_start)
    emit(f"leakage: causal={leakage.features_causal} kronos_overlap={leakage.kronos_test_overlap}")

    emit("training walk-forward fold…")
    folds = await asyncio.to_thread(
        walk_forward,
        ohlcv,
        None,
        train_end=settings.train_end,
        val_end=settings.validation_end,
        test_end=settings.test_end,
        out_dir=out_dir,
    )
    report_path = (DATA_DIR / "wf_report.json") if promote else (out_dir / "wf_report.json")
    out_dir.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(write_fold_report, folds, report_path)

    emit(f"walk-forward done ({'promoted to ' + str(out_dir) if promote else 'run kept at ' + str(out_dir)})")
    return {
        "promoted": promote,
        "out_dir": str(out_dir),
        "leakage": {
            "features_causal": leakage.features_causal,
            "kronos_test_overlap": leakage.kronos_test_overlap,
            "notes": leakage.notes,
        },
        "folds": [
            {
                "fold_name": f.fold_name,
                "winner": f.winner,
                "test_logloss": f.test_logloss,
                "test_auc": f.test_auc,
                "brier_score": f.brier_score,
                "ece": f.ece,
                "n_train": f.n_train,
                "n_test": f.n_test,
                "n_events_test": f.n_events_test,
                "calibration": f.calibration.to_dict(),
            }
            for f in folds
        ],
    }


async def _job_ablation(ctx: _JobContext, params: dict, emit: Callable[[str], None]) -> dict:
    from ...backtest.ablation import AblationEngine
    from ...backtest.runner import TransactionCostModel
    from ...backtest.walkforward import time_slices

    settings = ctx.settings
    ohlcv = await _load_ohlcv(settings, params, emit)
    profile = load_profile(settings.prop_profile_path)
    risk = RiskEngine(profile, KillSwitch(True), RateLimiter())
    ablation = AblationEngine(risk, cost_model=TransactionCostModel())
    slices = time_slices(
        ohlcv,
        train_end=settings.train_end,
        val_end=settings.validation_end,
        test_end=settings.test_end,
    )
    emit("running 6-way ablation across splits…")
    df_results = await asyncio.to_thread(ablation.run_all_ablations, slices)
    rows = df_results.to_dict(orient="records")
    emit(f"ablation done: {len(rows)} rows")
    return {"rows": rows}


async def _job_replay(ctx: _JobContext, params: dict, emit: Callable[[str], None]) -> dict:
    from ...backtest.replay import run_replay
    from ...execution.base import PaperBroker
    from ...execution.executor import Executor
    from ...metamodel.predict import MetaModelPredictor
    from ...strategy.engine import StrategyEngine

    settings = ctx.settings
    ohlcv = await _load_ohlcv(settings, params, emit)
    profile = load_profile(settings.prop_profile_path)
    kill = KillSwitch(True)
    risk = RiskEngine(profile, kill, RateLimiter())
    predictor = MetaModelPredictor(
        DATA_DIR / "models" / "wf_main", threshold=settings.meta_model_prob_threshold
    )
    strategy = StrategyEngine(settings, predictor)
    executor = Executor(settings, risk, PaperBroker(), kill)
    emit("replaying bars through the live stack…")
    stats = await run_replay(ctx.db, strategy, risk, executor, ohlcv)
    emit(f"replay done: bars={stats.bars_seen} signals={stats.signals_fired} filled={stats.orders_filled}")
    return {
        "bars_seen": stats.bars_seen,
        "signals_fired": stats.signals_fired,
        "orders_filled": stats.orders_filled,
        "rejected": stats.rejected,
    }


async def _job_poll_leaders(ctx: _JobContext, params: dict, emit: Callable[[str], None]) -> dict:
    from ...leaders.myfxbook import MyfxbookClient
    from ...leaders.poll import poll_leader_trades

    settings = ctx.settings
    if not settings.myfxbook_email:
        raise ValueError("MYFXBOOK_EMAIL / MYFXBOOK_PASSWORD not configured")
    client = MyfxbookClient(settings.myfxbook_email, settings.myfxbook_password)
    emit("polling live-tier leaders…")
    inserted = await poll_leader_trades(settings, ctx.db.session_factory, client)
    emit(f"inserted {inserted} leader trade events")
    return {"inserted": inserted}


async def _job_scan_leaders(ctx: _JobContext, params: dict, emit: Callable[[str], None]) -> dict:
    from ...leaders.myfxbook import MyfxbookClient
    from ...leaders.poll import refresh_leader_roster

    settings = ctx.settings
    if not settings.myfxbook_email:
        raise ValueError("MYFXBOOK_EMAIL / MYFXBOOK_PASSWORD not configured")
    client = MyfxbookClient(settings.myfxbook_email, settings.myfxbook_password)
    apply = bool(params.get("apply", False))
    emit(f"scanning watch list (apply={apply})…")
    result = await refresh_leader_roster(settings, client, ctx.db.session_factory, apply=apply)
    emit(f"scan done: watched={result.get('watched')} promoted={result.get('promoted')}")
    return result


_JOB_FUNCS: dict[str, Callable[..., Coroutine[Any, Any, dict]]] = {
    "backtest": _job_backtest,
    "walk-forward": _job_walk_forward,
    "ablation": _job_ablation,
    "replay": _job_replay,
    "poll-leaders": _job_poll_leaders,
    "scan-leaders": _job_scan_leaders,
}


async def _run_job(job: _Job, kind: str, settings: Settings, db: Database) -> None:
    job.status = "running"
    job.started_at = datetime.now(timezone.utc).isoformat()

    def emit(message: str) -> None:
        job.log.append(f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} {message}")

    emit(f"starting job '{kind}'")
    try:
        result = await _JOB_FUNCS[kind](_JobContext(settings, db, job.id), job.params, emit)
        job.result = result
        job.status = "done"
        emit("job finished")
    except Exception as exc:  # noqa: BLE001 - surface any job failure to the UI
        log.exception("Job %s (%s) failed", job.id, kind)
        job.error = str(exc)
        job.status = "error"
        emit(f"error: {exc}")
    finally:
        job.finished_at = datetime.now(timezone.utc).isoformat()


def _start_job(kind: str, params: dict, settings: Settings, db: Database) -> _Job:
    job = _Job(id=uuid.uuid4().hex[:12], kind=kind, params=params)
    with _JOBS_LOCK:
        _JOBS[job.id] = job
    job.task = asyncio.create_task(_run_job(job, kind, settings, db))
    return job


class JobStartPayload(BaseModel):
    params: dict = Field(default_factory=dict)


@router.post("/jobs/{kind}")
async def start_job(kind: str, payload: JobStartPayload | None = None, request: Request = None):
    if kind not in _JOB_FUNCS:
        raise HTTPException(404, f"unknown job kind; allowed: {sorted(_JOB_FUNCS)}")
    settings: Settings = request.app.state.settings
    db: Database = request.app.state.db
    params = (payload.params if payload else {}) or {}
    job = _start_job(kind, params, settings, db)
    return job.to_dict()


@router.get("/jobs")
async def list_jobs():
    with _JOBS_LOCK:
        jobs = sorted(_JOBS.values(), key=lambda j: j.created_at, reverse=True)
        return [j.to_dict() for j in jobs]


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job.to_dict()


# ---------------------------------------------------------------------- config
class ConfigPayload(BaseModel):
    trading_mode: str | None = None
    active_prop_profile: str | None = None
    execution_driver: str | None = None
    market_symbol: str | None = None
    market_timeframe: str | None = None
    meta_model_prob_threshold: float | None = None
    min_leader_score: float | None = None
    max_live_leaders: int | None = None
    max_watch_leaders: int | None = None
    live_leader_pool_size: int | None = None
    myfxbook_poll_interval_seconds: int | None = None
    persist: bool = True


@router.get("/config")
async def get_config(request: Request):
    settings: Settings = request.app.state.settings
    profiles = load_profile_dir(_PROFILE_DIR)
    return {
        "trading_mode": settings.trading_mode.value,
        "active_prop_profile": settings.active_prop_profile,
        "execution_driver": settings.execution_driver,
        "market_symbol": settings.market_symbol,
        "market_timeframe": settings.market_timeframe,
        "meta_model_prob_threshold": settings.meta_model_prob_threshold,
        "min_leader_score": settings.min_leader_score,
        "max_live_leaders": settings.max_live_leaders,
        "max_watch_leaders": settings.max_watch_leaders,
        "live_leader_pool_size": settings.live_leader_pool_size,
        "myfxbook_poll_interval_seconds": settings.myfxbook_poll_interval_seconds,
        "effective_db_url": _mask_db_url(settings.effective_db_url),
        "profiles": sorted(profiles),
        "integrations": {
            "myfxbook": bool(settings.myfxbook_email),
            "metaapi": bool(settings.metaapi_token),
            "ea_key": bool(settings.ea_api_key),
            "mt5": MT5_AVAILABLE,
        },
    }


@router.put("/config")
async def update_config(payload: ConfigPayload, request: Request, session: AsyncSession = Depends(_session)):
    settings: Settings = request.app.state.settings
    changes: dict[str, Any] = {}
    env_updates: dict[str, str] = {}

    if payload.trading_mode is not None:
        allowed = {m.value for m in TradingMode}
        if payload.trading_mode not in allowed:
            raise HTTPException(422, f"invalid trading_mode; allowed: {sorted(allowed)}")
        if payload.trading_mode in (TradingMode.PROP_EVAL.value, TradingMode.LIVE.value) and not settings.metaapi_token:
            raise HTTPException(409, f"mode '{payload.trading_mode}' requires METAAPI_TOKEN")
        if payload.trading_mode == TradingMode.LIVE.value and _active_profile(request).name == "default_demo":
            raise HTTPException(409, "live mode refused while an internal/demo profile is active")
        settings.trading_mode = TradingMode(payload.trading_mode)
        changes["trading_mode"] = payload.trading_mode
        env_updates["TRADING_MODE"] = payload.trading_mode

    if payload.active_prop_profile is not None:
        if payload.active_prop_profile not in load_profile_dir(_PROFILE_DIR):
            raise HTTPException(422, f"unknown profile: {payload.active_prop_profile}")
        settings.active_prop_profile = payload.active_prop_profile
        request.app.state.profile = load_profile(_PROFILE_DIR / f"{payload.active_prop_profile}.yaml")
        changes["active_prop_profile"] = payload.active_prop_profile
        env_updates["ACTIVE_PROP_PROFILE"] = payload.active_prop_profile

    if payload.execution_driver is not None:
        driver = payload.execution_driver.strip().lower()
        if driver not in _ALLOWED_DRIVERS:
            raise HTTPException(422, f"invalid execution_driver; allowed: {sorted(_ALLOWED_DRIVERS)}")
        settings.execution_driver = driver
        changes["execution_driver"] = driver
        env_updates["EXECUTION_DRIVER"] = driver

    if payload.market_symbol is not None:
        symbol = payload.market_symbol.strip().upper()
        if not symbol:
            raise HTTPException(422, "market_symbol cannot be empty")
        settings.market_symbol = symbol
        changes["market_symbol"] = symbol
        env_updates["MARKET_SYMBOL"] = symbol

    if payload.market_timeframe is not None:
        timeframe = payload.market_timeframe.strip().upper()
        if timeframe not in _ALLOWED_TIMEFRAMES:
            raise HTTPException(422, f"invalid market_timeframe; allowed: {sorted(_ALLOWED_TIMEFRAMES)}")
        settings.market_timeframe = timeframe
        changes["market_timeframe"] = timeframe
        env_updates["MARKET_TIMEFRAME"] = timeframe

    if payload.meta_model_prob_threshold is not None:
        if not (0.55 <= payload.meta_model_prob_threshold <= 0.95):
            raise HTTPException(422, "meta_model_prob_threshold must be within [0.55, 0.95]")
        settings.meta_model_prob_threshold = payload.meta_model_prob_threshold
        changes["meta_model_prob_threshold"] = payload.meta_model_prob_threshold
        env_updates["META_MODEL_PROB_THRESHOLD"] = str(payload.meta_model_prob_threshold)

    if payload.min_leader_score is not None:
        if payload.min_leader_score < 0:
            raise HTTPException(422, "min_leader_score must be >= 0")
        settings.min_leader_score = payload.min_leader_score
        changes["min_leader_score"] = payload.min_leader_score
        env_updates["MIN_LEADER_SCORE"] = str(payload.min_leader_score)

    for name, value, env_key in (
        ("max_live_leaders", payload.max_live_leaders, "MAX_LIVE_LEADERS"),
        ("max_watch_leaders", payload.max_watch_leaders, "MAX_WATCH_LEADERS"),
        ("live_leader_pool_size", payload.live_leader_pool_size, "LIVE_LEADER_POOL_SIZE"),
        ("myfxbook_poll_interval_seconds", payload.myfxbook_poll_interval_seconds, "MYFXBOOK_POLL_INTERVAL_SECONDS"),
    ):
        if value is None:
            continue
        if name == "myfxbook_poll_interval_seconds":
            if value < 30:
                raise HTTPException(422, "myfxbook_poll_interval_seconds must be >= 30")
        elif value < 1:
            raise HTTPException(422, f"{name} must be >= 1")
        setattr(settings, name, value)
        changes[name] = value
        env_updates[env_key] = str(value)

    if not changes:
        raise HTTPException(422, "no mutable fields supplied")

    persisted = _persist_env(env_updates) if payload.persist else False
    await audit.record_audit(
        session,
        actor="user",
        action="config.update",
        detail={"changes": changes, "persisted": persisted},
    )
    await session.commit()
    return {"ok": True, "changes": changes, "persisted": persisted}


# ---------------------------------------------------------------------- status
@router.get("/status")
async def ops_status(request: Request):
    settings: Settings = request.app.state.settings
    ks = _active_kill_switch(request)
    profile = _active_profile(request)
    return {
        "trading_mode": settings.trading_mode.value,
        "execution_driver": settings.execution_driver,
        "symbol": settings.market_symbol,
        "timeframe": settings.market_timeframe,
        "active_profile": profile.name,
        "kill_switch_enabled": ks.enabled,
        "trading_allowed": ks.enabled,
        "copy_trading_enabled": _copy_trading(request).enabled,
        "mt5_available": MT5_AVAILABLE,
        "myfxbook_configured": bool(settings.myfxbook_email),
        "server_time_utc": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------- leaders
class LeaderAddPayload(BaseModel):
    id: str
    name: str = ""
    url: str = ""


class LeaderIdPayload(BaseModel):
    id: str


class LeaderLivePayload(BaseModel):
    ids: list[str]


def _roster_for(settings: Settings):
    from ...leaders.discovery import LeaderRoster

    return LeaderRoster(
        settings.roster_path,
        max_live=settings.live_leader_pool_size or settings.max_live_leaders,
        max_watch=settings.max_watch_leaders,
        min_observations=settings.leader_min_observations,
    )


@router.get("/leaders/roster-full")
async def leaders_roster_full(request: Request):
    settings: Settings = request.app.state.settings
    roster = _roster_for(settings)
    candidates = sorted(
        (roster.get(i) for i in roster.all_ids()),
        key=lambda c: (c.tier != "live", -max(c.watch_score, c.last_composite)),
    )
    return {
        "live": roster.live_ids(),
        "watch": roster.watch_ids(),
        "candidates": [c.to_dict() for c in candidates],
    }


@router.post("/leaders/add")
async def leaders_add(payload: LeaderAddPayload, request: Request, session: AsyncSession = Depends(_session)):
    settings: Settings = request.app.state.settings
    roster = _roster_for(settings)
    added = roster.add(payload.id.strip(), name=payload.name, source="manual", url=payload.url)
    await audit.record_audit(
        session, actor="user", action="leader.add",
        detail={"leader_id": payload.id, "added": added},
    )
    await session.commit()
    return {"ok": True, "added": added, "live": roster.live_ids(), "watch": roster.watch_ids()}


@router.post("/leaders/remove")
async def leaders_remove(payload: LeaderIdPayload, request: Request, session: AsyncSession = Depends(_session)):
    settings: Settings = request.app.state.settings
    roster = _roster_for(settings)
    removed = roster.remove(payload.id.strip())
    await audit.record_audit(
        session, actor="user", action="leader.remove",
        detail={"leader_id": payload.id, "removed": removed},
    )
    await session.commit()
    return {"ok": True, "removed": removed, "live": roster.live_ids(), "watch": roster.watch_ids()}


@router.post("/leaders/live")
async def leaders_set_live(payload: LeaderLivePayload, request: Request, session: AsyncSession = Depends(_session)):
    settings: Settings = request.app.state.settings
    roster = _roster_for(settings)
    ids = [i.strip() for i in payload.ids if i.strip()]
    roster.set_live(ids)
    await audit.record_audit(
        session, actor="user", action="leader.set_live",
        detail={"ids": ids},
    )
    await session.commit()
    return {"ok": True, "live": roster.live_ids(), "watch": roster.watch_ids()}


@router.get("/leaders/check/{leader_id}")
async def leaders_check(leader_id: str, request: Request):
    settings: Settings = request.app.state.settings
    if not settings.myfxbook_email:
        raise HTTPException(409, "MYFXBOOK_EMAIL / MYFXBOOK_PASSWORD not configured")
    from ...leaders.myfxbook import MyfxbookClient

    client = MyfxbookClient(settings.myfxbook_email, settings.myfxbook_password)
    try:
        stats = await client.get_account_stats(leader_id)
    except Exception as exc:  # noqa: BLE001 - surface upstream errors to the UI
        raise HTTPException(502, f"myfxbook lookup failed: {exc}") from exc
    return {
        "id": stats.account_id or leader_id,
        "name": stats.name,
        "gain_pct": stats.gain_pct,
        "drawdown_pct": stats.drawdown_pct,
        "trades": stats.trades,
        "verified_track_record": stats.verified_track_record,
        "verified_trading_privileges": stats.verified_trading_privileges,
    }


# ---------------------------------------------------------------------- audit
@router.get("/audit")
async def get_audit(limit: int = Query(100, ge=1, le=500), session: AsyncSession = Depends(_session)):
    stmt = select(SystemEvent).order_by(desc(SystemEvent.timestamp)).limit(limit)
    rows = (await session.execute(stmt)).scalars().all()
    return [
        {
            "id": r.id,
            "timestamp": r.timestamp.isoformat() if r.timestamp else "",
            "component": r.component,
            "event": r.event,
            "detail": r.detail,
        }
        for r in rows
    ]


@router.get("/risk-events")
async def get_risk_events(limit: int = Query(100, ge=1, le=500), session: AsyncSession = Depends(_session)):
    stmt = select(RiskEvent).order_by(desc(RiskEvent.timestamp)).limit(limit)
    rows = (await session.execute(stmt)).scalars().all()
    return [
        {
            "id": r.id,
            "timestamp": r.timestamp.isoformat() if r.timestamp else "",
            "event": r.event,
            "severity": r.severity,
            "drawdown": r.drawdown,
            "daily_loss": r.daily_loss,
            "action": r.action,
            "detail": r.detail,
        }
        for r in rows
    ]


# ---------------------------------------------------------------------- model
@router.get("/model")
async def get_model():
    model_dir = _MODEL_DIR
    best = model_dir / "best_model.txt"
    artifacts: list[dict[str, Any]] = []
    if model_dir.exists():
        for path in sorted(model_dir.iterdir()):
            if path.is_file():
                artifacts.append({"name": path.name, "size": path.stat().st_size})
    wf_report = None
    report_path = DATA_DIR / "wf_report.json"
    if report_path.exists():
        import json

        try:
            wf_report = json.loads(report_path.read_text(encoding="utf-8"))
        except ValueError:
            wf_report = None
    test_auc = None
    if isinstance(wf_report, list) and wf_report:
        test_auc = wf_report[0].get("test_auc")
    return {
        "model_dir": str(model_dir),
        "available": best.exists(),
        "winner": best.read_text(encoding="utf-8").strip() if best.exists() else None,
        "artifacts": artifacts,
        "wf_report": wf_report,
        "warning": "test AUC below 0.55 — discrimination is near random" if (test_auc is not None and test_auc < 0.55) else None,
    }


@router.get("/cli")
async def get_cli_reference():
    return _CLI_REFERENCE


@router.post("/setup-db")
async def setup_db(request: Request, session: AsyncSession = Depends(_session)):
    db: Database = request.app.state.db
    await db.create_all()
    await audit.record_audit(session, actor="user", action="ops.setup_db", detail={})
    await session.commit()
    return {"ok": True, "db_url": _mask_db_url(request.app.state.settings.effective_db_url)}


# ---------------------------------------------------------------------- manual order
class ManualOrderPayload(BaseModel):
    side: str
    volume: float | None = None
    stop_points: float | None = None
    take_profit_points: float | None = None
    symbol: str | None = None
    confirm: bool = False


@router.post("/order")
async def place_manual_order(payload: ManualOrderPayload, request: Request):
    if not payload.confirm:
        raise HTTPException(422, "confirmation required: set confirm=true to place a manual order")

    side = payload.side.strip().lower()
    if side not in ("buy", "sell"):
        raise HTTPException(422, "side must be 'buy' or 'sell'")

    settings: Settings = request.app.state.settings
    db: Database = request.app.state.db
    symbol = (payload.symbol or settings.market_symbol).strip().upper()
    profile = _active_profile(request)
    kill_switch = _active_kill_switch(request)

    stop_points = float(payload.stop_points) if payload.stop_points is not None else _DEFAULT_STOP_POINTS
    if stop_points <= 0:
        raise HTTPException(422, "stop_points must be > 0")

    candles = await asyncio.to_thread(_get_mt5_candles_sync, symbol, settings.market_timeframe, 5)
    if not candles:
        raise HTTPException(503, f"no market data available for {symbol}")
    price = float(candles[-1]["close"])

    acct = await asyncio.to_thread(_get_mt5_account_sync)
    positions = await asyncio.to_thread(_get_mt5_positions_sync, symbol)
    if acct.get("connected"):
        equity = float(acct.get("equity", profile.capital))
        balance = float(acct.get("balance", equity))
    else:
        equity = balance = profile.capital

    account = AccountState(
        equity=equity,
        high_water_mark=equity,
        daily_start_balance=balance,
        daily_pnl=0.0,
        open_positions=len(positions),
    )
    candidate = OrderCandidate(
        side=side,
        entry_price=price,
        stop_points=stop_points,
        source_branch="manual",
        meta_probability=0.5,
        symbol=symbol,
    )
    decision = _risk_engine(profile, kill_switch).approve(candidate, account)

    if not decision.approved:
        async with db.session_factory() as session:
            await audit.record_risk_event(
                session,
                event=f"manual_order.rejected.{decision.reason_code}",
                severity="WARNING",
                action="reject",
                detail={"symbol": symbol, "side": side, "reason": decision.reason, "entry": price},
            )
            await session.commit()
        return {
            "ok": False,
            "approved": False,
            "reason_code": decision.reason_code,
            "reason": decision.reason,
            "entry_price": price,
        }

    volume = float(payload.volume) if payload.volume is not None else decision.volume
    tp_multiple = 2.0
    take_profit_points = (
        float(payload.take_profit_points) if payload.take_profit_points is not None else tp_multiple * stop_points
    )
    stop_loss = price - stop_points if side == "buy" else price + stop_points
    take_profit = price + take_profit_points if side == "buy" else price - take_profit_points

    client = _build_client(settings, db)
    order_req = OrderRequest(
        symbol=symbol,
        side=side,
        volume=volume,
        entry_ref_price=price,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )
    result = await client.execute_with_retry(
        order_req, timeout_seconds=float(settings.ea_order_timeout_seconds)
    )

    async with db.session_factory() as session:
        order = OrderRecord(
            symbol=symbol,
            side=TradeSide(side),
            volume=volume,
            entry_price=result.fill_price or price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            status=("filled" if result.ok else "failed"),
            mode=settings.trading_mode.value,
            broker_order_id=result.broker_order_id or None,
            error=result.error,
        )
        session.add(order)
        await audit.record_audit(
            session,
            actor="user",
            action=("manual_order.filled" if result.ok else "manual_order.failed"),
            detail={
                "symbol": symbol,
                "side": side,
                "volume": volume,
                "entry": price,
                "sl": stop_loss,
                "tp": take_profit,
                "driver": settings.execution_driver,
                "broker_order_id": result.broker_order_id,
                "error": result.error,
            },
            commit=False,
        )
        await session.commit()
        order_id = order.id

    return {
        "ok": result.ok,
        "approved": True,
        "order_id": order_id,
        "volume": volume,
        "entry_price": price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "fill_price": result.fill_price,
        "broker_order_id": result.broker_order_id,
        "driver": settings.execution_driver,
        "error": result.error,
    }
