"""Command-line entrypoints: setup db, train, backtest, walk-forward, replay,
leader poll. Run with ``python -m app.cli <command>``.

Examples
--------
python -m app.cli setup-db
python -m app.cli poll-leaders --once
python -m app.cli backtest --synthetic
python -m app.cli walk-forward --synthetic
python -m app.cli replay --synthetic
"""
from __future__ import annotations

import asyncio
import argparse
from datetime import timedelta
from pathlib import Path

from .config import BASE_DIR, get_settings
from .logging import setup_logging


def _settings(args) -> None:
    setup_logging(log_dir=BASE_DIR / "logs")


async def cmd_setup_db(args) -> None:
    from .config import get_settings
    from .db.base import Database

    settings = get_settings()
    settings.ensure_dirs()
    db = Database(settings)
    await db.create_all()
    print(f"Database ready at {settings.database_url}")


async def cmd_backtest(args) -> None:
    from .backtest.runner import BacktestRunner, ProbScorer
    from .execution.killswitch import KillSwitch
    from .risk.engine import RiskEngine
    from .risk.limiter import RateLimiter
    from .risk.prop_profiles import load_profile

    settings = get_settings()
    profile = load_profile(settings.prop_profile_path)
    ohlcv = await _load_ohlcv(settings, args)
    risk = RiskEngine(profile, KillSwitch(True), RateLimiter())
    result = BacktestRunner(risk, scorer=ProbScorer()).run(ohlcv, start_capital=profile.capital)
    _print_backtest(result)


def _print_backtest(result) -> None:
    print("-" * 52)
    print(f"trades:                {result.n_trades}  (win {result.win_rate:.1%})")
    print(f"total return:          {result.total_return_pct:.2f}%")
    print(f"max drawdown:          {result.max_drawdown_pct:.2%}")
    print(f"sharpe (approx):       {result.sharpe:.2f}")
    print(f"profit factor:         {result.profit_factor:.2f}")


async def cmd_walk_forward(args) -> None:
    from .backtest.leakage import run_leakage_report
    from .backtest.walkforward import walk_forward, write_fold_report
    from .data.ohlcv import SyntheticProvider  # noqa: F401

    settings = get_settings()
    ohlcv = await _load_ohlcv(settings, args)

    leakage = run_leakage_report(ohlcv, settings.test_start)
    print(f"leakage report: causal={leakage.features_causal} kronos_overlap={leakage.kronos_test_overlap}")
    for note in leakage.notes:
        print("  -", note)

    out_dir = BASE_DIR / "data" / "models"
    folds = walk_forward(
        ohlcv,
        train_end=settings.train_end,
        val_end=settings.validation_end,
        test_end=settings.test_end,
        out_dir=out_dir,
    )
    write_fold_report(folds, BASE_DIR / "data" / "wf_report.json")
    for fold in folds:
        print(f"{fold.fold_name}: winner={fold.winner} logloss={fold.test_logloss:.4f} auc={fold.test_auc:.3f}")


async def cmd_replay(args) -> None:
    from .backtest.replay import run_replay
    from .execution.base import PaperBroker
    from .execution.executor import Executor
    from .execution.killswitch import KillSwitch
    from .metamodel.predict import MetaModelPredictor
    from .risk.engine import RiskEngine
    from .risk.limiter import RateLimiter
    from .risk.prop_profiles import load_profile
    from .strategy.engine import StrategyEngine

    settings = get_settings()
    ohlcv = await _load_ohlcv(settings, args)
    profile = load_profile(settings.prop_profile_path)
    db = __import__("app.db.base", fromlist=["Database"]).Database(settings)
    await db.create_all()
    kill = KillSwitch(True)
    risk = RiskEngine(profile, kill, RateLimiter())
    predictor = MetaModelPredictor(BASE_DIR / "data" / "models", threshold=settings.meta_model_prob_threshold)
    strategy = StrategyEngine(settings, predictor)
    executor = Executor(settings, risk, PaperBroker(), kill)
    stats = await run_replay(db, strategy, risk, executor, ohlcv)
    print(f"replay: bars={stats.bars_seen} signals={stats.signals_fired} "
          f"filled={stats.orders_filled} rejected={stats.rejected}")


async def cmd_poll_once(args) -> None:
    from .config import get_settings
    from .leaders.myfxbook import MyfxbookClient
    from .scheduler.jobs import poll_leader_trades
    from .db.base import Database

    settings = get_settings()
    if not settings.myfxbook_email:
        raise SystemExit("MYFXBOOK_EMAIL / MYFXBOOK_PASSWORD not configured")
    db = Database(settings)
    await db.create_all()
    client = MyfxbookClient(settings.myfxbook_email, settings.myfxbook_password)
    inserted = await poll_leader_trades(settings, db.session_factory, client)
    print(f"polled leaders; inserted {inserted} events")


async def _load_ohlcv(settings, args):
    from datetime import datetime, timezone

    from .data.ohlcv import CSVProvider, OHLCVRequest, SyntheticProvider

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    if args.synthetic:
        provider = SyntheticProvider(seed=args.seed)
    else:
        provider = CSVProvider(str(BASE_DIR / "data" / "raw"))
    df = await provider.get_ohlcv(
        OHLCVRequest(symbol=settings.market_symbol, timeframe=settings.market_timeframe, start=start, end=end)
    )
    df.attrs["symbol"] = settings.market_symbol
    print(f"loaded {len(df)} bars  {df.index[0]} .. {df.index[-1]}")
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gold-trader")
    sub = parser.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("setup-db", help="create the database schema")

    sp = sub.add_parser("poll-leaders", help="poll Myfxbook leaders once")
    sp.add_argument("--once", action="store_true")

    sp = sub.add_parser("backtest", help="run the hybrid-signal backtest")
    sp.add_argument("--synthetic", action="store_true", help="use deterministic synthetic data")
    sp.add_argument("--days", type=int, default=730, help="lookback window in days")
    sp.add_argument("--seed", type=int, default=42)

    sp = sub.add_parser("walk-forward", help="walk-forward train/validate/test")
    sp.add_argument("--synthetic", action="store_true")
    sp.add_argument("--days", type=int, default=800)
    sp.add_argument("--seed", type=int, default=42)

    sp = sub.add_parser("replay", help="replay historical bars through the live stack")
    sp.add_argument("--synthetic", action="store_true")
    sp.add_argument("--days", type=int, default=180)
    sp.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _settings(args)
    if args.command == "setup-db":
        asyncio.run(cmd_setup_db(args))
    elif args.command == "poll-leaders":
        asyncio.run(cmd_poll_once(args))
    elif args.command == "backtest":
        asyncio.run(cmd_backtest(args))
    elif args.command == "walk-forward":
        asyncio.run(cmd_walk_forward(args))
    elif args.command == "replay":
        asyncio.run(cmd_replay(args))


if __name__ == "__main__":
    main()