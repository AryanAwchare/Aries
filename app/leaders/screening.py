"""Phase 2.1: Myfxbook leader discovery & gold-qualification screening.

Myfxbook has no public discovery API, so the pool is grown from three sources:
  1. accounts already on the user's Myfxbook watch list (bulk ``get-watched-accounts``),
  2. a plaintext seed file (``data/leader_seeds.txt``, one account id per line), and
  3. ids already in the roster.

Every candidate is then *screened* against gold-specific filters (age, trade
count, drawdown, profit factor, real-account, verification, martingale/grid)
and, if qualified, written to the watch tier of ``data/roster.json``. Promotion
to the live tier is handled by :meth:`LeaderRoster.rotate`.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..config import Settings
from ..logging import get_logger
from .classifier import classify_behavior
from .eligibility import detect_grid, detect_martingale
from .models import LeaderProfile, RawLeaderTrade
from .scoring import compute_score

log = get_logger(__name__)


@dataclass
class GoldScreenConfig:
    """Filters a Myfxbook account must pass to qualify as a gold leader."""

    market_symbol: str = "XAUUSD"
    min_account_age_days: int = 365
    min_closed_trades: int = 200
    max_drawdown_pct: float = 20.0
    min_profit_factor: float = 1.5
    require_real_account: bool = True
    require_verified_track_record: bool = True
    max_inactivity_days: int = 30

    @classmethod
    def from_settings(cls, s: Settings) -> GoldScreenConfig:
        return cls(
            market_symbol=s.market_symbol,
            min_account_age_days=s.leader_min_account_age_days,
            min_closed_trades=s.leader_min_discovery_trades,
            max_drawdown_pct=s.leader_max_drawdown_pct,
            min_profit_factor=s.leader_min_profit_factor,
            require_real_account=s.leader_require_real_account,
            require_verified_track_record=True,
            max_inactivity_days=s.max_inactivity_days,
        )


@dataclass
class AccountScreening:
    leader_id: str
    qualified: bool = False
    reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    strategy_type: str = "unknown"
    composite: float = 0.0

    def summary(self) -> str:
        if self.qualified:
            return f"QUALIFIED  composite={self.composite:.3f}  {self.metrics}"
        return f"rejected   reasons={self.reasons}"


def _profit_factor(profits: np.ndarray) -> float:
    if profits.size == 0:
        return 0.0
    wins = profits[profits > 0].sum()
    losses = abs(profits[profits < 0].sum())
    if losses > 0:
        return float(wins / losses)
    return 3.0 if wins > 0 else 0.0


def _drawdown_pct(profits: np.ndarray, starting_equity: float = 10_000.0) -> float:
    """Peak-to-trough drawdown of the cumulative PnL curve, as % of equity."""
    if profits.size == 0:
        return 0.0
    cum = np.cumsum(profits)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    dd = peak - cum
    base = max(starting_equity, float(peak.max()) if peak.size else starting_equity, 1.0)
    return float(dd.max() / base * 100.0)


def _age_days_from_trades(closed: Sequence[RawLeaderTrade], now: datetime) -> int:
    """Days since the account's first trade (a lower bound on real age)."""
    times = [t.opened_at for t in closed if t.opened_at is not None]
    if not times:
        return 0
    earliest = min(times)
    if earliest.tzinfo is None:
        earliest = earliest.replace(tzinfo=timezone.utc)
    return max(0, (now - earliest).days)


def _days_since_latest(closed: Sequence[RawLeaderTrade], now: datetime) -> int:
    """Days since the account's most recent trade (inactivity measure)."""
    times = [t.opened_at for t in closed if t.opened_at is not None]
    if not times:
        return 0
    latest = max(times)
    if latest.tzinfo is None:
        latest = latest.replace(tzinfo=timezone.utc)
    return max(0, (now - latest).days)


def screen_history(
    trades: Sequence[RawLeaderTrade],
    cfg: GoldScreenConfig,
    *,
    verified_track_record: bool = True,
    is_real_account: bool = True,
    account_age_days: int | None = None,
    reported_drawdown_pct: float | None = None,
    as_of: datetime | None = None,
) -> AccountScreening:
    """Pure screening over an account's trade history (unit-testable)."""
    now = as_of or datetime.now(timezone.utc)
    reasons: list[str] = []

    gold = [t for t in trades if t.symbol.upper() == cfg.market_symbol.upper()]
    closed = [t for t in gold if not t.is_open and t.profit is not None]
    profits = np.array([t.profit for t in closed], dtype=float) if closed else np.array([])

    age = account_age_days if account_age_days is not None else _age_days_from_trades(closed, now)
    dd_pct = reported_drawdown_pct if reported_drawdown_pct is not None else _drawdown_pct(profits)
    pf = _profit_factor(profits)
    wins = profits[profits > 0]
    losses = profits[profits < 0]
    win_rate = float(len(wins) / len(profits)) if profits.size else 0.0
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(abs(losses.mean())) if losses.size else 0.0
    last_trade_days = _days_since_latest(closed, now)

    if not gold:
        reasons.append(f"no_{cfg.market_symbol}_trades")
    if len(closed) < cfg.min_closed_trades:
        reasons.append(f"insufficient_trades:{len(closed)}<{cfg.min_closed_trades}")
    if age < cfg.min_account_age_days:
        reasons.append(f"account_too_young:{age}d<{cfg.min_account_age_days}d")
    if dd_pct > cfg.max_drawdown_pct:
        reasons.append(f"drawdown_too_high:{dd_pct:.1f}%>{cfg.max_drawdown_pct}%")
    if pf < cfg.min_profit_factor:
        reasons.append(f"profit_factor_low:{pf:.2f}<{cfg.min_profit_factor}")
    if cfg.require_real_account and not is_real_account:
        reasons.append("demo_account")
    if cfg.require_verified_track_record and not verified_track_record:
        reasons.append("unverified_track_record")
    if detect_martingale(gold):
        reasons.append("martingale_pattern")
    if detect_grid(gold):
        reasons.append("grid_pattern")
    if closed and last_trade_days > cfg.max_inactivity_days:
        reasons.append(f"inactive:{last_trade_days}d>{cfg.max_inactivity_days}d")

    profile = LeaderProfile(
        id="candidate",
        name="candidate",
        account_age=age,
        verified_track_record=verified_track_record,
        symbol=cfg.market_symbol,
    )
    score = compute_score(gold, profile) if closed else None
    strategy = classify_behavior(closed) if closed else "unknown"

    return AccountScreening(
        leader_id="",
        qualified=not reasons,
        reasons=reasons,
        strategy_type=strategy,
        composite=float(score.composite_score) if score else 0.0,
        metrics={
            "gold_trades": len(gold),
            "closed_trades": len(closed),
            "account_age_days": age,
            "profit_factor": round(pf, 3),
            "win_rate": round(win_rate, 4),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "drawdown_pct": round(dd_pct, 2),
            "is_real_account": is_real_account,
            "verified_track_record": verified_track_record,
            "last_trade_days_ago": last_trade_days,
        },
    )


async def screen_candidate(
    client, leader_id: str, cfg: GoldScreenConfig, *, as_of: datetime | None = None
) -> AccountScreening:
    """Fetch aggregate stats + history for one account and screen it."""
    stats = None
    try:
        stats = await client.get_account_stats(leader_id)
    except Exception as exc:  # network / API hiccup — fall back to history-only
        log.warning("account-stats fetch failed for %s: %s", leader_id, exc)

    try:
        history = await client.get_history(leader_id)
    except Exception as exc:
        log.warning("history fetch failed for %s: %s", leader_id, exc)
        history = []

    verified_tr = True
    is_real = True
    age_days: int | None = None
    reported_dd: float | None = None
    if stats is not None and getattr(stats, "account_id", ""):
        verified_tr = stats.verified_track_record
        reported_dd = stats.drawdown_pct or None
        raw = getattr(stats, "raw", {}) or {}
        acc_type = str(raw.get("type", "")).lower()
        if acc_type:
            is_real = acc_type.startswith("real")
        age_days = _age_days_from_meta(raw)

    result = screen_history(
        history,
        cfg,
        verified_track_record=verified_tr,
        is_real_account=is_real,
        account_age_days=age_days,
        reported_drawdown_pct=reported_dd,
        as_of=as_of,
    )
    result.leader_id = leader_id
    return result


def _age_days_from_meta(raw: dict) -> int | None:
    first = raw.get("firstUpdateDate") or raw.get("firstTradeDate")
    if not first:
        return None
    try:
        ts = int(first) / 1000.0
        return max(0, (datetime.now(timezone.utc) - datetime.fromtimestamp(ts, tz=timezone.utc)).days)
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def load_seed_ids(path: Path) -> list[str]:
    """Read candidate ids from a plaintext seed file (one per line).

    Supports ``#`` comments and ``id`` or ``id, name`` lines.
    """
    if not path.exists():
        return []
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line.split(",", 1)[0].strip())
    return list(dict.fromkeys(i for i in ids if i))


async def discover_leaders(
    settings: Settings,
    client,
    roster,
    *,
    seed_ids: Sequence[str] | None = None,
    apply: bool = False,
    rotate: bool = False,
    as_of: datetime | None = None,
) -> dict:
    """Screen the candidate pool and (optionally) persist qualified leaders.

    Candidate pool = roster ids + env leader ids + seed file + Myfxbook watch list.
    """
    cfg = GoldScreenConfig.from_settings(settings)

    seeds = list(seed_ids or [])
    seeds += settings.leader_ids
    seeds += load_seed_ids(settings.leader_seed_path)

    candidate_ids: list[str] = list(dict.fromkeys([*roster.all_ids(), *seeds]))

    # Best-effort: include everything on the user's Myfxbook watch list.
    try:
        watched = await client.get_watched_accounts()
        candidate_ids += [w.account_id for w in watched if w.account_id]
        candidate_ids = list(dict.fromkeys(candidate_ids))
    except Exception as exc:
        log.warning("watch-list fetch failed during discovery: %s", exc)

    results: list[dict] = []
    qualified: list[str] = []
    rejected: dict[str, list[str]] = {}

    for leader_id in candidate_ids:
        screen = await screen_candidate(client, leader_id, cfg, as_of=as_of)
        row = {
            "id": leader_id,
            "qualified": screen.qualified,
            "reasons": screen.reasons,
            "composite": round(screen.composite, 4),
            "strategy": screen.strategy_type,
            "metrics": screen.metrics,
        }
        results.append(row)
        if screen.qualified:
            qualified.append(leader_id)
            if apply:
                _upsert_candidate(roster, leader_id, screen)
        else:
            rejected[leader_id] = screen.reasons

    rotation = None
    if apply:
        if rotate:
            rotation = roster.rotate()
        roster.save()

    return {
        "candidates": len(candidate_ids),
        "qualified": qualified,
        "rejected": rejected,
        "results": results,
        "rotation": None if rotation is None else {
            "promoted": rotation.promoted,
            "demoted": rotation.demoted,
            "note": rotation.note,
        },
        "applied": apply,
    }


def _upsert_candidate(roster, leader_id: str, screen: AccountScreening) -> None:
    c = roster.get(leader_id)
    if c is None:
        roster.add(
            leader_id,
            source="discovery",
            verified_track_record=bool(screen.metrics.get("verified_track_record", True)),
        )
        c = roster.get(leader_id)
    if c is None:
        return
    c.is_eligible = True
    c.last_composite = screen.composite
    c.watch_score = max(c.watch_score, screen.composite)
    c.strategy_type = screen.strategy_type or c.strategy_type
    c.observations = max(c.observations, 1)
    c.last_seen_at = datetime.now(timezone.utc).isoformat()
