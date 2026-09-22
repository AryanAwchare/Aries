"""Phase 2 — daily leader scoring, plus DB persistence helpers.

Scoring is deliberately stateless (pure functions over closed trade lists)
so it can be unit tested without a database; ``persist`` then stores the
result.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np

from ..logging import get_logger
from .models import LeaderScore, RawLeaderTrade

log = get_logger(__name__)


def compute_score(trades: list[RawLeaderTrade]) -> LeaderScore:
    """Composite score for a leader from their closed trades.

    Returns a LeaderScore. Empty/invalid input yields all-zero score.
    """
    closed = [t for t in trades if not t.is_open and t.profit is not None]
    if not closed:
        return LeaderScore()

    profits = np.array([t.profit for t in closed], dtype=float)
    wins = profits[profits > 0]
    losses = profits[profits < 0]

    gross_win = float(wins.sum())
    gross_loss = float(abs(losses.sum()))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    win_rate = float(len(wins) / len(profits))
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    total = float(profits.sum())

    # max drawdown on cumulative equity (normalised units of |avg loss|)
    cum = np.cumsum(profits)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = float(dd.max()) if len(profits) else 0.0

    # consistency: penalise reliance on a single trade / single day
    top1_share = float(max(profits.max(), 0.0) / max(total, 1e-9))
    consistency = float(np.clip(1.0 - top1_share, 0.0, 1.0))

    composite = _composite(profit_factor, win_rate, consistency)
    return LeaderScore(
        profit_factor=_finite(profit_factor),
        win_rate=win_rate,
        avg_win_pips=avg_win,
        avg_loss_pips=avg_loss,
        max_drawdown_pct=max_dd,
        total_pips=total,
        trade_count=len(profits),
        consistency_score=consistency,
        composite=composite,
    )


def _composite(pf: float, wr: float, consistency: float) -> float:
    pf_sat = np.clip(pf / 2.0, 0.0, 1.0) if np.isfinite(pf) else 1.0
    return float(0.5 * pf_sat + 0.3 * wr + 0.2 * consistency)


def _finite(x: float) -> float:
    return x if np.isfinite(x) else 1e6 if x > 0 else 0.0


def score_window_as_of(events_by_day: list[tuple[date, list[RawLeaderTrade]]]) -> dict[date, LeaderScore]:
    """Score per day over a window — source for the ``leader_daily_scores`` table."""
    return {day: compute_score(trades) for day, trades in events_by_day}


def today_utc() -> date:
    return datetime.now(timezone.utc).date()