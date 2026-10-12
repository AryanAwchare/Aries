"""Stage B: Multi-Stage Composite Leader Scoring.

Weights (configurable from Settings):
- 20% Risk-adjusted return (return / max drawdown, not raw return)
- 20% Profit factor (scaled)
- 15% Drawdown quality (lower is better, scaled)
- 15% Consistency (penalizes reliance on single trade)
- 10% Trade expectancy (avg_win * win_rate - avg_loss * loss_rate)
- 10% Longevity (account age / track record length)
- 10% Recent performance (last 90 days performance)
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Sequence

import numpy as np

from ..config import Settings, get_settings
from ..logging import get_logger
from .models import LeaderProfile, LeaderScore, RawLeaderTrade

log = get_logger(__name__)


def compute_score(
    trades: Sequence[RawLeaderTrade],
    profile: LeaderProfile | None = None,
    settings: Settings | None = None,
    as_of: datetime | None = None,
) -> LeaderScore:
    """Multi-factor composite scoring for leaders that pass Stage A eligibility."""
    cfg = settings or get_settings()
    now = as_of or datetime.now(timezone.utc)

    closed = [t for t in trades if not t.is_open and t.profit is not None]
    if not closed:
        return LeaderScore()

    profits = np.array([t.profit for t in closed], dtype=float)
    wins = profits[profits > 0]
    losses = profits[profits < 0]

    gross_win = float(wins.sum())
    gross_loss = float(abs(losses.sum()))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else (3.0 if gross_win > 0 else 0.0)
    win_rate = float(len(wins) / len(profits))
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) else 0.0
    total = float(profits.sum())

    # Drawdown
    cum = np.cumsum(profits)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = float(dd.max()) if len(profits) else 0.0

    # 1. Risk-adjusted return component (0..1)
    # Ratio of total return to max drawdown
    if max_dd > 0:
        rar = total / max_dd
        rar_score = float(np.clip(rar / 5.0, 0.0, 1.0))
    else:
        rar_score = 1.0 if total > 0 else 0.0

    # 2. Profit factor component (0..1)
    pf_score = float(np.clip(profit_factor / 2.5, 0.0, 1.0))

    # 3. Drawdown quality component (0..1, lower DD is better)
    dd_ceiling = max(1.0, float(cfg.max_account_drawdown_ceiling))
    dd_pct = (max_dd / max(gross_win, 1.0)) * 100.0 if gross_win > 0 else max_dd
    dd_quality = float(np.clip(1.0 - (dd_pct / dd_ceiling), 0.0, 1.0))

    # 4. Consistency component (0..1, penalize reliance on single trade)
    top1_share = float(max(profits.max(), 0.0) / max(total, 1e-9)) if total > 0 else 1.0
    consistency = float(np.clip(1.0 - top1_share, 0.0, 1.0))

    # 5. Trade Expectancy component (0..1)
    # E = avg_win * win_rate - avg_loss * (1 - win_rate)
    expectancy_val = avg_win * win_rate - avg_loss * (1.0 - win_rate)
    expectancy_score = float(np.clip(expectancy_val / max(avg_win, 1e-4), 0.0, 1.0))

    # 6. Longevity component (0..1)
    account_days = profile.account_age if profile else len(closed) * 2
    longevity_score = float(np.clip(account_days / 730.0, 0.0, 1.0))  # 2 years = 1.0

    # 7. Recent performance component (last 90 days vs all-time)
    cutoff_90d = now - timedelta(days=90)
    recent_trades = [
        t for t in closed
        if t.opened_at is not None and (t.opened_at if t.opened_at.tzinfo else t.opened_at.replace(tzinfo=timezone.utc)) >= cutoff_90d
    ]
    if recent_trades:
        rec_profits = np.array([t.profit for t in recent_trades], dtype=float)
        rec_wins = rec_profits[rec_profits > 0]
        rec_wr = float(len(rec_wins) / len(rec_profits))
        recent_score = float(np.clip(rec_wr / 0.70, 0.0, 1.0))
    else:
        recent_score = 0.5  # Neutral

    # Weighted Composite Score using configurable Settings weights
    w_rar = cfg.weight_risk_adjusted_return
    w_pf = cfg.weight_profit_factor
    w_dd = cfg.weight_drawdown_quality
    w_cons = cfg.weight_consistency
    w_exp = cfg.weight_expectancy
    w_long = cfg.weight_longevity
    w_rec = cfg.weight_recent_performance

    composite = float(
        w_rar * rar_score
        + w_pf * pf_score
        + w_dd * dd_quality
        + w_cons * consistency
        + w_exp * expectancy_score
        + w_long * longevity_score
        + w_rec * recent_score
    )

    return LeaderScore(
        profit_factor=_finite(profit_factor),
        win_rate=win_rate,
        avg_win_pips=avg_win,
        avg_loss_pips=avg_loss,
        max_drawdown_pct=max_dd,
        total_pips=total,
        trade_count=len(profits),
        consistency_score=consistency,
        expectancy=expectancy_val,
        longevity=longevity_score,
        recent_score=recent_score,
        composite_score=composite,
        composite=composite,
    )


def _finite(x: float) -> float:
    return x if np.isfinite(x) else 1e6 if x > 0 else 0.0


def score_window_as_of(events_by_day: list[tuple[date, list[RawLeaderTrade]]]) -> dict[date, LeaderScore]:
    return {day: compute_score(trades) for day, trades in events_by_day}


def today_utc() -> date:
    return datetime.now(timezone.utc).date()