"""Stage A: Leader Eligibility Gate.

Ensures only verified, mature, active, non-toxic (no grid / martingale) traders
pass to Stage B composite scoring and consideration for the live tier.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Sequence

import numpy as np

from ..config import Settings, get_settings
from ..logging import get_logger
from .models import EligibilityResult, LeaderProfile, RawLeaderTrade

log = get_logger(__name__)


def detect_martingale(trades: Sequence[RawLeaderTrade], threshold_multiplier: float = 1.4, min_streak: int = 2) -> bool:
    """Detect if position volume escalates aggressively following losing trades.

    Returns True if a clear martingale sizing pattern is identified.
    """
    closed = [t for t in trades if not t.is_open and t.profit is not None and t.opened_at is not None]
    if len(closed) < 10:
        return False

    # Sort trades chronologically
    sorted_trades = sorted(closed, key=lambda t: t.opened_at or datetime.min.replace(tzinfo=timezone.utc))

    martingale_escalations = 0
    consecutive_losses = 0

    for i in range(1, len(sorted_trades)):
        prev = sorted_trades[i - 1]
        curr = sorted_trades[i]

        if prev.profit is not None and prev.profit < 0:
            consecutive_losses += 1
            if curr.volume >= prev.volume * threshold_multiplier and consecutive_losses >= min_streak:
                martingale_escalations += 1
        else:
            consecutive_losses = 0

    # If more than 3 distinct martingale escalations occur, flag the account
    return martingale_escalations >= 3


def detect_grid(trades: Sequence[RawLeaderTrade], max_simultaneous_same_side: int = 5) -> bool:
    """Detect grid trading pattern: clustering of multiple open positions in the same direction."""
    if not trades:
        return False

    # Check for excessive simultaneous trades opened close together with no stops
    open_trades = [t for t in trades if t.is_open]
    if len(open_trades) >= max_simultaneous_same_side:
        buys = sum(1 for t in open_trades if t.side == "buy")
        sells = sum(1 for t in open_trades if t.side == "sell")
        if buys >= max_simultaneous_same_side or sells >= max_simultaneous_same_side:
            return True

    return False


class LeaderEligibilityGate:
    """Evaluates whether a leader passes all mandatory Stage A filters."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def evaluate(
        self,
        profile: LeaderProfile,
        trades: Sequence[RawLeaderTrade],
        as_of: datetime | None = None,
    ) -> EligibilityResult:
        now = as_of or datetime.now(timezone.utc)
        reasons: list[str] = []

        # 1. Myfxbook Track Record verification (distinct from trading privileges)
        if not profile.verified_track_record:
            reasons.append("unverified_track_record")

        # 2. Actually trades target symbol (XAUUSD)
        symbol_trades = [t for t in trades if t.symbol == self.settings.market_symbol]
        if not symbol_trades:
            reasons.append(f"no_trades_in_{self.settings.market_symbol}")

        # 3. Minimum account age (default 180 days / 6 months)
        if profile.account_age < self.settings.min_account_age_days:
            reasons.append(
                f"account_age_below_min:{profile.account_age}d<{self.settings.min_account_age_days}d"
            )

        # 4. Minimum number of closed trades (default 100)
        closed_trades = [t for t in symbol_trades if not t.is_open]
        if len(closed_trades) < self.settings.min_closed_trades:
            reasons.append(
                f"insufficient_closed_trades:{len(closed_trades)}<{self.settings.min_closed_trades}"
            )

        # 5. Martingale & Grid pattern rejection
        if detect_martingale(symbol_trades):
            reasons.append("martingale_pattern_detected")
        if detect_grid(trades):
            reasons.append("grid_pattern_detected")

        # 6. Maximum drawdown below ceiling (default 25%)
        # Calculate max drawdown from trades if available
        if closed_trades:
            profits = np.array([t.profit or 0.0 for t in closed_trades], dtype=float)
            cum = np.cumsum(profits)
            peak = np.maximum.accumulate(cum)
            dd = peak - cum
            max_dd_val = float(dd.max()) if len(profits) else 0.0
            # Normalize DD pct if starting capital known or compare to profit
            if max_dd_val > 0 and len(profits) > 0:
                gross_profit = float(profits[profits > 0].sum())
                if gross_profit > 0 and (max_dd_val / gross_profit * 100.0) > (self.settings.max_account_drawdown_ceiling * 2):
                    reasons.append(f"excessive_drawdown_detected")

        # 7. Sufficient recent activity (traded within last N days)
        if symbol_trades:
            trade_times = [
                t.opened_at
                for t in symbol_trades
                if t.opened_at is not None
            ]
            if trade_times:
                most_recent = max(trade_times)
                if most_recent.tzinfo is None:
                    most_recent = most_recent.replace(tzinfo=timezone.utc)
                inactivity_delta = now - most_recent
                if inactivity_delta > timedelta(days=self.settings.max_inactivity_days):
                    reasons.append(
                        f"inactive_too_long:{inactivity_delta.days}d>{self.settings.max_inactivity_days}d"
                    )
        else:
            reasons.append("no_trade_history")

        is_eligible = len(reasons) == 0
        if not is_eligible:
            log.info("Leader %s failed eligibility gate: %s", profile.id, reasons)

        return EligibilityResult(
            eligible=is_eligible,
            rejection_reasons=reasons,
            metrics={
                "closed_trades": len(closed_trades),
                "account_age": profile.account_age,
                "verified_track_record": profile.verified_track_record,
            },
        )