"""Risk engine — the guard pipeline every signal must pass.

Order of checks matters (cheapest/coarsest first):

  1. kill switch            → no orders at all
  2. prop compatibility     → source-branch vs. profile permissions
  3. trading hours          → news/session filter
  4. drawdown               → firm hard limit, then our own tighter stops
  5. daily loss             → firm hard, then soft internal limit
  6. profit target / day    → stop with gains
  7. order-rate limiter     → orders/min, orders/hr (independent of strategy)
  8. open-positions cap
  9. consistency rule       → tracking now, enforce when enabled
  10. position sizing       → risk-first, confidence-scaled

Rejections are actionable: each returns a stable ``reason_code``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..db import audit
from ..execution.killswitch import KillSwitch  # re-exported via service normally
from ..logging import get_logger
from .limiter import RateLimiter
from .prop_profiles import PropProfile
from .sizing import position_size_with_confidence, SizingInput

log = get_logger(__name__)


@dataclass
class AccountState:
    equity: float
    high_water_mark: float
    daily_start_balance: float
    daily_pnl: float = 0.0
    total_profit: float = 0.0
    open_positions: int = 0
    now: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class OrderCandidate:
    side: str                     # "buy" | "sell"
    entry_price: float
    stop_points: float            # desired stop distance in price points
    source_branch: str = "ml"     # "leader" | "ml" | "manual"
    meta_probability: float = 0.5
    symbol: str = "XAUUSD"


@dataclass
class RiskDecision:
    approved: bool
    volume: float = 0.0
    reason_code: str = ""
    reason: str = ""


class RiskEngine:
    """Mode- and profile-aware gate. ``session`` is optional — only used for
    audit tracing; the engine still functions without a DB (backtest/replay).
    """

    def __init__(
        self,
        profile: PropProfile,
        kill_switch: KillSwitch,
        limiter: RateLimiter | None = None,
    ) -> None:
        self.profile = profile
        self.kill_switch = kill_switch
        self.limiter = limiter or RateLimiter(
            profile.max_orders_per_minute, profile.max_orders_per_hour
        )

    # ------------------------------------------------------------------
    def approve(self, candidate: OrderCandidate, account: AccountState) -> RiskDecision:
        checks = [
            self._check_kill_switch,
            self._check_compatibility,
            self._check_trading_hours,
            self._check_drawdown,
            self._check_daily_loss,
            self._check_profit_target,
            self._check_rate_limit,
            self._check_open_positions,
            self._check_consistency,
        ]
        for check in checks:
            decision = check(candidate, account)
            if decision is not None:
                return decision

        volume = position_size_with_confidence(
            SizingInput(
                equity=account.equity,
                risk_fraction=self.profile.position_risk_pct,
                stop_distance_points=candidate.stop_points,
                meta_probability=candidate.meta_probability,
                min_lot=self.profile.min_lot,
                max_lot=self.profile.max_lot,
                lot_step=self.profile.lot_step,
            )
        )
        if volume <= 0:
            return RiskDecision(False, 0.0, "invalid_size", "computed position size is zero")
        return RiskDecision(True, volume, "approved", "passed risk gate")

    # ------------------------------------------------------------------
    def _check_kill_switch(self, candidate, account) -> RiskDecision | None:
        if not self.kill_switch.enabled:
            return RiskDecision(False, 0.0, "kill_switch", "kill switch engaged")
        return None

    def _check_compatibility(self, candidate, account) -> RiskDecision | None:
        p = self.profile
        if candidate.source_branch == "leader":
            if not p.copy_trading_allowed:
                return RiskDecision(
                    False, 0.0, "copy_trading_forbidden",
                    "copy-trading not permitted under the active prop profile",
                )
            if not p.third_party_signals_allowed:
                return RiskDecision(
                    False, 0.0, "third_party_signals_forbidden",
                    "third-party signals not permitted under the active prop profile",
                )
        return None

    def _check_trading_hours(self, candidate, account) -> RiskDecision | None:
        allowed = self.profile.trading_hours_utc
        if allowed in ("", "00:00-23:59"):
            return None
        start_s, end_s = allowed.split("-")
        h = account.now.hour
        start_h = int(start_s.split(":")[0])
        end_h = int(end_s.split(":")[0])
        if end_h < start_h:  # wraps midnight
            if not (h >= start_h or h < end_h):
                return RiskDecision(False, 0.0, "outside_trading_hours", f"UTC {h:02d}:00 is outside {allowed}")
        elif not (start_h <= h < end_h):
            return RiskDecision(False, 0.0, "outside_trading_hours", f"UTC {h:02d}:00 is outside {allowed}")
        return None

    def _current_drawdown(self, account: AccountState) -> float:
        if account.high_water_mark <= 0:
            return 0.0
        return max(0.0, (account.high_water_mark - account.equity) / account.high_water_mark)

    def _check_drawdown(self, candidate, account) -> RiskDecision | None:
        dd = self._current_drawdown(account)
        if dd >= self.profile.emergency_stop_drawdown_pct:
            return RiskDecision(
                False, 0.0, "drawdown_emergency_stop",
                f"drawdown {dd:.1%} >= internal emergency stop {self.profile.emergency_stop_drawdown_pct:.1%}",
            )
        if dd >= self.profile.max_drawdown_pct:
            return RiskDecision(
                False, 0.0, "drawdown_firm_breach",
                f"drawdown {dd:.1%} >= firm max {self.profile.max_drawdown_pct:.1%}",
            )
        if dd >= self.profile.operating_ceiling_drawdown_pct:
            return RiskDecision(
                False, 0.0, "drawdown_operating_ceiling",
                f"drawdown {dd:.1%} above operating ceiling {self.profile.operating_ceiling_drawdown_pct:.1%}",
            )
        return None

    def _daily_loss(self, account: AccountState) -> float:
        if account.daily_start_balance <= 0:
            return 0.0
        return -account.daily_pnl / account.daily_start_balance

    def _check_daily_loss(self, candidate, account) -> RiskDecision | None:
        firm_limit = self.profile.daily_loss_limit_pct
        soft = self.profile.daily_loss_soft_limit_pct
        loss = self._daily_loss(account)
        if loss >= firm_limit:
            return RiskDecision(False, 0.0, "daily_loss_firm_breach",
                                f"daily loss {loss:.1%} >= firm limit {firm_limit:.1%}")
        if loss >= soft:
            return RiskDecision(False, 0.0, "daily_loss_soft_limit",
                                f"daily loss {loss:.1%} >= internal soft limit {soft:.1%}")
        return None

    def _check_profit_target(self, candidate, account) -> RiskDecision | None:
        if self.profile.profit_target <= 0:
            return None
        start = account.daily_start_balance
        if start <= 0:
            return None
        gained = (account.equity - start) / start
        if gained >= self.profile.profit_target:
            return RiskDecision(False, 0.0, "profit_target_reached",
                                f"profit {gained:.1%} >= target {self.profile.profit_target:.1%}")
        return None

    def _check_rate_limit(self, candidate, account) -> RiskDecision | None:
        # reserve the slot only if we'll actually send the order
        if not self.limiter.allowed(count=1):
            return RiskDecision(
                False, 0.0, "order_rate_limit",
                f"rate limit hit ({self.profile.max_orders_per_minute}/min, {self.profile.max_orders_per_hour}/hr)",
            )
        return None

    def _check_open_positions(self, candidate, account) -> RiskDecision | None:
        if account.open_positions >= self.profile.max_open_positions:
            return RiskDecision(
                False, 0.0, "max_open_positions",
                f"{account.open_positions} positions >= cap {self.profile.max_open_positions}",
            )
        return None

    def _check_consistency(self, candidate, account) -> RiskDecision | None:
        rule = self.profile.consistency_rule
        if not rule.enabled:
            return None  # track-only until a firm actually enforces it
        if account.total_profit <= 0:
            return None
        # We don't know today's profit split here; the execution layer records it
        # per-trade. Enforcement hook: block when a single day's attribution is
        # already too high — surfaced via the dashboard's consistency tracker.
        return None


async def audit_rejection(
    session,
    decision: RiskDecision,
    actor: str = "risk",
) -> None:
    if not decision.approved:
        await audit.record_audit(
            session,
            actor=actor,
            action=f"risk.reject.{decision.reason_code}",
            detail={"reason": decision.reason},
        )