"""Stage F: Live Loss-Rate Circuit Breaker.

Continuously monitors live-tier leaders in real-time. If a live leader
breaches rolling loss rate or drawdown limits (e.g., 3 consecutive losses
or live drawdown > X%), immediately demotes them from the live tier
without waiting for the scheduled rescore, logging a risk_event.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from ..config import Settings, get_settings
from ..db.audit import record_risk_event
from ..logging import get_logger
from .discovery import LeaderRoster
from .models import RawLeaderTrade

log = get_logger(__name__)


@dataclass
class LeaderLiveState:
    leader_id: str
    promoted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    consecutive_losses: int = 0
    peak_equity: float = 0.0
    current_equity: float = 0.0
    max_drawdown_pct: float = 0.0
    demoted: bool = False
    demotion_reason: str = ""


class LeaderLiveCircuitBreaker:
    """Monitors live trade stream for leader risk breaches."""

    def __init__(
        self,
        roster: LeaderRoster,
        settings: Settings | None = None,
        base_capital: float = 10000.0,
    ) -> None:
        self.roster = roster
        self.settings = settings or get_settings()
        self.base_capital = base_capital or self.settings.risk_capital_base
        self._states: dict[str, LeaderLiveState] = {}

    def get_or_create_state(self, leader_id: str) -> LeaderLiveState:
        if leader_id not in self._states:
            self._states[leader_id] = LeaderLiveState(leader_id=leader_id)
        return self._states[leader_id]

    async def update_trade_and_check(
        self,
        session: AsyncSession,
        leader_id: str,
        trade: RawLeaderTrade,
    ) -> bool:
        """Process a closed trade event and check if circuit breaker is tripped.

        Returns True if the leader was demoted by the circuit breaker.
        """
        # Only monitor active live-tier leaders
        if leader_id not in self.roster.live_ids():
            return False

        if trade.is_open or trade.profit is None:
            return False

        state = self.get_or_create_state(leader_id)
        pnl = trade.profit

        # Update equity & drawdown tracking
        state.current_equity += pnl
        state.peak_equity = max(state.peak_equity, state.current_equity)
        dd_dollars = state.peak_equity - state.current_equity
        state.max_drawdown_pct = (dd_dollars / max(self.base_capital, 1.0)) * 100.0

        # Update consecutive loss counter
        if pnl < 0:
            state.consecutive_losses += 1
        else:
            state.consecutive_losses = 0

        # Check breach conditions
        should_demote = False
        reason = ""

        if state.consecutive_losses >= self.settings.leader_consecutive_loss_limit:
            should_demote = True
            reason = f"consecutive_losses_breach:{state.consecutive_losses}>={self.settings.leader_consecutive_loss_limit}"
        elif state.max_drawdown_pct >= self.settings.leader_live_drawdown_limit_pct:
            should_demote = True
            reason = f"live_drawdown_breach:{state.max_drawdown_pct:.2f}%>={self.settings.leader_live_drawdown_limit_pct:.2f}%"

        if should_demote and not state.demoted:
            state.demoted = True
            state.demotion_reason = reason
            log.warning("CIRCUIT BREAKER TRIPPED for leader %s: %s. Demoting immediately!", leader_id, reason)

            # Demote in roster
            candidate = self.roster.get(leader_id)
            if candidate:
                candidate.tier = "watch"
                self.roster.save()

            # Record RiskEvent in DB
            await record_risk_event(
                session,
                event="leader_circuit_breaker_demote",
                severity="WARNING",
                drawdown=state.max_drawdown_pct,
                action="demote_to_watch",
                detail={
                    "leader_id": leader_id,
                    "reason": reason,
                    "consecutive_losses": state.consecutive_losses,
                    "max_drawdown": state.max_drawdown_pct,
                },
                commit=True,
            )
            return True

        return False