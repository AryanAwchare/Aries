"""Executor — orchestrates risk-check → place → persist → audit."""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from ..config import Settings
from ..db import audit
from ..db.models import OrderRecord, SignalDecision, SignalStatus
from ..logging import get_logger
from ..risk.engine import AccountState, OrderCandidate, RiskDecision, RiskEngine
from .base import ExecutionClient, OrderRequest
from .killswitch import KillSwitch

log = get_logger(__name__)


@dataclass
class ExecutionOutcome:
    order: OrderRecord | None
    risk: RiskDecision
    message: str = ""


class Executor:
    def __init__(
        self,
        settings: Settings,
        risk_engine: RiskEngine,
        client: ExecutionClient,
        kill_switch: KillSwitch,
    ) -> None:
        self.settings = settings
        self.risk = risk_engine
        self.client = client
        self.kill_switch = kill_switch

    async def execute_signal(
        self,
        session: AsyncSession,
        decision: SignalDecision,
        account: AccountState,
        entry_ref_price: float,
        stop_points: float,
        tp_points_multiple: float = 2.0,
    ) -> ExecutionOutcome:
        """Turn an APPROVED signal decision into an order (or a logged rejection)."""
        candidate = OrderCandidate(
            side=decision.side.value,
            entry_price=entry_ref_price,
            stop_points=stop_points,
            source_branch=("leader" if decision.verb == "copy" and decision.leader_id else "ml"),
            meta_probability=decision.meta_model_probability,
            symbol=decision.symbol,
        )
        risk_outcome = self.risk.approve(candidate, account)

        if not risk_outcome.approved:
            log.warning("Order rejected [%s]: %s", risk_outcome.reason_code, risk_outcome.reason)
            await audit.record_audit(
                session,
                actor="executor",
                action=f"execution.reject.{risk_outcome.reason_code}",
                detail={"signal_id": decision.id, "reason": risk_outcome.reason},
            )
            await session.commit()
            return ExecutionOutcome(None, risk_outcome, risk_outcome.reason)

        # Slot is reserved by the rate limiter at this point.
        order = OrderRecord(
            signal_id=decision.id,
            symbol=decision.symbol,
            side=decision.side,
            volume=risk_outcome.volume,
            entry_price=entry_ref_price,
            stop_loss=entry_ref_price - stop_points if decision.side.value == "buy" else entry_ref_price + stop_points,
            take_profit=entry_ref_price + tp_points_multiple * stop_points if decision.side.value == "buy" else entry_ref_price - tp_points_multiple * stop_points,
            mode=self.settings.trading_mode.value,
        )
        session.add(order)
        await session.flush()

        order_result = await self.client.place_market_order(
            OrderRequest(
                symbol=decision.symbol,
                side=decision.side.value,
                volume=risk_outcome.volume,
                entry_ref_price=entry_ref_price,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
            )
        )

        if not order_result.ok:
            # release the reserved rate-limit slot on failure
            order.status = "failed"
            order.error = order_result.error
            log.error("Order failed: %s", order_result.error)
            await audit.record_audit(session, actor="executor", action="execution.failed",
                                     detail={"signal_id": decision.id, "error": order_result.error})
        else:
            order.status = "filled"
            order.broker_order_id = order_result.broker_order_id
            order.entry_price = order_result.fill_price
            log.info("Filled %s %s %s lots @ %.2f (%s)",
                     order.side.value, order.symbol, order.volume, order_result.fill_price, order.mode)
            await audit.record_audit(session, actor="executor", action="execution.filled",
                                     detail={"signal_id": decision.id, "broker_order_id": order_result.broker_order_id})

        decision.status = SignalStatus.EXECUTED
        await session.commit()
        return ExecutionOutcome(order, risk_outcome, "filled" if order.status == "filled" else order.error)