"""Executor — orchestrates risk-check -> symbol validation -> place -> persist -> audit."""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from ..config import Settings
from ..db import audit
from ..db.models import OrderRecord, SignalDecision, SignalStatus
from ..logging import get_logger
from ..risk.engine import AccountState, OrderCandidate, RiskDecision, RiskEngine
from .base import ExecutionClient, OrderRequest
from .killswitch import KillSwitch
from .validator import OrderValidator, ValidationResult

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
        validator: OrderValidator | None = None,
    ) -> None:
        self.settings = settings
        self.risk = risk_engine
        self.client = client
        self.kill_switch = kill_switch
        self.validator = validator or OrderValidator()

    async def execute_signal(
        self,
        session: AsyncSession,
        decision: SignalDecision,
        account: AccountState,
        entry_ref_price: float,
        stop_points: float,
        tp_points_multiple: float = 2.0,
        client_order_id: str | None = None,
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
            log.warning("Order rejected by risk engine [%s]: %s", risk_outcome.reason_code, risk_outcome.reason)
            await audit.record_audit(
                session,
                actor="executor",
                action=f"execution.reject.{risk_outcome.reason_code}",
                detail={"signal_id": decision.id, "reason": risk_outcome.reason},
            )
            await session.commit()
            return ExecutionOutcome(None, risk_outcome, risk_outcome.reason)

        cid = client_order_id or f"aries-{uuid.uuid4().hex[:12]}"
        stop_loss = (
            entry_ref_price - stop_points
            if decision.side.value == "buy"
            else entry_ref_price + stop_points
        )
        take_profit = (
            entry_ref_price + tp_points_multiple * stop_points
            if decision.side.value == "buy"
            else entry_ref_price - tp_points_multiple * stop_points
        )

        req = OrderRequest(
            symbol=decision.symbol,
            side=decision.side.value,
            volume=risk_outcome.volume,
            entry_ref_price=entry_ref_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            client_order_id=cid,
        )

        # Symbol-aware validation BEFORE sending to broker/EA
        val = self.validator.validate(req)
        if not val.valid:
            log.error("Order pre-execution validation failed for signal %s: %s", decision.id, val.error)
            await audit.record_audit(
                session,
                actor="executor",
                action="execution.validation_failed",
                detail={"signal_id": decision.id, "error": val.error, "client_order_id": cid},
            )
            decision.status = SignalStatus.REJECTED
            decision.rejection_reason = f"validation_error: {val.error}"
            await session.commit()
            return ExecutionOutcome(None, risk_outcome, val.error)

        # Create OrderRecord row in DB
        order = OrderRecord(
            signal_id=decision.id,
            symbol=decision.symbol,
            side=decision.side,
            volume=risk_outcome.volume,
            entry_price=entry_ref_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            mode=self.settings.trading_mode.value,
        )
        session.add(order)
        await session.flush()

        try:
            # Place order with retry capability and timeout protection
            order_result = await self.client.execute_with_retry(
                req,
                timeout_seconds=float(self.settings.ea_order_timeout_seconds),
            )
        except Exception as exc:
            log.error("Fatal exception during order dispatch: %s", exc, exc_info=True)
            order.status = "failed"
            order.error = str(exc)
            await audit.record_audit(
                session,
                actor="executor",
                action="execution.exception",
                detail={"signal_id": decision.id, "error": str(exc)},
            )
            decision.status = SignalStatus.REJECTED
            decision.rejection_reason = str(exc)
            await session.commit()
            return ExecutionOutcome(order, risk_outcome, f"fatal: {exc}")

        # Process fill / rejection report with explicit error handling and audit
        try:
            if not order_result.ok:
                order.status = "failed"
                order.error = order_result.error
                log.error("Order failed to fill: %s", order_result.error)
                await audit.record_audit(
                    session,
                    actor="executor",
                    action="execution.failed",
                    detail={"signal_id": decision.id, "error": order_result.error},
                )
            else:
                order.status = "filled"
                order.broker_order_id = order_result.broker_order_id
                order.entry_price = order_result.fill_price
                log.info(
                    "Filled %s %s %.2f lots @ %.2f (broker ticket: %s, mode: %s)",
                    order.side.value,
                    order.symbol,
                    order.volume,
                    order_result.fill_price,
                    order_result.broker_order_id,
                    order.mode,
                )
                await audit.record_audit(
                    session,
                    actor="executor",
                    action="execution.filled",
                    detail={
                        "signal_id": decision.id,
                        "broker_order_id": order_result.broker_order_id,
                        "fill_price": order_result.fill_price,
                    },
                )

            decision.status = SignalStatus.EXECUTED
            await session.commit()
        except Exception as exc:
            log.critical("Failed during fill acknowledgement persistence: %s", exc, exc_info=True)
            # Ensure DB rollback or best effort audit
            await session.rollback()
            return ExecutionOutcome(order, risk_outcome, f"persistence_error: {exc}")

        return ExecutionOutcome(order, risk_outcome, "filled" if order.status == "filled" else order.error)