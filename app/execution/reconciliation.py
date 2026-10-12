"""Position reconciliation between internal state and MT5/broker reality.

Periodic and on-demand job that compares Aries internal active positions
against actual open positions reported by MT5/broker.
Detects:
- Ghost positions: internal record open, but closed or missing at broker
- Orphan positions: position open at broker, but missing in internal records
- Volume / side / stop discrepancies
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import AuditLogEntry, OrderRecord
from ..logging import get_logger

log = get_logger(__name__)


@dataclass
class BrokerPosition:
    ticket: str
    symbol: str
    side: str  # "buy" | "sell"
    volume: float
    open_price: float
    stop_loss: float | None = None
    take_profit: float | None = None
    comment: str = ""


@dataclass
class InternalPosition:
    order_id: str
    broker_order_id: str | None
    symbol: str
    side: str
    volume: float
    entry_price: float
    stop_loss: float | None = None
    take_profit: float | None = None
    status: str = "filled"


@dataclass
class ReconciliationMismatch:
    mismatch_type: str  # "ghost_position" | "orphan_position" | "volume_mismatch" | "side_mismatch"
    ticket: str | None = None
    order_id: str | None = None
    details: str = ""


@dataclass
class ReconciliationResult:
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_clean: bool = True
    total_internal: int = 0
    total_broker: int = 0
    matched: int = 0
    mismatches: list[ReconciliationMismatch] = field(default_factory=list)


class PositionReconciler:
    """Detects and logs discrepancies between internal and external positions."""

    def reconcile(
        self,
        internal_positions: list[InternalPosition],
        broker_positions: list[BrokerPosition],
    ) -> ReconciliationResult:
        result = ReconciliationResult(
            total_internal=len(internal_positions),
            total_broker=len(broker_positions),
        )

        broker_by_ticket = {str(p.ticket): p for p in broker_positions}
        internal_by_broker_id = {
            str(p.broker_order_id): p for p in internal_positions if p.broker_order_id
        }

        # Check internal against broker
        for internal in internal_positions:
            b_id = str(internal.broker_order_id) if internal.broker_order_id else None
            if not b_id or b_id not in broker_by_ticket:
                result.mismatches.append(
                    ReconciliationMismatch(
                        mismatch_type="ghost_position",
                        order_id=internal.order_id,
                        ticket=b_id,
                        details=f"Internal order {internal.order_id} (ticket {b_id}) is marked open, but not found on broker",
                    )
                )
            else:
                broker_pos = broker_by_ticket[b_id]
                # Compare volume, side, symbol
                if abs(broker_pos.volume - internal.volume) > 1e-4:
                    result.mismatches.append(
                        ReconciliationMismatch(
                            mismatch_type="volume_mismatch",
                            order_id=internal.order_id,
                            ticket=b_id,
                            details=f"Volume mismatch for order {internal.order_id}: internal={internal.volume}, broker={broker_pos.volume}",
                        )
                    )
                elif broker_pos.side.lower() != internal.side.lower():
                    result.mismatches.append(
                        ReconciliationMismatch(
                            mismatch_type="side_mismatch",
                            order_id=internal.order_id,
                            ticket=b_id,
                            details=f"Side mismatch for order {internal.order_id}: internal={internal.side}, broker={broker_pos.side}",
                        )
                    )
                else:
                    result.matched += 1

        # Check broker against internal for orphans
        for ticket, broker_pos in broker_by_ticket.items():
            if ticket not in internal_by_broker_id:
                result.mismatches.append(
                    ReconciliationMismatch(
                        mismatch_type="orphan_position",
                        ticket=ticket,
                        details=f"Broker position ticket {ticket} ({broker_pos.symbol} {broker_pos.volume} {broker_pos.side}) is missing from internal records",
                    )
                )

        result.is_clean = len(result.mismatches) == 0

        if not result.is_clean:
            log.warning(
                "Position reconciliation mismatch detected! Total internal=%d, total broker=%d, mismatches=%d",
                result.total_internal,
                result.total_broker,
                len(result.mismatches),
            )
            for m in result.mismatches:
                log.warning("Reconciliation discrepancy [%s]: %s", m.mismatch_type, m.details)
        else:
            log.info(
                "Position reconciliation OK: %d matched positions cleanly",
                result.matched,
            )

        return result

    async def reconcile_with_db(
        self,
        session: AsyncSession,
        broker_positions: list[BrokerPosition],
    ) -> ReconciliationResult:
        """Fetch open internal orders from DB and run reconciliation."""
        stmt = select(OrderRecord).where(
            OrderRecord.status.in_(("filled", "open")),
            OrderRecord.broker_order_id.isnot(None),
        )
        rows = (await session.execute(stmt)).scalars().all()
        internal_positions = [
            InternalPosition(
                order_id=r.id,
                broker_order_id=r.broker_order_id,
                symbol=r.symbol,
                side=r.side.value if hasattr(r.side, "value") else str(r.side),
                volume=r.volume,
                entry_price=r.entry_price,
                stop_loss=r.stop_loss,
                take_profit=r.take_profit,
                status=r.status,
            )
            for r in rows
        ]
        result = self.reconcile(internal_positions, broker_positions)
        if not result.is_clean:
            session.add(
                AuditLogEntry(
                    ts=datetime.now(timezone.utc),
                    actor="reconciler",
                    action="reconciliation.mismatch",
                    detail={
                        "mismatches": [
                            {"type": m.mismatch_type, "ticket": m.ticket, "order_id": m.order_id, "detail": m.details}
                            for m in result.mismatches
                        ]
                    },
                )
            )
            await session.commit()
        return result