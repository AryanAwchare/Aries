"""Tests for Phase 1: Execution Reliability.

Covers:
1. Rejecting orders with invalid SL/TP direction (BUY SL>=entry or TP<=entry, SELL SL<=entry or TP>=entry).
2. Rejecting orders below broker minimum stop distance.
3. Rejecting invalid volumes (below min, above max, off volume step).
4. Duplicate order ID rejection / idempotency.
5. Robust fill-report error handling without unhandled crashes.
6. Position reconciliation (ghost, orphan, volume mismatches).
"""
from __future__ import annotations

import uuid
import httpx
import pytest
from sqlalchemy import select

from app.config import Settings
from app.db.base import Database
from app.db.models import EaOrder, OrderRecord, SystemEvent, TradeSide
from app.execution.base import OrderRequest, OrderResult, PaperBroker
from app.execution.reconciliation import (
    BrokerPosition,
    InternalPosition,
    PositionReconciler,
)
from app.execution.validator import (
    DEFAULT_SYMBOL_SPECS,
    OrderValidator,
    SymbolSpecification,
)

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------- Validator Tests
def test_order_validator_valid_buy_and_sell():
    validator = OrderValidator()
    # Valid BUY
    buy_req = OrderRequest(
        symbol="XAUUSD",
        side="buy",
        volume=0.10,
        entry_ref_price=2000.00,
        stop_loss=1990.00,
        take_profit=2020.00,
    )
    res = validator.validate(buy_req)
    assert res.valid is True
    assert res.error == ""

    # Valid SELL
    sell_req = OrderRequest(
        symbol="XAUUSD",
        side="sell",
        volume=0.10,
        entry_ref_price=2000.00,
        stop_loss=2010.00,
        take_profit=1980.00,
    )
    res = validator.validate(sell_req)
    assert res.valid is True
    assert res.error == ""


def test_order_validator_invalid_sltp_direction():
    validator = OrderValidator()

    # BUY with SL >= entry
    bad_buy_sl = OrderRequest(
        symbol="XAUUSD",
        side="buy",
        volume=0.10,
        entry_ref_price=2000.00,
        stop_loss=2005.00,  # Invalid: SL above entry on buy
        take_profit=2020.00,
    )
    res = validator.validate(bad_buy_sl)
    assert res.valid is False
    assert "strictly below entry" in res.error

    # BUY with TP <= entry
    bad_buy_tp = OrderRequest(
        symbol="XAUUSD",
        side="buy",
        volume=0.10,
        entry_ref_price=2000.00,
        stop_loss=1990.00,
        take_profit=1995.00,  # Invalid: TP below entry on buy
    )
    res = validator.validate(bad_buy_tp)
    assert res.valid is False
    assert "strictly above entry" in res.error

    # SELL with SL <= entry
    bad_sell_sl = OrderRequest(
        symbol="XAUUSD",
        side="sell",
        volume=0.10,
        entry_ref_price=2000.00,
        stop_loss=1995.00,  # Invalid: SL below entry on sell
        take_profit=1980.00,
    )
    res = validator.validate(bad_sell_sl)
    assert res.valid is False
    assert "strictly above entry" in res.error

    # SELL with TP >= entry
    bad_sell_tp = OrderRequest(
        symbol="XAUUSD",
        side="sell",
        volume=0.10,
        entry_ref_price=2000.00,
        stop_loss=2010.00,
        take_profit=2005.00,  # Invalid: TP above entry on sell
    )
    res = validator.validate(bad_sell_tp)
    assert res.valid is False
    assert "strictly below entry" in res.error


def test_order_validator_below_min_stop_distance():
    # Gold min stop distance is 0.50 ($0.50 / 50 points)
    spec = SymbolSpecification(symbol="XAUUSD", min_stop_distance_points=0.50)
    validator = OrderValidator(specs={"XAUUSD": spec})

    # BUY with SL distance 0.20 < 0.50
    too_close_sl = OrderRequest(
        symbol="XAUUSD",
        side="buy",
        volume=0.10,
        entry_ref_price=2000.00,
        stop_loss=1999.80,  # Distance only 0.20
        take_profit=2020.00,
    )
    res = validator.validate(too_close_sl)
    assert res.valid is False
    assert "below minimum stop distance" in res.error

    # BUY with TP distance 0.10 < 0.50
    too_close_tp = OrderRequest(
        symbol="XAUUSD",
        side="buy",
        volume=0.10,
        entry_ref_price=2000.00,
        stop_loss=1990.00,
        take_profit=2000.10,  # Distance only 0.10
    )
    res = validator.validate(too_close_tp)
    assert res.valid is False
    assert "below minimum stop distance" in res.error


def test_order_validator_volume_constraints():
    spec = SymbolSpecification(
        symbol="XAUUSD",
        min_volume=0.01,
        max_volume=10.0,
        volume_step=0.01,
    )
    validator = OrderValidator(specs={"XAUUSD": spec})

    # Volume below min
    res = validator.validate(OrderRequest(symbol="XAUUSD", side="buy", volume=0.005, entry_ref_price=2000.0))
    assert res.valid is False
    assert "below minimum" in res.error

    # Volume above max
    res = validator.validate(OrderRequest(symbol="XAUUSD", side="buy", volume=15.0, entry_ref_price=2000.0))
    assert res.valid is False
    assert "exceeds maximum" in res.error

    # Invalid volume step (0.015 with step 0.01)
    res = validator.validate(OrderRequest(symbol="XAUUSD", side="buy", volume=0.015, entry_ref_price=2000.0))
    assert res.valid is False
    assert "align with volume step" in res.error


# ---------------------------------------------------------------- Idempotency & Retry Tests
async def test_duplicate_order_id_rejection():
    broker = PaperBroker()
    cid = "test-unique-order-123"
    req = OrderRequest(
        symbol="XAUUSD",
        side="buy",
        volume=0.1,
        entry_ref_price=2000.0,
        stop_loss=1990.0,
        take_profit=2020.0,
        client_order_id=cid,
    )

    # First attempt succeeds
    res1 = await broker.execute_with_retry(req)
    assert res1.ok is True
    assert res1.client_order_id == cid

    # Second attempt with same client_order_id returns cached result (no duplicate fill)
    res2 = await broker.execute_with_retry(req)
    assert res2.ok is True
    assert res2.client_order_id == cid
    assert res2.broker_order_id == res1.broker_order_id
    assert res2.fill_price == res1.fill_price


# ---------------------------------------------------------------- EA Bridge Execution Report Resilience
@pytest.fixture()
async def bridge_client(tmp_path):
    from app.main import app

    db_path = tmp_path / "ea_resilience.db"
    settings = Settings(database_url=f"sqlite+aiosqlite:///{db_path}")
    db = Database(settings)
    await db.create_all()
    app.state.db = db
    app.state.settings = settings
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, db
    await db.dispose()


async def test_execution_report_malformed_payload_no_crash(bridge_client):
    client, db = bridge_client

    # 1. Invalid plain-text format (less than 5 parts) -> Returns 400 Bad Request, no crash
    resp1 = await client.post("/ea/execution", content="INVALID|DATA")
    assert resp1.status_code == 400
    assert "ERR:bad-format" in resp1.text

    # 2. Corrupt JSON -> Returns 400 Bad Request, no crash
    resp2 = await client.post("/ea/execution", content="{malformed json,")
    assert resp2.status_code == 400
    assert "ERR:parse-error" in resp2.text

    # 3. Unknown order -> Returns 404 Not Found, no crash
    resp3 = await client.post("/ea/execution", content="RESULT|nonexistent_order_id|1|999|2000.0")
    assert resp3.status_code == 404
    assert "ERR:unknown-order" in resp3.text


async def test_execution_report_idempotent_duplicate_fills(bridge_client):
    client, db = bridge_client
    ext = uuid.uuid4().hex

    async with db.session_factory() as session:
        session.add(
            EaOrder(
                external_id=ext,
                symbol="XAUUSD",
                side=TradeSide.BUY,
                volume=0.1,
            )
        )
        await session.commit()

    # First fill
    resp1 = await client.post("/ea/execution", content=f"RESULT|{ext}|1|ticket-100|2035.50")
    assert resp1.status_code == 200
    assert resp1.text == "OK"

    # Second identical fill should not crash, returns OK idempotently
    resp2 = await client.post("/ea/execution", content=f"RESULT|{ext}|1|ticket-100|2035.50")
    assert resp2.status_code == 200
    assert resp2.text == "OK"

    async with db.session_factory() as session:
        row = (await session.execute(
            select(EaOrder).where(EaOrder.external_id == ext)
        )).scalar_one()
        assert row.status == "filled"
        assert row.broker_order_id == "ticket-100"


# ---------------------------------------------------------------- Position Reconciliation Tests
def test_position_reconciler_detects_ghosts_and_orphans():
    reconciler = PositionReconciler()

    internal = [
        InternalPosition(order_id="ord-1", broker_order_id="1001", symbol="XAUUSD", side="buy", volume=0.5, entry_price=2000.0),
        InternalPosition(order_id="ord-2", broker_order_id="1002", symbol="XAUUSD", side="sell", volume=0.2, entry_price=2010.0),
    ]

    broker = [
        BrokerPosition(ticket="1001", symbol="XAUUSD", side="buy", volume=0.5, open_price=2000.0),
        # Ticket 1003 is open on broker but missing in internal (orphan)
        BrokerPosition(ticket="1003", symbol="XAUUSD", side="buy", volume=0.1, open_price=2005.0),
        # Notice ticket 1002 is missing on broker (ghost internal position)
    ]

    result = reconciler.reconcile(internal, broker)
    assert result.is_clean is False
    assert result.matched == 1
    assert len(result.mismatches) == 2

    mismatch_types = {m.mismatch_type for m in result.mismatches}
    assert "ghost_position" in mismatch_types
    assert "orphan_position" in mismatch_types


async def test_reconcile_with_db_persists_audit_on_mismatch(tmp_path):
    """Regression: the DB-backed path builds a SystemEvent via record_audit.

    Previously it did ``AuditLogEntry(ts=..., actor=..., action=...)`` whose
    attributes are read-only properties -> AttributeError exactly when a
    mismatch needed to be logged.
    """
    db_path = tmp_path / "recon.db"
    settings = Settings(database_url=f"sqlite+aiosqlite:///{db_path}")
    db = Database(settings)
    await db.create_all()

    async with db.session_factory() as session:
        session.add(
            OrderRecord(
                broker_order_id="ticket-ghost",
                symbol="XAUUSD",
                direction=TradeSide.BUY,
                volume=0.5,
                entry=2000.0,
                status="filled",
            )
        )
        await session.commit()

    reconciler = PositionReconciler()
    # Broker reports no open positions -> the internal order is a ghost.
    async with db.session_factory() as session:
        result = await reconciler.reconcile_with_db(session, broker_positions=[])

    assert result.is_clean is False
    assert any(m.mismatch_type == "ghost_position" for m in result.mismatches)

    async with db.session_factory() as session:
        rows = (await session.execute(select(SystemEvent))).scalars().all()

    assert len(rows) == 1
    assert rows[0].component == "reconciler"
    assert rows[0].event == "reconciliation.mismatch"
    assert rows[0].detail["mismatches"][0]["type"] == "ghost_position"

    await db.dispose()
