"""EA bridge lifecycle: dispatch -> claim -> execute report -> fill matched."""
from __future__ import annotations

import uuid

import httpx
import pytest

from app.config import Settings
from app.db.base import Database
from app.db.models import EaOrder, TradeSide

pytestmark = pytest.mark.asyncio


@pytest.fixture()
async def bridge_app(tmp_path):
    """App state bound to a throwaway SQLite DB so tests never share rows."""
    from app.main import app

    db_path = tmp_path / "ea_test.db"
    settings = Settings(database_url=f"sqlite+aiosqlite:///{db_path}")
    db = Database(settings)
    await db.create_all()
    app.state.db = db
    app.state.settings = settings
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, db
    await db.dispose()


async def test_ea_bridge_lifecycle(bridge_app):
    client, db = bridge_app

    ping = await client.get("/ea/ping")
    assert ping.status_code == 200
    assert ping.json()["ok"] is True

    # nothing queued yet
    assert (await client.get("/ea/next-order")).text == "NONE"

    # executor-equivalent: a pending order lands in the bridge table
    external = uuid.uuid4().hex
    async with db.session_factory() as session:
        session.add(EaOrder(
            external_id=external, order_id=None, symbol="XAUUSD",
            side=TradeSide.BUY, volume=0.5, stop_loss=None, take_profit=None,
        ))
        await session.commit()

    # terminal claims exactly one order; a second poll gets nothing
    claimed = (await client.get("/ea/next-order")).text
    assert claimed.startswith("ORDER|")
    parts = claimed.split("|")
    assert parts[1] == external
    assert parts[2] == "XAUUSD" and parts[3] == "buy" and parts[4] == "0.5"
    assert (await client.get("/ea/next-order")).text == "NONE"

    # terminal reports a fill
    resp = await client.post(
        "/ea/execution",
        content=f"RESULT|{external}|1|661234|2034.12",
    )
    assert resp.text == "OK"

    from sqlalchemy import select

    async with db.session_factory() as session:
        row = (await session.execute(
            select(EaOrder).where(EaOrder.external_id == external)
        )).scalar_one()
        assert row.status == "filled"
        assert row.broker_order_id == "661234"
        assert row.fill_price == 2034.12


async def test_test_order_endpoint_queues(bridge_app):
    client, db = bridge_app

    resp = await client.post("/ea/test-order")
    assert resp.text.startswith("QUEUED|")
    external = resp.text.split("|")[1]

    # the EA would claim it immediately
    claimed = (await client.get("/ea/next-order")).text
    assert claimed.startswith("ORDER|") and external in claimed

    # terminal reports the fill, then status check shows filled
    await client.post("/ea/execution", content=f"RESULT|{external}|1|998877|2040.55")
    status = (await client.get(f"/ea/order/{external}")).text
    assert status == "STATUS|filled|2040.55|"


async def test_ea_bridge_rejected_order(bridge_app):
    client, db = bridge_app

    external = uuid.uuid4().hex
    async with db.session_factory() as session:
        session.add(EaOrder(
            external_id=external, symbol="XAUUSD", side=TradeSide.SELL,
            volume=0.1, stop_loss=2100.0, take_profit=2050.0,
        ))
        await session.commit()

    claimed = (await client.get("/ea/next-order")).text
    assert "ORDER|" in claimed
    parts = claimed.split("|")
    assert parts[5] == "2100" and parts[6] == "2050"

    resp = await client.post(
        "/ea/execution",
        content=f"RESULT|{external}|0|||off quotes",
    )
    assert resp.text == "OK"

    from sqlalchemy import select

    async with db.session_factory() as session:
        row = (await session.execute(
            select(EaOrder).where(EaOrder.external_id == external)
        )).scalar_one()
        assert row.status == "rejected"
        assert row.error == "off quotes"