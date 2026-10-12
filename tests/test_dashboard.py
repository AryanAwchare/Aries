"""Unit tests for the Aries Terminal Dashboard API and root endpoint."""
from __future__ import annotations

import httpx
import pytest

from app.config import Settings
from app.db.base import Database

pytestmark = pytest.mark.asyncio


@pytest.fixture()
async def dashboard_client(tmp_path):
    from app.main import app

    db_path = tmp_path / "dash_test.db"
    settings = Settings(database_url=f"sqlite+aiosqlite:///{db_path}")
    db = Database(settings)
    await db.create_all()
    app.state.db = db
    app.state.settings = settings
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    await db.dispose()


async def test_root_serves_dashboard_html(dashboard_client):
    res = await dashboard_client.get("/")
    assert res.status_code == 200
    assert "Aries" in res.text
    assert "TradingView" in res.text or "LightweightCharts" in res.text or "XAU/USD" in res.text


async def test_dashboard_overview_endpoint(dashboard_client):
    res = await dashboard_client.get("/api/dashboard/overview")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "account" in data
    assert "risk" in data
    assert "system" in data
    assert "open_positions" in data


async def test_dashboard_candles_endpoint(dashboard_client):
    res = await dashboard_client.get("/api/dashboard/candles?symbol=XAUUSD&timeframe=M15&limit=20")
    assert res.status_code == 200
    data = res.json()
    assert data["symbol"] == "XAUUSD"
    assert "candles" in data


async def test_dashboard_history_endpoint(dashboard_client):
    res = await dashboard_client.get("/api/dashboard/history?days=7")
    assert res.status_code == 200
    data = res.json()
    assert "deals" in data
