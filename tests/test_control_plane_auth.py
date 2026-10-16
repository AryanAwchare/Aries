"""Tests for control-plane authentication (``OPS_API_KEY``).

Mutating/dangerous endpoints must reject unauthenticated requests whenever a
key is configured, and accept the key via ``X-API-Key`` or ``Bearer``. When no
key is configured (local single-user mode) they stay open.
"""
from __future__ import annotations

import httpx
import pytest

from app.config import Settings
from app.db.base import Database

pytestmark = pytest.mark.asyncio

_CONFIG_BODY = {"market_timeframe": "H1", "max_live_leaders": 3, "persist": False}


@pytest.fixture()
async def auth_client(tmp_path):
    from app.main import app

    db_path = tmp_path / "auth_test.db"
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{db_path}",
        leader_roster_path=str(tmp_path / "roster.json"),
        ops_api_key="test-secret-key",
    )
    db = Database(settings)
    await db.create_all()

    app.state.db = db
    app.state.settings = settings
    for attr in ("profile", "kill_switch", "copy_trading"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    await db.dispose()


@pytest.fixture()
async def open_client(tmp_path):
    from app.main import app

    db_path = tmp_path / "open_test.db"
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{db_path}",
        leader_roster_path=str(tmp_path / "roster.json"),
        ops_api_key="",
    )
    db = Database(settings)
    await db.create_all()

    app.state.db = db
    app.state.settings = settings
    for attr in ("profile", "kill_switch", "copy_trading"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    await db.dispose()


async def test_mutating_endpoints_reject_missing_key(auth_client):
    for method, path, body in (
        ("put", "/api/ops/config", _CONFIG_BODY),
        ("put", "/api/modes", {"mode": "paper"}),
        ("put", "/api/risk/killswitch", {"enabled": True}),
        ("post", "/api/ops/order", {"side": "buy", "confirm": True}),
        ("post", "/api/ops/setup-db", None),
        ("post", "/api/ops/leaders/add", {"id": "x"}),
    ):
        res = await getattr(auth_client, method)(path, json=body)
        assert res.status_code == 401, f"{method.upper()} {path} -> {res.status_code}"


async def test_mutating_endpoint_accepts_x_api_key(auth_client):
    res = await auth_client.put(
        "/api/ops/config", json=_CONFIG_BODY, headers={"X-API-Key": "test-secret-key"}
    )
    assert res.status_code == 200
    assert res.json()["ok"] is True


async def test_mutating_endpoint_accepts_bearer_token(auth_client):
    res = await auth_client.put(
        "/api/ops/config",
        json=_CONFIG_BODY,
        headers={"Authorization": "Bearer test-secret-key"},
    )
    assert res.status_code == 200


async def test_mutating_endpoint_rejects_wrong_key(auth_client):
    res = await auth_client.put(
        "/api/ops/config", json=_CONFIG_BODY, headers={"X-API-Key": "nope"}
    )
    assert res.status_code == 401


async def test_read_endpoints_remain_open_without_key(auth_client):
    assert (await auth_client.get("/api/ops/status")).status_code == 200
    assert (await auth_client.get("/api/ops/config")).status_code == 200


async def test_mutating_open_when_key_unconfigured(open_client):
    res = await open_client.put("/api/ops/config", json=_CONFIG_BODY)
    assert res.status_code == 200
    assert res.json()["ok"] is True
