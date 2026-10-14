"""Tests for the operations control-plane router (``/api/ops``).

These exercise the terminal-facing actions: runtime config, research jobs,
leader-roster management, audit/risk logs, model info and the risk-gated
manual order ticket. Everything runs against a throwaway SQLite DB and a
throwaway roster file; ``persist`` is always false so the real ``.env`` is
never touched.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from app.config import Settings
from app.db.base import Database

pytestmark = pytest.mark.asyncio


@pytest.fixture()
async def ops_client(tmp_path):
    from app.main import app

    db_path = tmp_path / "ops_test.db"
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{db_path}",
        leader_roster_path=str(tmp_path / "roster.json"),
    )
    db = Database(settings)
    await db.create_all()

    # Reset any state a previous test may have left on the shared app object.
    app.state.db = db
    app.state.settings = settings
    for attr in ("profile", "kill_switch", "copy_trading"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    await db.dispose()


async def test_status(ops_client):
    res = await ops_client.get("/api/ops/status")
    assert res.status_code == 200
    data = res.json()
    assert data["trading_mode"] == "paper"
    assert data["active_profile"] == "default_demo"
    assert data["trading_allowed"] == data["kill_switch_enabled"]
    assert "mt5_available" in data


async def test_get_config(ops_client):
    res = await ops_client.get("/api/ops/config")
    assert res.status_code == 200
    data = res.json()
    assert data["trading_mode"] == "paper"
    assert data["active_prop_profile"] == "default_demo"
    assert "default_demo" in data["profiles"]
    assert set(data["integrations"]) == {"myfxbook", "metaapi", "ea_key", "mt5"}
    # secrets are masked
    assert "***" in data["effective_db_url"] or "@" not in data["effective_db_url"]


async def test_update_config_valid(ops_client):
    res = await ops_client.put(
        "/api/ops/config",
        json={"market_timeframe": "H1", "max_live_leaders": 3, "persist": False},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["changes"] == {"market_timeframe": "H1", "max_live_leaders": 3}
    assert body["persisted"] is False

    check = await ops_client.get("/api/ops/config")
    assert check.json()["market_timeframe"] == "H1"


async def test_update_config_rejects_bad_values(ops_client):
    no_fields = await ops_client.put("/api/ops/config", json={"persist": False})
    assert no_fields.status_code == 422

    bad_tf = await ops_client.put("/api/ops/config", json={"market_timeframe": "W1", "persist": False})
    assert bad_tf.status_code == 422

    bad_threshold = await ops_client.put(
        "/api/ops/config", json={"meta_model_prob_threshold": 0.4, "persist": False}
    )
    assert bad_threshold.status_code == 422

    bad_mode = await ops_client.put("/api/ops/config", json={"trading_mode": "turbo", "persist": False})
    assert bad_mode.status_code == 422

    unknown_profile = await ops_client.put(
        "/api/ops/config", json={"active_prop_profile": "nope", "persist": False}
    )
    assert unknown_profile.status_code == 422


async def test_live_mode_refused_without_token(ops_client):
    res = await ops_client.put("/api/ops/config", json={"trading_mode": "live", "persist": False})
    assert res.status_code == 409


async def test_cli_reference(ops_client):
    res = await ops_client.get("/api/ops/cli")
    assert res.status_code == 200
    cmds = {c["command"] for c in res.json()}
    assert {"backtest", "walk-forward", "ablation", "replay"} <= cmds


async def test_model_info(ops_client):
    res = await ops_client.get("/api/ops/model")
    assert res.status_code == 200
    data = res.json()
    assert "available" in data
    assert isinstance(data["artifacts"], list)
    assert isinstance(data["model_dir"], str)


async def test_audit_and_risk_events_empty(ops_client):
    audit = await ops_client.get("/api/ops/audit")
    assert audit.status_code == 200
    assert audit.json() == []
    risk = await ops_client.get("/api/ops/risk-events")
    assert risk.status_code == 200
    assert risk.json() == []


async def test_jobs_list_and_errors(ops_client):
    jobs = await ops_client.get("/api/ops/jobs")
    assert jobs.status_code == 200
    assert isinstance(jobs.json(), list)

    unknown_kind = await ops_client.post("/api/ops/jobs/not-a-job", json={"params": {}})
    assert unknown_kind.status_code == 404

    missing = await ops_client.get("/api/ops/jobs/does-not-exist")
    assert missing.status_code == 404


async def test_backtest_job_runs(ops_client):
    start = await ops_client.post(
        "/api/ops/jobs/backtest",
        json={"params": {"synthetic": True, "days": 20, "seed": 42}},
    )
    assert start.status_code == 200
    job_id = start.json()["id"]

    status = None
    for _ in range(120):
        res = await ops_client.get(f"/api/ops/jobs/{job_id}")
        assert res.status_code == 200
        status = res.json()
        if status["status"] in ("done", "error"):
            break
        await asyncio.sleep(0.5)

    assert status is not None
    assert status["status"] == "done", status.get("error")
    assert "n_trades" in status["result"]


async def test_leaders_roster_lifecycle(ops_client):
    empty = await ops_client.get("/api/ops/leaders/roster-full")
    assert empty.status_code == 200
    assert empty.json() == {"live": [], "watch": [], "candidates": []}

    added = await ops_client.post("/api/ops/leaders/add", json={"id": "12345", "name": "Alice"})
    assert added.status_code == 200
    assert added.json()["added"] is True
    assert added.json()["watch"] == ["12345"]

    dup = await ops_client.post("/api/ops/leaders/add", json={"id": "12345"})
    assert dup.json()["added"] is False

    live = await ops_client.post("/api/ops/leaders/live", json={"ids": ["12345"]})
    assert live.json()["live"] == ["12345"]

    roster = await ops_client.get("/api/ops/leaders/roster-full")
    cands = roster.json()["candidates"]
    assert cands and cands[0]["id"] == "12345" and cands[0]["tier"] == "live"

    removed = await ops_client.post("/api/ops/leaders/remove", json={"id": "12345"})
    assert removed.json()["removed"] is True
    assert removed.json()["live"] == []


async def test_manual_order_requires_confirmation(ops_client):
    res = await ops_client.post("/api/ops/order", json={"side": "buy"})
    assert res.status_code == 422
    assert "confirm" in res.json()["detail"].lower()

    bad_side = await ops_client.post("/api/ops/order", json={"side": "sideways", "confirm": True})
    assert bad_side.status_code == 422


async def test_setup_db(ops_client):
    res = await ops_client.post("/api/ops/setup-db")
    assert res.status_code == 200
    assert res.json()["ok"] is True
