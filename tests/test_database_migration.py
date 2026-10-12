"""Tests for Phase 2: Supabase Migration & Async Persistence Architecture.

Validates:
1. Environment-driven DB URL resolution (SUPABASE_DB_URL vs SQLite fallback).
2. All 7 core schema tables (leaders, leader_scores, leader_trades, signals, orders, risk_events, system_events).
3. Non-blocking asynchronous writer: execution proceeds without blocking even if DB writes fail/queue.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import pytest
from sqlalchemy import select

from app.config import Settings
from app.db.async_writer import AsyncDatabaseWriter
from app.db.audit import record_audit, record_risk_event
from app.db.base import Database
from app.db.models import (
    Leader,
    LeaderScoreRecord,
    LeaderTrade,
    OrderRecord,
    RiskEvent,
    Signal,
    SystemEvent,
    TradeSide,
)

pytestmark = pytest.mark.asyncio


def test_db_url_environment_resolution(monkeypatch):
    # Default to SQLite
    s = Settings()
    assert "sqlite" in s.effective_db_url

    # When SUPABASE_DB_URL is set with standard postgres URI, normalizes to asyncpg
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://user:pass@db.supabase.co:5432/postgres")
    s2 = Settings()
    assert s2.effective_db_url == "postgresql+asyncpg://user:pass@db.supabase.co:5432/postgres"


async def test_all_core_tables_crud(tmp_path):
    db_path = tmp_path / "test_migration.db"
    settings = Settings(database_url=f"sqlite+aiosqlite:///{db_path}")
    db = Database(settings)
    await db.create_all()

    async with db.session_factory() as session:
        # 1. Leader
        leader = Leader(
            id="test-leader-1",
            myfxbook_id="123456",
            name="Gold Titan",
            broker="ICMarkets",
            account_age=240,
            verified_track_record=True,
            verified_trading_privileges=True,
            symbol="XAUUSD",
            strategy_type="trend_following",
            status="live",
            is_active=True,
        )
        session.add(leader)

        # 2. LeaderScore
        score = LeaderScoreRecord(
            leader_id="test-leader-1",
            date=datetime.now(timezone.utc),
            profit_factor=2.45,
            win_rate=0.68,
            max_drawdown=8.5,
            expectancy=15.2,
            consistency=0.88,
            longevity=0.90,
            recent_score=0.85,
            composite_score=0.84,
        )
        session.add(score)

        # 3. LeaderTrade
        trade = LeaderTrade(
            leader_id="test-leader-1",
            trade_id="tr-9988",
            symbol="XAUUSD",
            direction=TradeSide.BUY,
            entry=2050.50,
            exit=2065.00,
            lot_size=1.0,
            profit=1450.0,
        )
        session.add(trade)

        # 4. Signal
        sig = Signal(
            symbol="XAUUSD",
            timeframe="M15",
            leader_direction="buy",
            kronos_direction="buy",
            kronos_confidence=0.82,
            xgb_probability=0.74,
            regime="trending_bull",
            final_signal="BUY",
            contributing_factors={"leader_weight": 0.4, "xgb_weight": 0.6},
        )
        session.add(sig)
        await session.flush()

        # 5. Order
        ord_rec = OrderRecord(
            signal_id=sig.id,
            broker_order_id="ticket-5544",
            symbol="XAUUSD",
            direction=TradeSide.BUY,
            volume=0.5,
            entry=2050.50,
            sl=2040.00,
            tp=2070.00,
            status="filled",
        )
        session.add(ord_rec)

        # 6. RiskEvent
        risk_ev = await record_risk_event(
            session,
            event="drawdown_warning",
            severity="WARNING",
            drawdown=3.2,
            action="reduce_risk_fraction",
            commit=False,
        )

        # 7. SystemEvent (Audit)
        sys_ev = await record_audit(
            session,
            actor="test_runner",
            action="schema_verification",
            commit=False,
        )

        await session.commit()

    # Query back and verify
    async with db.session_factory() as session:
        l_res = (await session.execute(select(Leader).where(Leader.id == "test-leader-1"))).scalar_one()
        assert l_res.verified_track_record is True
        assert l_res.verified_trading_privileges is True
        assert l_res.strategy_type == "trend_following"

        s_res = (await session.execute(select(Signal).where(Signal.symbol == "XAUUSD"))).scalar_one()
        assert s_res.final_signal == "BUY"
        assert s_res.xgb_probability == 0.74

        r_res = (await session.execute(select(RiskEvent))).scalar_one()
        assert r_res.severity == "WARNING"

        sys_res = (await session.execute(select(SystemEvent))).scalar_one()
        assert sys_res.event == "schema_verification"

    await db.dispose()


async def test_async_database_writer_non_blocking():
    """Simulate fire-and-forget logging when DB write might fail or lag."""
    writer = AsyncDatabaseWriter(max_queue_size=100)
    writer.start()

    processed_items = []

    async def mock_slow_db_write(item_id: int):
        await asyncio.sleep(0.01)
        processed_items.append(item_id)

    # Enqueue 5 write tasks
    for i in range(5):
        ok = writer.enqueue(f"task_{i}", mock_slow_db_write, i)
        assert ok is True

    # Critical path did not wait for writes to finish!
    await asyncio.sleep(0.1)
    await writer.stop()

    assert len(processed_items) == 5