"""Scheduled jobs (APScheduler): leader polling, scoring, COT pulls.

Wired from ``app/main.py``; each job is a dependency-injected callable so
tests can invoke them directly.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from ..config import Settings
from ..db import audit
from ..leaders.models import LeaderProfile
from ..leaders.myfxbook import MyfxbookClient
from ..leaders.repo import get_or_create_leader, upsert_trade_event
from ..leaders.scoring import compute_score, today_utc
from ..logging import get_logger

log = get_logger(__name__)


async def poll_leader_trades(settings: Settings, session_factory, myfxbook: MyfxbookClient) -> int:
    """Phase 1 — poll curated leaders for new XAUUSD trades, persist + alert."""
    inserted = 0
    for leader_id in settings.leader_ids:
        try:
            open_trades = await myfxbook.get_open_trades(leader_id)
            closed = await myfxbook.get_history(leader_id)
        except Exception as exc:  # network / API hiccup — do not kill the loop
            log.warning("Leader %s poll failed: %s", leader_id, exc)
            continue

        profile = LeaderProfile(id=leader_id, name=f"leader-{leader_id}", url="", audited=True)
        async with session_factory() as session:
            leader = await get_or_create_leader(session, profile)
            for trade in [*open_trades, *closed]:
                if trade.symbol != settings.market_symbol:
                    continue
                if await upsert_trade_event(session, leader.id, trade) is not None:
                    inserted += 1
            await session.commit()
    log.info("Leader poll finished: %d new %s events", inserted, settings.market_symbol)
    return inserted


async def run_daily_leader_scoring(settings: Settings, session_factory, myfxbook: MyfxbookClient) -> None:
    """Phase 2 — daily profit-factor / win-rate / drawdown / consistency per leader."""
    day = today_utc()
    from ..leaders.repo import store_daily_score

    for leader_id in settings.leader_ids:
        try:
            closed = await myfxbook.get_history(leader_id)
        except Exception as exc:
            log.warning("Leader %s scoring failed: %s", leader_id, exc)
            continue
        gold_closed = [t for t in closed if t.symbol == settings.market_symbol and not t.is_open]
        score = compute_score(gold_closed)
        async with session_factory() as session:
            await store_daily_score(session, leader_id, day, score)
            await audit.record_audit(
                session, actor="scheduler", action="leader.score",
                detail={"leader_id": leader_id, "composite": score.composite,
                        "profit_factor": _finite(score.profit_factor), "win_rate": score.win_rate},
            )
            await session.commit()
    log.info("Daily leader scoring complete for %s", day)


async def pull_cot_report(settings: Settings, session_factory) -> None:
    """Phase 2 — weekly COT ingestion. Persists a compact snapshot for features."""
    from ..data.cot import COTClient, cot_feature_vector

    client = COTClient()
    try:
        history = await client.history()
        vector = cot_feature_vector(history)
        log.info("COT refresh: %s", vector)
        async with session_factory() as session:
            await audit.record_audit(
                session, actor="scheduler", action="cot.refresh",
                detail={"report_date": str(history.index[-1]), "features": vector},
            )
            await session.commit()
    except Exception as exc:  # noqa: BLE001 - network sources fail; log and move on
        log.warning("COT pull failed: %s", exc)


def build_scheduler(settings: Settings, session_factory, myfxbook: MyfxbookClient) -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(
        poll_leader_trades,
        IntervalTrigger(seconds=settings.myfxbook_poll_interval_seconds),
        args=[settings, session_factory, myfxbook],
        id="leader_poll",
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        run_daily_leader_scoring,
        trigger="cron", hour=0, minute=5,
        args=[settings, session_factory, myfxbook],
        id="leader_scoring",
        coalesce=True,
    )
    scheduler.add_job(
        pull_cot_report,
        trigger="cron", day_of_week="fri", hour=21, minute=0,
        args=[settings, session_factory],
        id="cot_refresh",
        coalesce=True,
    )
    return scheduler


def _finite(x: float) -> float:
    import math

    return x if math.isfinite(x) else 0.0