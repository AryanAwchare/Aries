"""Scheduled jobs (APScheduler): leader polling, roster rotation, COT, scoring.

Implementation lives in ``app/leaders/poll.py`` (scheduler-free, shared with
the CLI); this module only wires the triggers. Built via ``build_scheduler``.
"""
from __future__ import annotations

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from ..config import Settings
from ..leaders.myfxbook import MyfxbookClient
from ..leaders.poll import (
    poll_leader_trades,
    refresh_leader_roster,
    run_daily_leader_scoring,
)


async def pull_cot_report(settings: Settings, session_factory) -> None:
    """Weekly COT ingestion. Persists a compact snapshot for features."""
    from ..data.cot import COTClient, cot_feature_vector
    from ..db import audit

    client = COTClient()
    try:
        history = await client.history()
        vector = cot_feature_vector(history)
        async with session_factory() as session:
            await audit.record_audit(
                session, actor="scheduler", action="cot.refresh",
                detail={"report_date": str(history.index[-1]), "features": vector},
            )
            await session.commit()
    except Exception as exc:  # noqa: BLE001 - network sources fail; log and move on
        from ..logging import get_logger

        get_logger(__name__).warning("COT pull failed: %s", exc)


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
        refresh_leader_roster,
        trigger="cron", hour=8, minute=0,
        args=[settings, myfxbook, session_factory],
        id="leader_roster",
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