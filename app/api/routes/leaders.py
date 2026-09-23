"""Leader overview + leader scores."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.base import Database
from ...db.models import LeaderDailyScore
from ...db.models import Leader

router = APIRouter(prefix="/api/leaders", tags=["leaders"])


async def _session(request: Request) -> AsyncSession:
    db: Database = request.app.state.db
    async with db.session_factory() as session:
        yield session


@router.get("")
async def list_leaders(session: AsyncSession = Depends(_session)):
    rows = (await session.execute(select(Leader))).scalars().all()
    return [
        {
            "id": l.id,
            "name": l.name,
            "url": l.url,
            "audited": l.audited,
            "last_seen_at": l.last_seen_at,
        }
        for l in rows
    ]


@router.get("/roster")
async def roster(request: Request):
    """Two-tier roster: watch pool, live tier, scores, observations."""
    from ...leaders.discovery import LeaderRoster

    settings = request.app.state.settings
    roster = LeaderRoster(
        settings.roster_path,
        max_live=settings.max_live_leaders,
        max_watch=settings.max_watch_leaders,
        min_observations=settings.leader_min_observations,
    )
    candidates = sorted(
        (roster.get(i) for i in roster.all_ids()),
        key=lambda c: (c.tier != "live", -c.watch_score),
    )
    return {
        "live": roster.live_ids(),
        "watch": roster.watch_ids(),
        "candidates": [
            {
                "id": c.id,
                "name": c.name,
                "tier": c.tier,
                "watch_score": round(c.watch_score, 4),
                "observations": c.observations,
                "source": c.source,
            }
            for c in candidates
        ],
    }


@router.get("/{leader_id}/scores")
async def leader_scores(leader_id: str, session: AsyncSession = Depends(_session)):
    stmt = (
        select(LeaderDailyScore)
        .where(LeaderDailyScore.leader_id == leader_id)
        .order_by(LeaderDailyScore.day.desc())
        .limit(30)
    )
    rows = (await session.execute(stmt)).scalars().all()
    return [
        {
            "day": r.day,
            "profit_factor": r.profit_factor,
            "win_rate": r.win_rate,
            "max_drawdown": r.max_drawdown,
            "trade_count": r.trade_count,
            "consistency": r.consistency_score,
            "composite": r.composite_score,
        }
        for r in rows
    ]