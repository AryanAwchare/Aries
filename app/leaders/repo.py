"""Persistence helpers for leaders + trade events."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import Leader, LeaderDailyScore, TradeEvent
from .models import LeaderProfile, RawLeaderTrade


async def get_or_create_leader(session: AsyncSession, profile: LeaderProfile) -> Leader:
    leader = await session.get(Leader, profile.id)
    if leader is None:
        leader = Leader(
            id=profile.id,
            name=profile.name,
            url=profile.url,
            audited=profile.audited,
        )
        session.add(leader)
        await session.flush()
    return leader


async def upsert_trade_event(session: AsyncSession, leader_id: str, trade: RawLeaderTrade) -> TradeEvent | None:
    """Insert-or-skip a normalised leader trade. Returns None if already present."""
    stmt = select(TradeEvent).where(
        TradeEvent.leader_id == leader_id,
        TradeEvent.external_trade_id == trade.external_id,
    )
    existing = (await session.execute(stmt)).scalar_one_or_none()
    if existing is not None:
        return None

    event = TradeEvent(
        leader_id=leader_id,
        external_trade_id=trade.external_id,
        symbol=trade.symbol,
        side=trade.side.lower(),  # type: ignore[arg-type]
        volume=trade.volume,
        open_price=trade.open_price,
        close_price=trade.close_price,
        profit=trade.profit,
        lot_pips=trade.pips or trade.lots,
        opened_at=trade.opened_at or __import__("datetime").datetime.utcnow(),
        closed_at=trade.closed_at,
        is_open=trade.is_open,
        raw_json=trade.raw,
    )
    session.add(event)
    return event


async def store_daily_score(
    session: AsyncSession,
    leader_id: str,
    day,
    score,
) -> None:
    stmt = select(LeaderDailyScore).where(
        LeaderDailyScore.leader_id == leader_id,
        LeaderDailyScore.day == day,
    )
    existing = (await session.execute(stmt)).scalar_one_or_none()
    if existing is not None:
        return  # already computed today; never overwrite append-only history

    session.add(
        LeaderDailyScore(
            leader_id=leader_id,
            day=day,
            profit_factor=score.profit_factor,
            win_rate=score.win_rate,
            avg_win=score.avg_win_pips,
            avg_loss=score.avg_loss_pips,
            max_drawdown=score.max_drawdown_pct,
            total_pips=score.total_pips,
            trade_count=score.trade_count,
            consistency_score=score.consistency_score,
            composite_score=score.composite,
        )
    )