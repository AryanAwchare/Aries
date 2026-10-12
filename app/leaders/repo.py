"""Persistence helpers for leaders + trade events."""
from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import Leader, LeaderScoreRecord, LeaderTrade
from .models import LeaderProfile, RawLeaderTrade


async def get_or_create_leader(session: AsyncSession, profile: LeaderProfile) -> Leader:
    leader = await session.get(Leader, profile.id)
    if leader is None:
        leader = Leader(
            id=profile.id,
            name=profile.name,
            url=profile.url,
            audited=profile.audited,
            broker=profile.broker,
            account_age=profile.account_age,
            verified_track_record=profile.verified_track_record,
            verified_trading_privileges=profile.verified_trading_privileges,
            symbol=profile.symbol,
            strategy_type=profile.strategy_type,
            status=profile.status,
            is_active=profile.is_active,
        )
        session.add(leader)
        await session.flush()
    else:
        # Update mutable fields
        leader.verified_track_record = profile.verified_track_record
        leader.verified_trading_privileges = profile.verified_trading_privileges
        if profile.strategy_type != "unknown":
            leader.strategy_type = profile.strategy_type
        leader.status = profile.status
        leader.last_seen_at = datetime.now(timezone.utc)
    return leader


async def upsert_trade_event(session: AsyncSession, leader_id: str, trade: RawLeaderTrade) -> LeaderTrade | None:
    """Insert-or-skip a normalised leader trade. Returns None if already present."""
    stmt = select(LeaderTrade).where(
        LeaderTrade.leader_id == leader_id,
        LeaderTrade.trade_id == trade.external_id,
    )
    existing = (await session.execute(stmt)).scalar_one_or_none()
    if existing is not None:
        return None

    event = LeaderTrade(
        leader_id=leader_id,
        trade_id=trade.external_id,
        symbol=trade.symbol,
        direction=trade.side.lower(),  # type: ignore[arg-type]
        lot_size=trade.volume,
        entry=trade.open_price,
        exit=trade.close_price,
        profit=trade.profit,
        opened_at=trade.opened_at or datetime.now(timezone.utc),
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
    score_dt = day if isinstance(day, datetime) else datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
    stmt = select(LeaderScoreRecord).where(
        LeaderScoreRecord.leader_id == leader_id,
        LeaderScoreRecord.date == score_dt,
    )
    existing = (await session.execute(stmt)).scalar_one_or_none()
    if existing is not None:
        return  # already computed today

    session.add(
        LeaderScoreRecord(
            leader_id=leader_id,
            date=score_dt,
            profit_factor=score.profit_factor,
            win_rate=score.win_rate,
            avg_win=score.avg_win_pips,
            avg_loss=score.avg_loss_pips,
            max_drawdown=score.max_drawdown_pct,
            total_pips=score.total_pips,
            trade_count=score.trade_count,
            consistency_score=score.consistency_score,
            expectancy=score.expectancy,
            consistency=score.consistency_score,
            longevity=score.longevity,
            recent_score=score.recent_score,
            composite_score=score.composite_score or score.composite,
        )
    )