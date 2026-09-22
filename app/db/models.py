"""ORM models for the platform.

Schema is intentionally append-only where possible: everything is keyed
by time, so audit history is never lost.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


class TradeSide(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"


class SignalStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    SKIPPED = "skipped"
    FAILED = "failed"


class RiskReason(str, enum.Enum):
    """Why a signal was rejected by the risk/compat gate."""


class Leader(Base):
    __tablename__ = "leaders"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # Myfxbook session/account id
    name: Mapped[str] = mapped_column(String(128))
    url: Mapped[str] = mapped_column(String(512), default="")
    audited: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    events: Mapped[list["TradeEvent"]] = relationship(back_populates="leader")


class TradeEvent(Base):
    """A normalised open/close event sourced from a leader's Myfxbook feed."""

    __tablename__ = "trade_events"
    __table_args__ = (UniqueConstraint("leader_id", "external_trade_id", name="uq_leader_trade"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    leader_id: Mapped[str] = mapped_column(ForeignKey("leaders.id"), index=True)
    external_trade_id: Mapped[str] = mapped_column(String(128), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[TradeSide] = mapped_column(Enum(TradeSide))
    volume: Mapped[float] = mapped_column(Float)
    open_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    close_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    lot_pips: Mapped[float | None] = mapped_column(Float, nullable=True)
    profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_open: Mapped[bool] = mapped_column(Boolean, default=True)
    raw_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    leader: Mapped[Leader] = relationship(back_populates="events")


class LeaderDailyScore(Base):
    """Phase 2 — per-leader, per-day performance statistics."""

    __tablename__ = "leader_daily_scores"
    __table_args__ = (UniqueConstraint("leader_id", "day", name="uq_leader_day"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    leader_id: Mapped[str] = mapped_column(ForeignKey("leaders.id"), index=True)
    day: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    profit_factor: Mapped[float] = mapped_column(Float, default=0.0)
    win_rate: Mapped[float] = mapped_column(Float, default=0.0)
    avg_win: Mapped[float] = mapped_column(Float, default=0.0)
    avg_loss: Mapped[float] = mapped_column(Float, default=0.0)
    max_drawdown: Mapped[float] = mapped_column(Float, default=0.0)
    total_pips: Mapped[float] = mapped_column(Float, default=0.0)
    trade_count: Mapped[int] = mapped_column(Integer, default=0)
    consistency_score: Mapped[float] = mapped_column(Float, default=0.0)
    composite_score: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class SignalDecision(Base):
    """Full record of every hybrid decision — leader input, meta-model P, outcome."""

    __tablename__ = "signal_decisions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    trade_event_id: Mapped[str | None] = mapped_column(
        ForeignKey("trade_events.id"), nullable=True, index=True
    )
    leader_id: Mapped[str | None] = mapped_column(ForeignKey("leaders.id"), nullable=True, index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[TradeSide] = mapped_column(Enum(TradeSide))
    leader_score: Mapped[float] = mapped_column(Float, default=0.0)
    kronos_forecast_pct: Mapped[float] = mapped_column(Float, default=0.0)  # forecast move %
    meta_model_probability: Mapped[float] = mapped_column(Float, default=0.0)
    threshold: Mapped[float] = mapped_column(Float, default=0.6)
    verb: Mapped[str] = mapped_column(String(16))  # "alert" | "copy"
    status: Mapped[SignalStatus] = mapped_column(Enum(SignalStatus), default=SignalStatus.PENDING)
    rejection_reason: Mapped[str] = mapped_column(Text, default="")
    features_json: Mapped[dict] = mapped_column(JSON, default=dict)
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)

    order: Mapped["OrderRecord | None"] = relationship(back_populates="signal")


class OrderRecord(Base):
    """A single order attempt produced by the execution engine."""

    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    signal_id: Mapped[str | None] = mapped_column(
        ForeignKey("signal_decisions.id"), nullable=True, index=True
    )
    broker_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[TradeSide] = mapped_column(Enum(TradeSide))
    volume: Mapped[float] = mapped_column(Float, default=0.0)
    entry_price: Mapped[float] = mapped_column(Float, default=0.0)
    stop_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    mode: Mapped[str] = mapped_column(String(16))  # paper | prop_eval | live | replay
    status: Mapped[str] = mapped_column(String(16), default="submitted")
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)

    signal: Mapped[SignalDecision | None] = relationship(back_populates="order")


class AuditLogEntry(Base):
    """Fully-append-only audit trail — every gate decision, override, kill switch."""

    __tablename__ = "audit_log"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    actor: Mapped[str] = mapped_column(String(64), default="system")  # system | user | scheduler
    action: Mapped[str] = mapped_column(String(64), index=True)  # signal.reject | killswitch.on ...
    detail: Mapped[dict] = mapped_column(JSON, default=dict)