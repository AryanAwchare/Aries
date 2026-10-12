"""SQLAlchemy 2.0 ORM models for Aries (PostgreSQL / Supabase + SQLite for tests).

Supabase PostgreSQL schema:
- leaders
- leader_scores
- leader_trades
- signals
- orders
- risk_events
- system_events
- ea_orders
"""
from __future__ import annotations

import enum
import uuid
from datetime import date, datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
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
    return uuid.uuid4().hex


class TradeSide(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"


class SignalStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    SKIPPED = "skipped"
    EXECUTED = "executed"


# ---------------------------------------------------------------- leaders
class Leader(Base):
    """Myfxbook audited trader profile."""

    __tablename__ = "leaders"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # account / id
    myfxbook_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(128))
    broker: Mapped[str] = mapped_column(String(64), default="")
    account_age: Mapped[int] = mapped_column(Integer, default=0)  # in days / months
    verified_track_record: Mapped[bool] = mapped_column(Boolean, default=False)
    verified_trading_privileges: Mapped[bool] = mapped_column(Boolean, default=False)
    symbol: Mapped[str] = mapped_column(String(32), default="XAUUSD", index=True)
    strategy_type: Mapped[str] = mapped_column(String(64), default="unknown")  # trend_following, scalping, etc.
    status: Mapped[str] = mapped_column(String(32), default="watch")  # watch | live | rejected | demoted
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    url: Mapped[str] = mapped_column(String(512), default="")
    audited: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    trades: Mapped[list["LeaderTrade"]] = relationship(back_populates="leader", foreign_keys="LeaderTrade.leader_id")
    scores: Mapped[list["LeaderScoreRecord"]] = relationship(back_populates="leader", foreign_keys="LeaderScoreRecord.leader_id")

    @property
    def events(self):
        return self.trades


# ---------------------------------------------------------------- leader_scores
class LeaderScoreRecord(Base):
    """Multi-stage leader score snapshot."""

    __tablename__ = "leader_scores"
    __table_args__ = (UniqueConstraint("leader_id", "date", name="uq_leader_score_date"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    leader_id: Mapped[str] = mapped_column(ForeignKey("leaders.id"), index=True)
    date: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    profit_factor: Mapped[float] = mapped_column(Float, default=0.0)
    win_rate: Mapped[float] = mapped_column(Float, default=0.0)
    max_drawdown: Mapped[float] = mapped_column(Float, default=0.0)
    expectancy: Mapped[float] = mapped_column(Float, default=0.0)
    consistency: Mapped[float] = mapped_column(Float, default=0.0)
    longevity: Mapped[float] = mapped_column(Float, default=0.0)
    recent_score: Mapped[float] = mapped_column(Float, default=0.0)
    composite_score: Mapped[float] = mapped_column(Float, default=0.0)

    # Backwards-compatible legacy fields
    avg_win: Mapped[float] = mapped_column(Float, default=0.0)
    avg_loss: Mapped[float] = mapped_column(Float, default=0.0)
    total_pips: Mapped[float] = mapped_column(Float, default=0.0)
    trade_count: Mapped[int] = mapped_column(Integer, default=0)
    consistency_score: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    leader: Mapped[Leader] = relationship(back_populates="scores")

    @property
    def day(self) -> datetime:
        return self.date


# Backward compatibility alias
LeaderDailyScore = LeaderScoreRecord


# ---------------------------------------------------------------- leader_trades
class LeaderTrade(Base):
    """Normalised trade event from a leader."""

    __tablename__ = "leader_trades"
    __table_args__ = (UniqueConstraint("leader_id", "trade_id", name="uq_leader_trade_trade_id"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    leader_id: Mapped[str] = mapped_column(ForeignKey("leaders.id"), index=True)
    trade_id: Mapped[str] = mapped_column(String(128), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    direction: Mapped[TradeSide] = mapped_column(Enum(TradeSide))
    entry: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit: Mapped[float | None] = mapped_column(Float, nullable=True)
    lot_size: Mapped[float] = mapped_column(Float, default=0.0)
    profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, default=_now)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_open: Mapped[bool] = mapped_column(Boolean, default=True)
    raw_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    # Legacy field aliases
    @property
    def external_trade_id(self) -> str:
        return self.trade_id

    @external_trade_id.setter
    def external_trade_id(self, val: str):
        self.trade_id = val

    @property
    def open_price(self) -> float | None:
        return self.entry

    @open_price.setter
    def open_price(self, val: float | None):
        self.entry = val

    @property
    def close_price(self) -> float | None:
        return self.exit

    @close_price.setter
    def close_price(self, val: float | None):
        self.exit = val

    @property
    def volume(self) -> float:
        return self.lot_size

    @volume.setter
    def volume(self, val: float):
        self.lot_size = val

    @property
    def side(self) -> TradeSide:
        return self.direction

    @side.setter
    def side(self, val: TradeSide):
        self.direction = val

    leader: Mapped[Leader] = relationship(back_populates="trades")


# Backward compatibility alias
TradeEvent = LeaderTrade


# ---------------------------------------------------------------- signals
class Signal(Base):
    """Signal decision generated by strategy aggregator."""

    __tablename__ = "signals"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    symbol: Mapped[str] = mapped_column(String(32), default="XAUUSD", index=True)
    timeframe: Mapped[str] = mapped_column(String(16), default="M15")
    leader_direction: Mapped[str | None] = mapped_column(String(16), nullable=True)
    kronos_direction: Mapped[str | None] = mapped_column(String(16), nullable=True)
    kronos_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    xgb_probability: Mapped[float] = mapped_column(Float, default=0.0)
    regime: Mapped[str] = mapped_column(String(32), default="ranging")
    final_signal: Mapped[str] = mapped_column(String(16), default="NO_TRADE")  # BUY | SELL | NO_TRADE
    contributing_factors: Mapped[dict] = mapped_column(JSON, default=dict)

    # Legacy fields
    trade_event_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    leader_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    side: Mapped[TradeSide | None] = mapped_column(Enum(TradeSide), nullable=True)
    leader_score: Mapped[float] = mapped_column(Float, default=0.0)
    kronos_forecast_pct: Mapped[float] = mapped_column(Float, default=0.0)
    meta_model_probability: Mapped[float] = mapped_column(Float, default=0.0)
    threshold: Mapped[float] = mapped_column(Float, default=0.6)
    verb: Mapped[str] = mapped_column(String(16), default="alert")  # alert | copy | skip
    status: Mapped[SignalStatus] = mapped_column(Enum(SignalStatus), default=SignalStatus.PENDING)
    rejection_reason: Mapped[str] = mapped_column(Text, default="")
    features_json: Mapped[dict] = mapped_column(JSON, default=dict)
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)

    order: Mapped["OrderRecord | None"] = relationship(back_populates="signal")


SignalDecision = Signal


# ---------------------------------------------------------------- orders
class OrderRecord(Base):
    """Order attempt and fill record."""

    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    signal_id: Mapped[str | None] = mapped_column(
        ForeignKey("signals.id"), nullable=True, index=True
    )
    broker_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    symbol: Mapped[str] = mapped_column(String(32), default="XAUUSD", index=True)
    direction: Mapped[TradeSide] = mapped_column(Enum(TradeSide))
    volume: Mapped[float] = mapped_column(Float, default=0.0)
    entry: Mapped[float] = mapped_column(Float, default=0.0)
    sl: Mapped[float | None] = mapped_column(Float, nullable=True)
    tp: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="submitted")  # submitted | filled | failed | rejected
    mode: Mapped[str] = mapped_column(String(16), default="paper")  # paper | prop_eval | live | replay
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    filled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Legacy properties
    @property
    def side(self) -> TradeSide:
        return self.direction

    @side.setter
    def side(self, val: TradeSide):
        self.direction = val

    @property
    def entry_price(self) -> float:
        return self.entry

    @entry_price.setter
    def entry_price(self, val: float):
        self.entry = val

    @property
    def stop_loss(self) -> float | None:
        return self.sl

    @stop_loss.setter
    def stop_loss(self, val: float | None):
        self.sl = val

    @property
    def take_profit(self) -> float | None:
        return self.tp

    @take_profit.setter
    def take_profit(self, val: float | None):
        self.tp = val

    signal: Mapped[Signal | None] = relationship(back_populates="order")


# ---------------------------------------------------------------- risk_events
class RiskEvent(Base):
    """Risk management event log."""

    __tablename__ = "risk_events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    event: Mapped[str] = mapped_column(String(64), index=True)
    severity: Mapped[str] = mapped_column(String(32), default="INFO")  # INFO | WARNING | CRITICAL
    drawdown: Mapped[float] = mapped_column(Float, default=0.0)
    daily_loss: Mapped[float] = mapped_column(Float, default=0.0)
    action: Mapped[str] = mapped_column(String(64), default="")
    detail: Mapped[dict] = mapped_column(JSON, default=dict)


# ---------------------------------------------------------------- system_events (general audit)
class SystemEvent(Base):
    """System-wide operational and audit event log."""

    __tablename__ = "system_events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    component: Mapped[str] = mapped_column(String(64), index=True)  # executor | strategy | leader_pool | etc.
    event: Mapped[str] = mapped_column(String(64), index=True)
    detail: Mapped[dict] = mapped_column(JSON, default=dict)

    # Legacy compatibility fields
    @property
    def ts(self) -> datetime:
        return self.timestamp

    @property
    def actor(self) -> str:
        return self.component

    @property
    def action(self) -> str:
        return self.event


AuditLogEntry = SystemEvent


# ---------------------------------------------------------------- ea_orders
class EaOrder(Base):
    """Bridge order polled by the local MetaTrader EA."""

    __tablename__ = "ea_orders"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    external_id: Mapped[str] = mapped_column(String(64), index=True)
    order_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[TradeSide] = mapped_column(Enum(TradeSide))
    volume: Mapped[float] = mapped_column(Float, default=0.0)
    stop_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending|dispatched|filled|rejected
    broker_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    fill_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    dispatched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)