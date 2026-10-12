"""Shared leader-signal types (keep imports light — used by workers & API)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RawLeaderTrade:
    external_id: str
    symbol: str
    side: str                     # "buy" | "sell"
    volume: float
    open_price: float
    close_price: float | None = None
    profit: float | None = None
    lots: float | None = None
    pips: float | None = None
    opened_at: datetime | None = None
    closed_at: datetime | None = None
    is_open: bool = True
    raw: dict = field(default_factory=dict)


@dataclass
class LeaderProfile:
    id: str
    name: str
    url: str = ""
    audited: bool = False
    broker: str = ""
    account_age: int = 0  # In days
    verified_track_record: bool = False
    verified_trading_privileges: bool = False
    symbol: str = "XAUUSD"
    strategy_type: str = "unknown"
    status: str = "watch"
    is_active: bool = True


@dataclass
class LeaderScore:
    profit_factor: float = 0.0
    win_rate: float = 0.0
    avg_win_pips: float = 0.0
    avg_loss_pips: float = 0.0
    max_drawdown_pct: float = 0.0
    total_pips: float = 0.0
    trade_count: int = 0
    consistency_score: float = 0.0
    expectancy: float = 0.0
    longevity: float = 0.0
    recent_score: float = 0.0
    composite_score: float = 0.0
    composite: float = 0.0

    def __post_init__(self):
        if self.composite_score != 0.0 and self.composite == 0.0:
            self.composite = self.composite_score
        elif self.composite != 0.0 and self.composite_score == 0.0:
            self.composite_score = self.composite


@dataclass
class EligibilityResult:
    eligible: bool
    rejection_reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)