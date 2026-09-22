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
    composite: float = 0.0