"""Prop-firm profile models + loader.

One YAML per firm's ruleset. Switching firms is a config change, never a
code change. Internal limits intentionally sit BELOW the firm's hard limits.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ConsistencyRule(BaseModel):
    max_single_day_profit_pct: float = 1.0  # single day may own at most this share of total profit
    enabled: bool = False


class PropProfile(BaseModel):
    name: str
    description: str = ""
    broker: str = ""

    # Firm hard limits
    capital: float = 100_000.0
    profit_target: float = 0.10
    daily_loss_limit_pct: float = 0.03
    max_drawdown_pct: float = 0.10
    trailing_drawdown: bool = False
    max_open_positions: int = 3
    max_leverage: float = 100.0

    # Allowed tooling (the prop compatibility gate reads these)
    copy_trading_allowed: bool = True
    third_party_signals_allowed: bool = True
    ea_allowed: bool = True
    api_allowed: bool = True
    news_trading_allowed: bool = True
    trading_hours_utc: str = "00:00-23:59"
    consistency_rule: ConsistencyRule = Field(default_factory=ConsistencyRule)

    # Internal (bot) limits — always stricter than the firm's
    emergency_stop_drawdown_pct: float = 0.06
    operating_ceiling_drawdown_pct: float = 0.04
    daily_loss_soft_limit_pct: float = 0.02
    position_risk_pct: float = 0.01
    max_orders_per_minute: int = 2
    max_orders_per_hour: int = 10
    lot_step: float = 0.01
    min_lot: float = 0.01
    max_lot: float = 10.0


def load_profile(path: Path) -> PropProfile:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return PropProfile.model_validate(data)


def load_profile_dir(directory: Path) -> dict[str, PropProfile]:
    profiles: dict[str, PropProfile] = {}
    for path in sorted(directory.glob("*.yaml")):
        profile = load_profile(path)
        profiles[profile.name] = profile
    return profiles