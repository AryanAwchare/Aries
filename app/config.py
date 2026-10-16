"""Application configuration via pydantic-settings.

All values come from environment variables / .env. A trading platform
must be explicit about its mode — `TRADING_MODE` is validated here so a
typo can never silently become ``live``.
"""
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


class TradingMode(str, Enum):
    BACKTEST = "backtest"
    REPLAY = "replay"
    PAPER = "paper"
    PROP_EVAL = "prop_eval"
    LIVE = "live"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Mode ---------------------------------------------------------
    trading_mode: TradingMode = TradingMode.PAPER
    active_prop_profile: str = "default_demo"

    # --- Database ------------------------------------------------------
    database_url: str = "sqlite+aiosqlite:///./data/app.db"
    supabase_db_url: str = ""

    # --- Market data ----------------------------------------------------
    market_symbol: str = "XAUUSD"
    market_timeframe: str = "M15"
    metaapi_token: str = ""
    metas_api_demo_account_id: str = ""
    metas_api_prop_account_id: str = ""

    # --- Execution driver ------------------------------------------------
    execution_driver: str = "paper"
    ea_api_key: str = ""
    ea_order_timeout_seconds: int = 60

    # --- Control plane ----------------------------------------------------
    # When set, mutating control-plane endpoints require this key
    # (X-API-Key header or Authorization: Bearer). Empty = unauthenticated
    # (local single-user mode); the app logs a CRITICAL warning at startup.
    ops_api_key: str = ""

    # --- Myfxbook ---------------------------------------------------------
    myfxbook_email: str = ""
    myfxbook_password: str = ""
    myfxbook_leaders: str = ""
    myfxbook_poll_interval_seconds: int = 300

    # --- Leader roster (two-tier discovery) --------------------------------
    leader_roster_path: str = "data/roster.json"
    max_live_leaders: int = 5
    max_watch_leaders: int = 50
    live_leader_pool_size: int = 5
    leader_min_observations: int = 3
    leader_rescore_interval_hours: int = 4

    # --- Leader discovery (Phase 2.1 gold screening) -----------------------
    leader_min_profit_factor: float = 1.5
    leader_max_drawdown_pct: float = 20.0
    leader_min_account_age_days: int = 365
    leader_min_discovery_trades: int = 200
    leader_require_real_account: bool = True
    leader_seed_file: str = "data/leader_seeds.txt"

    # --- Leader Eligibility & Scoring (Phase 3) ---------------------------
    min_account_age_days: int = 180
    min_closed_trades: int = 100
    max_account_drawdown_ceiling: float = 25.0
    max_inactivity_days: int = 30
    leader_consecutive_loss_limit: int = 3
    leader_live_drawdown_limit_pct: float = 5.0

    # Configurable Stage B weights
    weight_risk_adjusted_return: float = 0.20
    weight_profit_factor: float = 0.20
    weight_drawdown_quality: float = 0.15
    weight_consistency: float = 0.15
    weight_expectancy: float = 0.10
    weight_longevity: float = 0.10
    weight_recent_performance: float = 0.10

    # --- Kronos / forecast -------------------------------------------------
    kronos_model_path: str = ""
    predict_horizon_bars: int = 24

    # --- Telegram ----------------------------------------------------------
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # --- Research windows ---------------------------------------------------
    train_start: str = "2020-01-01"
    train_end: str = "2023-12-31"
    validation_start: str = "2024-01-01"
    validation_end: str = "2024-12-31"
    test_start: str = "2025-01-01"
    test_end: str = "2026-12-31"

    # --- Signal thresholds ----------------------------------------------------
    meta_model_prob_threshold: float = 0.60
    min_leader_score: float = 3.0

    # --- Risk ------------------------------------------------------------------
    risk_capital_base: float = 100_000.0
    position_risk_percent: float = 0.01
    max_drawdown_percent: float = 0.06
    daily_loss_limit_percent: float = 0.03
    kill_switch_enabled: bool = True

    @field_validator("trading_mode", mode="before")
    @classmethod
    def _normalise_mode(cls, v: object) -> TradingMode:
        if isinstance(v, TradingMode):
            return v
        if isinstance(v, str):
            return TradingMode(v.strip().lower())
        raise ValueError(f"invalid trading mode: {v!r}")

    @field_validator("myfxbook_leaders", mode="before")
    @classmethod
    def _empty_to_default(cls, v: object) -> str:
        return v if isinstance(v, str) else ""

    @field_validator("execution_driver", mode="before")
    @classmethod
    def _normalise_driver(cls, v: object) -> str:
        if isinstance(v, str):
            v = v.strip().lower()
            if v not in ("metaapi", "ea", "paper"):
                raise ValueError(f"invalid execution_driver: {v!r}")
        return v or "paper"

    # ---- computed helpers -------------------------------------------------------

    @property
    def effective_db_url(self) -> str:
        """Prefers SUPABASE_DB_URL if configured, otherwise falls back to database_url."""
        env_supabase = os.getenv("SUPABASE_DB_URL") or self.supabase_db_url
        if env_supabase and env_supabase.strip():
            url = env_supabase.strip()
            # asyncpg adapter normalization if postgresql:// is passed
            if url.startswith("postgresql://") and not url.startswith("postgresql+asyncpg://"):
                url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
            elif url.startswith("postgres://"):
                url = url.replace("postgres://", "postgresql+asyncpg://", 1)
            return url
        return self.database_url

    @property
    def leader_ids(self) -> list[str]:
        """Curated, audited Myfxbook leader identifiers."""
        return [s.strip() for s in self.myfxbook_leaders.split(",") if s.strip()]

    @property
    def is_execution_mode(self) -> bool:
        return self.trading_mode in (TradingMode.PAPER, TradingMode.PROP_EVAL, TradingMode.LIVE)

    @property
    def prop_profile_path(self) -> Path:
        return BASE_DIR / "app" / "risk" / "prop_profiles" / f"{self.active_prop_profile}.yaml"

    @property
    def roster_path(self) -> Path:
        p = Path(self.leader_roster_path)
        return p if p.is_absolute() else BASE_DIR / p

    @property
    def leader_seed_path(self) -> Path:
        p = Path(self.leader_seed_file)
        return p if p.is_absolute() else BASE_DIR / p

    def ensure_dirs(self) -> None:
        (DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "models").mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "replay").mkdir(parents=True, exist_ok=True)
        (BASE_DIR / "logs").mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
