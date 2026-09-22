"""Pytest fixtures shared across the suite."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.execution.killswitch import KillSwitch
from app.risk.limiter import RateLimiter
from app.risk.prop_profiles import PropProfile
from app.risk.engine import AccountState, RiskEngine


@pytest.fixture
def profile() -> PropProfile:
    return PropProfile(name="test", capital=100_000.0)


@pytest.fixture
def risk_engine(profile) -> RiskEngine:
    return RiskEngine(profile, KillSwitch(True), RateLimiter(per_minute=100, per_hour=1000))


@pytest.fixture
def healthy_account() -> AccountState:
    return AccountState(
        equity=100_000.0,
        high_water_mark=100_000.0,
        daily_start_balance=100_000.0,
        open_positions=0,
    )


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 400
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    close = 2000 + np.cumsum(rng.normal(0, 2, n))
    df = pd.DataFrame(
        {
            "open": close - 1,
            "high": close + 3,
            "low": close - 3,
            "close": close,
            "volume": rng.integers(20, 200, n),
        },
        index=idx,
    )
    return df