"""Tests for Gold Macro Ingestion, News Blackout Filter, and Market Regime Detection."""
from datetime import datetime, timezone
import pandas as pd
import numpy as np

from app.features.regime import detect_regimes, MarketRegime
from app.risk.news_filter import NewsFilter, NewsEvent
from app.data.macro import MacroProvider


def test_market_regime_detection():
    # Synthetic trending dataframe
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    # Strong upward trend
    closes = np.linspace(2000, 2100, n)
    highs = closes + 2.0
    lows = closes - 2.0
    opens = closes - 0.5
    df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": 100}, index=dates)

    regimes = detect_regimes(df)
    assert len(regimes) == n
    assert isinstance(regimes.iloc[-1], str)


def test_news_blackout_filter():
    filter = NewsFilter(blackout_before_minutes=20, blackout_after_minutes=15)

    # 1. Test custom event: US CPI at 13:30 UTC
    cpi_time = datetime(2024, 3, 12, 13, 30, tzinfo=timezone.utc)
    filter.events.append(NewsEvent(name="US CPI", impact="HIGH", timestamp=cpi_time))

    # 10 mins before CPI -> inside blackout
    t_inside = datetime(2024, 3, 12, 13, 20, tzinfo=timezone.utc)
    is_blocked, reason = filter.is_blackout(t_inside)
    assert is_blocked is True
    assert "US CPI" in reason

    # 1 hour before CPI -> outside blackout
    t_outside = datetime(2024, 3, 12, 12, 0, tzinfo=timezone.utc)
    is_blocked, reason = filter.is_blackout(t_outside)
    assert is_blocked is False


def test_macro_context_bias():
    provider = MacroProvider()
    ctx = provider.get_context()
    assert -1.0 <= ctx.macro_bias <= 1.0
    assert ctx.dxy_close > 0
    assert ctx.us10y_yield > 0
