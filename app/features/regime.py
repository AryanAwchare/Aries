"""Market Regime Detection for Gold (XAUUSD).

Inspired by JonusNattapong/Ai-XAUUSD-Trading and institutional commodity desks.
Gold regimes dictate strategy viability:
1. TRENDING_BULL: Strong directional uptrend (EMAs aligned, ADX > 25, +DI > -DI).
2. TRENDING_BEAR: Strong directional downtrend (EMAs aligned downward, ADX > 25, -DI > +DI).
3. RANGING_LOW_VOL: Mean-reverting consolidation (ADX <= 20, tight Bollinger Bands).
4. EXPANSION_HIGH_VOL: Shock volatility expansion (ATR ratio > 1.4, macro news or geopolitical breakout).
"""
from __future__ import annotations

from enum import Enum
import numpy as np
import pandas as pd


class MarketRegime(str, Enum):
    TRENDING_BULL = "TRENDING_BULL"
    TRENDING_BEAR = "TRENDING_BEAR"
    RANGING_LOW_VOL = "RANGING_LOW_VOL"
    EXPANSION_HIGH_VOL = "EXPANSION_HIGH_VOL"
    NEUTRAL = "NEUTRAL"


def detect_regimes(df: pd.DataFrame) -> pd.Series:
    """Classify each candle in an OHLCV dataframe into a quantitative MarketRegime."""
    if len(df) < 50:
        return pd.Series(MarketRegime.NEUTRAL.value, index=df.index)

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # 1. EMAs for structural trend
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=min(200, len(df)), adjust=False).mean()

    # 2. ADX (Average Directional Index) approximation for trend strength
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=1).mean()
    atr100 = tr.rolling(100, min_periods=1).mean()
    atr_ratio = atr14 / atr100.replace(0, 1e-6)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(14, min_periods=1).mean() / atr14.replace(0, 1e-6))
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(14, min_periods=1).mean() / atr14.replace(0, 1e-6))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-6))
    adx = dx.rolling(14, min_periods=1).mean().fillna(20.0)

    # 3. Classify regime
    conditions = [
        # Shock volatility expansion
        (atr_ratio > 1.45),
        # Strong Bullish Trend
        (adx > 25) & (ema20 > ema50) & (plus_di > minus_di) & (close > ema200),
        # Strong Bearish Trend
        (adx > 25) & (ema20 < ema50) & (minus_di > plus_di) & (close < ema200),
        # Low volatility range / compression
        (adx <= 20) & (atr_ratio < 0.95),
    ]

    choices = [
        MarketRegime.EXPANSION_HIGH_VOL.value,
        MarketRegime.TRENDING_BULL.value,
        MarketRegime.TRENDING_BEAR.value,
        MarketRegime.RANGING_LOW_VOL.value,
    ]

    regimes = np.select(conditions, choices, default=MarketRegime.NEUTRAL.value)
    return pd.Series(regimes, index=df.index)
