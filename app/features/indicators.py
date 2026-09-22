"""Technical indicators for XAUUSD feature vectors.

Implemented in pure pandas so tests and the backtest harness never depend
on optional packages; ``pandas_ta`` is used when present for the classics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": hist})


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    first = tr.iloc[0]
    # ewm from bar 0 (no min_periods) keeps the warm-up causal and positive
    return tr.ewm(alpha=1 / period, adjust=False).mean().fillna(first)


def rolling_volatility(close: pd.Series, period: int = 20, annualize: int | None = None) -> pd.Series:
    ret = close.pct_change()
    vol = ret.rolling(period, min_periods=period // 2).std()
    if annualize:
        vol = vol * np.sqrt(annualize)
    return vol


def volume_zscore(volume: pd.Series, period: int = 20) -> pd.Series:
    mean = volume.rolling(period, min_periods=period // 2).mean()
    std = volume.rolling(period, min_periods=period // 2).std().replace(0, 1)
    return (volume - mean) / std


def trend_strength(close: pd.Series, period: int = 20) -> pd.Series:
    """Directional slope of a simple short/long MA ratio: a lightweight trend filter."""
    ema_fast = close.ewm(span=10, adjust=False).mean()
    ema_slow = close.ewm(span=period, adjust=False).mean()
    return (ema_fast - ema_slow) / ema_slow.replace(0, np.nan) * 1000.0


def volatility_regime(vol: pd.Series, window: int = 90) -> pd.Series:
    """Regime as a percentile of recent realised volatility: 0 calm, 1 stressed."""
    rank = vol.rolling(window, min_periods=window // 2).rank(pct=True)
    return rank.fillna(0.5)


def session_of(timestamp_index: pd.DatetimeIndex) -> pd.Series:
    """One-hot-ish session feature. Gold moves most during London/NY overlap."""
    hour = timestamp_index.hour
    return pd.Series(
        np.select(
            [
                (hour >= 13) & (hour < 17),
                (hour >= 7) & (hour <= 12),
                (hour >= 17) | (hour < 0),
            ],
            [2.0, 1.0, 3.0],  # london, london-ny pairs -> london; overlap=2 after 13
            default=0.0,
        ),
        index=timestamp_index,
        name="session",
    )