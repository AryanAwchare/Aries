"""Technical indicators for XAUUSD feature vectors.

Implemented in pure pandas so tests and the backtest harness never depend
on optional packages; ``pandas_ta`` is used when present for the classics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    if close.empty:
        return pd.Series([], dtype=float, index=close.index)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    if close.empty:
        return pd.DataFrame({"macd": [], "macd_signal": [], "macd_hist": []}, index=close.index)
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": hist})


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if df.empty:
        return pd.Series([], dtype=float, index=df.index)
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
    first = tr.iloc[0] if not tr.empty else 0.0
    return tr.ewm(alpha=1 / period, adjust=False).mean().fillna(first)


def rolling_volatility(close: pd.Series, period: int = 20, annualize: int | None = None) -> pd.Series:
    if close.empty:
        return pd.Series([], dtype=float, index=close.index)
    ret = close.pct_change()
    vol = ret.rolling(period, min_periods=period // 2).std()
    if annualize:
        vol = vol * np.sqrt(annualize)
    return vol.fillna(0.0)


def volatility_regime(close: pd.Series, period: int = 20) -> pd.Series:
    if close.empty:
        return pd.Series([], dtype=float, index=close.index)
    vol = rolling_volatility(close, period=period)
    vol_mean = vol.rolling(100, min_periods=20).mean()
    vol_std = vol.rolling(100, min_periods=20).std().replace(0, np.nan)
    z = ((vol - vol_mean) / vol_std).fillna(0.0)
    return pd.cut(
        z,
        bins=[-np.inf, -0.5, 0.5, 2.0, np.inf],
        labels=[0, 1, 2, 3],
    ).astype(float).fillna(1.0)


def trend_strength(close: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    if close.empty:
        return pd.Series([], dtype=float, index=close.index)
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    raw = (ema_fast - ema_slow) / ema_slow.replace(0, np.nan)
    return raw.fillna(0.0)


def volume_zscore(volume: pd.Series, period: int = 50) -> pd.Series:
    if volume.empty:
        return pd.Series([], dtype=float, index=volume.index)
    mean = volume.rolling(period, min_periods=10).mean()
    std = volume.rolling(period, min_periods=10).std().replace(0, np.nan)
    return ((volume - mean) / std).fillna(0.0)


def parkinson_volatility(high: pd.Series, low: pd.Series, period: int = 20) -> pd.Series:
    """Range-based volatility (Parkinson 1980) — no close-to-close dependence."""
    if high.empty:
        return pd.Series([], dtype=float, index=high.index)
    hl = np.log(high / low.replace(0, np.nan))
    var = (hl**2).rolling(period, min_periods=max(2, period // 2)).mean() / (4 * np.log(2))
    return np.sqrt(var.clip(lower=0.0)).fillna(0.0)


def garman_klass_volatility(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
) -> pd.Series:
    """OHLC volatility estimator (Garman & Klass 1980)."""
    if close.empty:
        return pd.Series([], dtype=float, index=close.index)
    hl = np.log(high / low.replace(0, np.nan))
    co = np.log(close / open_.replace(0, np.nan))
    var = (0.5 * hl**2 - (2 * np.log(2) - 1) * co**2).rolling(
        period, min_periods=max(2, period // 2)
    ).mean()
    return np.sqrt(var.clip(lower=0.0)).fillna(0.0)


def efficiency_ratio(close: pd.Series, period: int = 10) -> pd.Series:
    """Kaufman efficiency ratio: net move / summed absolute path (0..1)."""
    if close.empty:
        return pd.Series([], dtype=float, index=close.index)
    change = (close - close.shift(period)).abs()
    path = close.diff().abs().rolling(period, min_periods=max(2, period // 2)).sum()
    return (change / path.replace(0, np.nan)).fillna(0.0).clip(0.0, 1.0)