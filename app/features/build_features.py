"""Assembly of tabular feature vectors for the meta-model.

A single point-in-time feature vector combines:
  * technical state       (RSI, MACD, ATR, volatility, regime, session)
  * COT positioning       (money-manager net z-score, OI ratios)
  * Kronos forecast       (forecast % over horizon + model confidence proxy)
  * calendar context      (hour-of-day, day-of-week)

No future information may leak into a vector — all transforms are causal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.ohlcv import OHLCV_COLUMNS
from . import indicators as ind


def calculate_indicator_block(df: pd.DataFrame) -> pd.DataFrame:
    """Append indicator columns to an OHLCV frame (all causal, no lookahead)."""
    if df.empty:
        return df.copy()
    out = df.copy()
    out["rsi"] = ind.rsi(out["close"])
    macd = ind.macd(out["close"])
    out["macd"] = macd["macd"]
    out["macd_hist"] = macd["macd_hist"]
    out["atr"] = ind.atr(out)
    out["atr_pct"] = out["atr"] / out["close"].replace(0, np.nan)
    # vol_20: raw 20-bar return std (no over-annualization)
    out["vol_20"] = ind.rolling_volatility(out["close"], period=20)
    out["vol_regime"] = ind.volatility_regime(out["vol_20"].fillna(0))
    out["volume_z"] = ind.volume_zscore(out["volume"])
    out["trend_strength"] = ind.trend_strength(out["close"])
    out["session"] = _session_of(out.index)
    out["hour"] = out.index.hour / 24.0
    out["dow_cos"] = np.cos(2 * np.pi * out.index.dayofweek / 7)
    out["dow_sin"] = np.sin(2 * np.pi * out.index.dayofweek / 7)

    # --- Price-relative features for directional signal ---
    roll_high = out["high"].rolling(20, min_periods=5).max()
    roll_low  = out["low"].rolling(20, min_periods=5).min()
    rng = (roll_high - roll_low).replace(0, np.nan)
    out["close_vs_high20"] = ((out["close"] - roll_high) / rng).fillna(0.0)
    out["close_vs_low20"]  = ((out["close"] - roll_low)  / rng).fillna(0.0)

    candle_range = (out["high"] - out["low"]).replace(0, np.nan)
    out["body_to_range"] = ((out["close"] - out["open"]).abs() / candle_range).fillna(0.0)
    out["upper_wick"]    = ((out["high"] - out[["close", "open"]].max(axis=1)) / candle_range).fillna(0.0)
    out["lower_wick"]    = ((out[["close", "open"]].min(axis=1) - out["low"]) / candle_range).fillna(0.0)

    # Gap from previous close (session-open jumps)
    prev_close = out["close"].shift(1)
    out["gap_from_prev"] = ((out["open"] - prev_close) / prev_close.replace(0, np.nan)).fillna(0.0)

    # Fill NaNs
    out["atr_pct"]        = out["atr_pct"].fillna(0.0)
    out["vol_20"]         = out["vol_20"].fillna(0.0)
    out["trend_strength"] = out["trend_strength"].fillna(0.0)
    out["volume_z"]       = out["volume_z"].fillna(0.0)
    return out


def _session_of(index: pd.DatetimeIndex) -> pd.Series:
    """Session indicator: 0 = Asian, 1 = London, 2 = New York."""
    hours = index.hour
    return pd.Series(
        np.select(
            [
                (hours >= 0) & (hours < 8),
                (hours >= 8) & (hours < 13),
                (hours >= 13) & (hours < 21),
            ],
            [0, 1, 2],
            default=0,
        ),
        index=index,
    )


FEATURE_COLUMNS: list[str] = [
    "rsi",
    "macd",
    "macd_hist",
    "atr_pct",
    "vol_20",
    "vol_regime",
    "volume_z",
    "trend_strength",
    "session",
    "hour",
    "dow_cos",
    "dow_sin",
    # Price-relative features (new — carry directional edge)
    "close_vs_high20",
    "close_vs_low20",
    "body_to_range",
    "upper_wick",
    "lower_wick",
    "gap_from_prev",
    # External signals
    "cot_mm_net_z",
    "cot_mm_net_oi",
    "cot_producer_net_oi",
    "kronos_forecast_pct",
    "kronos_conf",
]


def build_feature_row(
    ohlcv: pd.DataFrame,
    cot: dict[str, float] | None = None,
    kronos_forecast_pct: float = 0.0,
    kronos_conf: float = 0.0,
    index: pd.Timestamp | None = None,
) -> dict[str, float]:
    df = calculate_indicator_block(ohlcv)
    if df.empty:
        return {c: 0.0 for c in FEATURE_COLUMNS}
    row = df.loc[index] if index is not None else df.iloc[-1]
    out: dict[str, float] = {}
    for col in FEATURE_COLUMNS:
        if col in row:
            out[col] = float(row[col])
        elif cot and col in cot:
            out[col] = float(cot[col])
        elif col == "kronos_forecast_pct":
            out[col] = float(kronos_forecast_pct)
        elif col == "kronos_conf":
            out[col] = float(kronos_conf)
        else:
            out[col] = 0.0
    return out


def label_trades(
    df: pd.DataFrame,
    horizon_bars: int = 16,
    side: str = "buy",
    threshold: float = 0.0005,
) -> pd.Series:
    """Label whether a directional trade was profitable after ``horizon_bars``.

    ``threshold`` is the fractional move that counts as a win. The default
    0.0005 (~$1.35 on XAUUSD at $2700) is realistic for a 4-hour M15 trade with
    a 5-point ATR stop. Shorter timeframes need a smaller threshold; see
    :mod:`app.features.timeframe`. The threshold must clear the round-trip
    transaction cost, otherwise positives are untradeable.
    """
    if len(df) <= horizon_bars:
        return pd.Series(np.nan, index=df.index)
    close = df["close"]
    future = close.shift(-horizon_bars)
    ret = (future - close) / close
    if side == "buy":
        label = (ret > threshold).astype(float)
    else:
        label = (ret < -threshold).astype(float)
    label.iloc[-horizon_bars:] = np.nan
    return label