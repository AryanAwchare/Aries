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
    out["vol_20"] = ind.rolling_volatility(out["close"]) * np.sqrt(52 * 24 * 4)  # ~ per-bar scales
    out["vol_regime"] = ind.volatility_regime(out["vol_20"].fillna(0))
    out["volume_z"] = ind.volume_zscore(out["volume"])
    out["trend_strength"] = ind.trend_strength(out["close"])
    out["session"] = _session_of(out.index)
    out["hour"] = out.index.hour / 24.0
    out["dow_cos"] = np.cos(2 * np.pi * out.index.dayofweek / 7)
    out["dow_sin"] = np.sin(2 * np.pi * out.index.dayofweek / 7)
    out["atr_pct"] = out["atr_pct"].fillna(0.0)
    out["vol_20"] = out["vol_20"].fillna(0.0)
    out["trend_strength"] = out["trend_strength"].fillna(0.0)
    out["volume_z"] = out["volume_z"].fillna(0.0)
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


def label_trades(df: pd.DataFrame, horizon_bars: int = 8, side: str = "buy") -> pd.Series:
    if len(df) <= horizon_bars:
        return pd.Series(np.nan, index=df.index)
    close = df["close"]
    future = close.shift(-horizon_bars)
    ret = (future - close) / close
    if side == "buy":
        label = (ret > 0.001).astype(float)
    else:
        label = (ret < -0.001).astype(float)
    label.iloc[-horizon_bars:] = np.nan
    return label