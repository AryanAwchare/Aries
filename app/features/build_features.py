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
    out = df.copy()
    out["rsi"] = ind.rsi(out["close"])
    macd = ind.macd(out["close"])
    out["macd"] = macd["macd"]
    out["macd_hist"] = macd["macd_hist"]
    out["atr"] = ind.atr(out)
    out["atr_pct"] = out["atr"] / out["close"].replace(0, np.nan)
    out["vol_20"] = ind.rolling_volatility(out["close"]) * np.sqrt(52 * 24 * 4)  # ~ per-bar scales
    out["vol_regime"] = ind.volatility_regime(out["vol_20"].fillna(0), window=90)
    out["volume_z"] = ind.volume_zscore(out["volume"])
    out["trend_strength"] = ind.trend_strength(out["close"])
    out["session"] = ind.session_of(out.index)
    out["hour"] = out.index.hour / 24.0
    out["dow_cos"] = np.cos(2 * np.pi * out.index.dayofweek / 7)
    out["dow_sin"] = np.sin(2 * np.pi * out.index.dayofweek / 7)
    # Indicator warm-up: the first N bars of ewm/rolling statistics are NaN.
    # Replace with neutral values so feature rows exist for every bar (values
    # are still causal — NaN just propagates infinitely-regressed means).
    out["atr_pct"] = out["atr_pct"].fillna(0.0)
    out["vol_20"] = out["vol_20"].fillna(0.0)
    out["volume_z"] = out["volume_z"].fillna(0.0)
    out["trend_strength"] = out["trend_strength"].fillna(0.0)
    out["macd"] = out["macd"].fillna(0.0)
    out["macd_hist"] = out["macd_hist"].fillna(0.0)
    return out


# Feature order must be stable across train and inference.
FEATURE_COLUMNS = [
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


def _cot_defaults() -> dict[str, float]:
    return {
        "cot_mm_net_z": 0.0,
        "cot_mm_net_oi": 0.0,
        "cot_producer_net_oi": 0.0,
    }


def build_feature_row(
    df: pd.DataFrame,
    index: pd.Timestamp | None = None,
    cot: dict[str, float] | None = None,
    kronos_forecast_pct: float = 0.0,
    kronos_conf: float = 0.0,
) -> dict[str, float]:
    """One point-in-time feature row.

    Parameters
    ----------
    df : OHLCV frame (may already include indicator columns)
    index : timestamp to snapshot; defaults to the last row
    cot : output of ``cot_feature_vector`` (defaults to neutral zeros)
    kronos_forecast_pct : Kronos forecasted % move over the prediction horizon
    kronos_conf : 0..1 model-confidence proxy
    """
    if "rsi" not in df.columns:
        df = calculate_indicator_block(df)

    if index is None:
        idx = df.index[-1]
    else:
        idx = pd.Timestamp(index)
        if idx not in df.index:
            # snapshot the most-recent strictly-prior bar (causal)
            prior = df.index[df.index <= idx]
            if len(prior) == 0:
                raise ValueError(f"index {idx} precedes the first bar")
            idx = prior[-1]

    row = {c: df.loc[idx, c] for c in FEATURE_COLUMNS if c in df.columns}
    row.update(_cot_defaults())
    if cot:
        row.update({k: float(v) for k, v in cot.items() if k in FEATURE_COLUMNS})
    row["kronos_forecast_pct"] = float(kronos_forecast_pct)
    row["kronos_conf"] = float(kronos_conf)
    # ensure all columns present in the exact order
    return {c: float(row.get(c, 0.0)) for c in FEATURE_COLUMNS}


def future_return(df: pd.DataFrame, horizon_bars: int = 8) -> pd.Series:
    """Forward return over ``horizon_bars`` — for building labels. NOT causal."""
    return df["close"].shift(-horizon_bars) / df["close"] - 1.0


def label_trades(df: pd.DataFrame, horizon_bars: int = 8, side: str = "buy") -> pd.Series:
    """Label each bar as profitable-at-``horizon_bars`` for the given side.

    Bars without a full future window stay NaN so training can drop them.
    """
    fwd = future_return(df, horizon_bars)
    labels = pd.Series(float("nan"), index=df.index, dtype="float64")
    ok = fwd.notna()
    if side == "buy":
        labels.loc[ok] = (fwd[ok] > 0).astype("float64")
    else:
        labels.loc[ok] = (fwd[ok] < 0).astype("float64")
    return labels