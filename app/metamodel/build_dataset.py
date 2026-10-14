"""Build point-in-time training frames for the meta-model.

The meta-model learns: *given a proposed directional trade (side) and the
current market context, what is P(trade is profitable after H bars)?*

Historical "candidate events" stand in for leader trades — during live
operation the same feature row is built from a real leader event. This is
deliberate: the leader signal and the ML gate stay separable, exactly as
the architecture requires.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.build_features import (
    FEATURE_COLUMNS,
    calculate_indicator_block,
    label_trades,
)

# Canonical feature list for model input (side encoded separately).
MODEL_FEATURES = FEATURE_COLUMNS + ["side_buy", "side_sell"]


def candidate_events(df: pd.DataFrame, min_move_pct: float = 0.05) -> pd.DataFrame:
    """Momentum-ish candidate points (bar-level 'a leader acted here' proxy).

    Returns rows ``[index, side]`` for bars whose candle body and EMA-spread
    structure imply a directional bet. Not a strategy — just a way to generate
    labelled examples for training.
    """
    if "rsi" not in df.columns:
        df = calculate_indicator_block(df)
    body = (df["close"] - df["open"]) / df["open"] * 100.0
    ema20 = df["close"].ewm(span=20, adjust=False).mean()
    spread = (df["close"] - ema20) / ema20.replace(0, 1e-12) * 1000.0
    moving = body.abs() > min_move_pct
    directional = ((body > 0) & (spread > 0)) | ((body < 0) & (spread < 0))
    candidates = df.loc[moving & directional]
    side = np.where(candidates["close"] >= candidates["open"], "buy", "sell")
    # Indexed by bar timestamp — callers can do ``ts in events.index``.
    return pd.DataFrame({"side": side}, index=candidates.index)


def build_training_data(
    ohlcv: pd.DataFrame,
    cot: dict[str, float] | None = None,
    horizon_bars: int = 16,
    min_move_pct: float = 0.05,
    threshold: float = 0.0005,
) -> pd.DataFrame:
    """Feature rows + ``label`` for every candidate event.

    Returns columns: MODEL_FEATURES + ``label`` (1 if the side was profitable
    over ``horizon_bars`` by at least ``threshold``).
    """
    df = calculate_indicator_block(ohlcv)
    events = candidate_events(df, min_move_pct)
    if events.empty:
        return pd.DataFrame(columns=MODEL_FEATURES + ["label"])

    valid_idx = events.index.intersection(df.index)
    if valid_idx.empty:
        return pd.DataFrame(columns=MODEL_FEATURES + ["label"])

    buy_labels = label_trades(df, horizon_bars, side="buy", threshold=threshold)
    sell_labels = label_trades(df, horizon_bars, side="sell", threshold=threshold)

    ev_subset = events.loc[valid_idx]
    is_buy = ev_subset["side"].values == "buy"
    b_lbls = buy_labels.loc[valid_idx].values
    s_lbls = sell_labels.loc[valid_idx].values
    combined_labels = np.where(is_buy, b_lbls, s_lbls)

    mask = ~np.isnan(combined_labels)
    if not np.any(mask):
        return pd.DataFrame(columns=MODEL_FEATURES + ["label"])

    final_idx = valid_idx[mask]
    lbl_values = combined_labels[mask].astype(int)

    present_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    feat_df = df.loc[final_idx, present_cols].copy()

    for c in ("cot_mm_net_z", "cot_mm_net_oi", "cot_producer_net_oi", "kronos_forecast_pct", "kronos_conf"):
        if c not in feat_df.columns:
            feat_df[c] = 0.0

    if cot:
        for k, v in cot.items():
            if k in feat_df.columns:
                feat_df[k] = float(v)

    feat_df["side_buy"] = np.where(is_buy[mask], 1, 0)
    feat_df["side_sell"] = np.where(~is_buy[mask], 1, 0)
    feat_df["label"] = lbl_values

    return feat_df[MODEL_FEATURES + ["label"]].reset_index(drop=True)