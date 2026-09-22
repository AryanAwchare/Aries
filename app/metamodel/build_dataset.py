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
    horizon_bars: int = 8,
    min_move_pct: float = 0.05,
) -> pd.DataFrame:
    """Feature rows + ``label`` for every candidate event.

    Returns columns: MODEL_FEATURES + ``label`` (1 if the side was profitable
    over ``horizon_bars``).
    """
    df = calculate_indicator_block(ohlcv)
    events = candidate_events(df, min_move_pct)
    if events.empty:
        return pd.DataFrame(columns=MODEL_FEATURES + ["label"])

    rows = []
    labels = label_trades(df, horizon_bars)
    for ts, ev in events.iterrows():
        idx = ts
        side = str(ev["side"])
        lbl = labels.loc[idx]
        if pd.isna(lbl):
            continue  # no full future window — cannot label honestly
        lbl = int(lbl)
        feature = {c: float(df.loc[idx, c]) for c in FEATURE_COLUMNS if c in df.columns}
        # Context columns that are not part of the OHLCV frame default to zero.
        for c in ("cot_mm_net_z", "cot_mm_net_oi", "cot_producer_net_oi", "kronos_forecast_pct", "kronos_conf"):
            feature.setdefault(c, 0.0)
        if cot:
            feature.update({k: float(v) for k, v in cot.items() if k in feature})
        feature["side_buy"] = 1 if side == "buy" else 0
        feature["side_sell"] = 1 if side == "sell" else 0
        feature["label"] = lbl
        rows.append(feature)
    return pd.DataFrame(rows, columns=MODEL_FEATURES + ["label"])