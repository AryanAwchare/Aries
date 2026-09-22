"""Position sizing — risk-first, confidence-aware.

For XAUUSD we treat a standard lot as 100 oz; a $1 price move per position
lot = $100 notional movement. Stop distance is expressed in price points.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LOT_OZ = 100.0
USD_PER_OZ_MOVE_PER_LOT = LOT_OZ  # $1 move on 1 lot = $100


@dataclass
class SizingInput:
    equity: float
    risk_fraction: float          # fraction of equity risked on this trade
    stop_distance_points: float   # entry - stop (price points, e.g. 5.00 = $5)
    meta_probability: float = 0.5
    min_lot: float = 0.01
    max_lot: float = 10.0
    lot_step: float = 0.01


def position_size(inp: SizingInput) -> float:
    """Compute the number of lots, uncapped, based on static risk."""
    if inp.stop_distance_points <= 0 or inp.equity <= 0:
        return 0.0
    risk_usd = inp.equity * inp.risk_fraction
    price_risk_per_lot = inp.stop_distance_points * USD_PER_OZ_MOVE_PER_LOT
    lots = risk_usd / price_risk_per_lot
    return float(lots)


def confidence_multiplier(probability: float) -> float:
    """Scale size by meta-model edge: 0.5 P => 0.5x; 0.75 P => ~1.25x."""
    return float(np.clip((probability - 0.5) * 3.0 + 0.5, 0.5, 1.5))


def position_size_with_confidence(inp: SizingInput) -> float:
    raw = position_size(inp)
    scaled = raw * confidence_multiplier(inp.meta_probability)
    return _round_to_step(scaled, inp.min_lot, inp.max_lot, inp.lot_step)


def _round_to_step(lots: float, min_lot: float, max_lot: float, step: float) -> float:
    if lots <= 0:
        return 0.0
    n = round(lots / step) * step
    n = min(max(n, min_lot), max_lot)
    return float(n)