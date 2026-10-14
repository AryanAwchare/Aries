"""Timeframe profiles — the knobs that must change when the bar size changes.

The feature *windows* (RSI 14, MACD 12/26/9, ATR 14, ...) are deliberately kept
bar-based and identical across timeframes so that a model trained at one bar
size is directly comparable to another. What genuinely differs per timeframe is:

* ``horizon_bars``     — how far ahead the label looks (the "hold").
* ``label_threshold``  — the fractional move that counts as a win; must clear
  the round-trip transaction cost or the label is untradeable.
* ``min_move_pct``     — the candle-body threshold that generates candidate
  events (training proxies for "a leader acted here").
* ``stop_points``      — fallback stop distance in price units.
* ``lookback_hours`` / ``kronos_horizon_bars`` — live pipeline context.

Defaults for M15 reproduce the values the platform shipped with, so existing
behaviour and tests are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeframeProfile:
    timeframe: str
    minutes_per_bar: int
    horizon_bars: int
    label_threshold: float
    min_move_pct: float
    stop_points: float
    lookback_hours: int
    kronos_horizon_bars: int


_PROFILES: dict[str, TimeframeProfile] = {
    # M1 scalping: ~15-minute hold. 0.0003 (~$0.81 @ $2700) clears the ~$0.47
    # round-trip cost (spread+slippage+commission) with a small margin.
    "M1": TimeframeProfile("M1", 1, horizon_bars=15, label_threshold=0.0003,
                           min_move_pct=0.02, stop_points=2.0, lookback_hours=24,
                           kronos_horizon_bars=24),
    "M5": TimeframeProfile("M5", 5, horizon_bars=16, label_threshold=0.0004,
                           min_move_pct=0.03, stop_points=3.0, lookback_hours=48,
                           kronos_horizon_bars=24),
    # The original M15 profile — unchanged from the shipped defaults.
    "M15": TimeframeProfile("M15", 15, horizon_bars=16, label_threshold=0.0005,
                            min_move_pct=0.05, stop_points=5.0, lookback_hours=72,
                            kronos_horizon_bars=24),
    "M30": TimeframeProfile("M30", 30, horizon_bars=16, label_threshold=0.0006,
                            min_move_pct=0.05, stop_points=6.0, lookback_hours=96,
                            kronos_horizon_bars=24),
    "H1": TimeframeProfile("H1", 60, horizon_bars=16, label_threshold=0.0008,
                           min_move_pct=0.06, stop_points=8.0, lookback_hours=120,
                           kronos_horizon_bars=24),
}


def get_profile(timeframe: str) -> TimeframeProfile:
    """Return the profile for ``timeframe`` (falls back to ``M15``)."""
    return _PROFILES.get((timeframe or "").strip().upper(), _PROFILES["M15"])


def available() -> list[str]:
    return sorted(_PROFILES)
