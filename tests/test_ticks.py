"""Phase 1.2 tick ingestion + microstructure tests (offline, no MT5 needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.data.ticks import (
    TICK_COLUMNS,
    load_ticks_csv,
    save_ticks,
    tick_coverage,
    tick_microstructure_features,
    ticks_to_bars,
)


def _synthetic_ticks(n: int = 6000, start: str = "2026-01-02 08:00:00") -> pd.DataFrame:
    rng = np.random.default_rng(7)
    ts = pd.date_range(start, periods=n, freq="500ms", tz="UTC")
    mid = 2000 + np.cumsum(rng.normal(0, 0.05, n))
    spread = 0.30 + rng.uniform(0, 0.10, n)
    return pd.DataFrame(
        {
            "time": ts,
            "bid": mid - spread / 2,
            "ask": mid + spread / 2,
            "last": 0.0,
            "volume": 0,
            "time_msc": (ts.view("int64") // 1_000_000),
            "flags": 1026,
            "volume_real": 0.0,
        }
    )


def test_ticks_to_bars_produces_ohlcv_and_microstructure():
    ticks = _synthetic_ticks()
    bars = ticks_to_bars(ticks, "M1")
    assert not bars.empty
    for col in ["open", "high", "low", "close", "volume", "tick_count",
                "buy_ticks", "sell_ticks", "ofi", "spread_mean", "spread_pct"]:
        assert col in bars.columns, col
    # OHLC sanity
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["high"] >= bars["close"]).all() and (bars["low"] <= bars["close"]).all()
    # OFI bounded in [-1, 1]; tick counts positive
    assert bars["ofi"].between(-1.0, 1.0).all()
    assert (bars["tick_count"] > 0).all()
    # spread derived from bid/ask, positive
    assert (bars["spread_mean"] > 0).all()
    assert (bars["spread_pct"] > 0).all()


def test_ticks_to_bars_empty():
    assert ticks_to_bars(pd.DataFrame(columns=TICK_COLUMNS)).empty


def test_microstructure_features_added_and_finite():
    bars = ticks_to_bars(_synthetic_ticks(), "M1")
    feat = tick_microstructure_features(bars, window=20)
    for col in ["vpin", "roll_spread", "kyle_lambda", "intensity_z", "spread_z"]:
        assert col in feat.columns, col
        assert np.isfinite(feat[col]).all(), col
    assert (feat["vpin"] >= 0).all()
    assert (feat["roll_spread"] >= 0).all()


def test_save_load_roundtrip(tmp_path):
    ticks = _synthetic_ticks(500)
    p = save_ticks(ticks, tmp_path / "t.csv")
    back = load_ticks_csv(p)
    assert len(back) == 500
    assert pd.api.types.is_datetime64_any_dtype(back["time"])
    cov = tick_coverage(back, "XAUUSD")
    assert cov.ticks == 500
    assert cov.start is not None and cov.end is not None
    assert cov.start <= cov.end


def test_tick_coverage_empty():
    cov = tick_coverage(pd.DataFrame(columns=TICK_COLUMNS), "XAUUSD")
    assert cov.ticks == 0 and cov.start is None


def test_ticks_to_bars_handles_missing_quotes():
    # last-only feed: no bid/ask -> still aggregates OHLCV, zero spread
    ticks = _synthetic_ticks(1000).drop(columns=["bid", "ask"])
    bars = ticks_to_bars(ticks, "M1")
    assert not bars.empty
    assert (bars["spread_mean"] == 0.0).all()
