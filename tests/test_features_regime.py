"""Regime detection and OHLCV microstructure proxies wired into feature rows."""
from __future__ import annotations

import numpy as np

from app.features.build_features import (
    FEATURE_COLUMNS,
    build_feature_row,
    calculate_indicator_block,
)
from app.features.regime import MarketRegime, detect_regimes

NEW_FEATURES = [
    "parkinson_vol",
    "garman_klass_vol",
    "efficiency_ratio",
    "signed_body",
    "wick_imbalance",
    "regime_code",
]


def test_detect_regimes_returns_valid_labels(synthetic_ohlcv):
    regimes = detect_regimes(synthetic_ohlcv)
    assert len(regimes) == len(synthetic_ohlcv)
    valid = {r.value for r in MarketRegime}
    assert set(regimes.unique()).issubset(valid)


def test_detect_regimes_short_frame_is_neutral(synthetic_ohlcv):
    regimes = detect_regimes(synthetic_ohlcv.head(10))
    assert (regimes == MarketRegime.NEUTRAL.value).all()


def test_new_features_are_registered_and_finite(synthetic_ohlcv):
    df = calculate_indicator_block(synthetic_ohlcv)
    for col in NEW_FEATURES:
        assert col in FEATURE_COLUMNS, col
        assert col in df.columns, col
        assert np.isfinite(df[col].to_numpy(dtype=float)).all(), col


def test_feature_row_exposes_new_features(synthetic_ohlcv):
    row = build_feature_row(synthetic_ohlcv)
    for col in NEW_FEATURES:
        assert col in row
        assert np.isfinite(row[col])
