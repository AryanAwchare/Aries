"""Meta-model training pipeline tests (skipped if xgboost is unavailable,
e.g. Python 3.14 without wheels)."""
from __future__ import annotations

import numpy as np
import pytest

xgb = pytest.importorskip("xgboost", reason="xgboost not installed")

from app.metamodel.build_dataset import build_training_data  # noqa: E402
from app.metamodel.predict import CalibrationTracker  # noqa: E402
from app.metamodel.train import (  # noqa: E402
    calibrate,
    pick_best,
    save_result,
    train_baseline,
    train_xgboost,
)


@pytest.fixture
def framed(synthetic_ohlcv):
    df = build_training_data(synthetic_ohlcv, horizon_bars=8)
    assert len(df) > 20
    cols = [c for c in df.columns if c != "label"]
    X = df[cols]
    y = df["label"]
    split = int(len(X) * 0.7)
    return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]


def test_train_xgboost(framed):
    X_tr, y_tr, X_val, y_val = framed
    res = train_xgboost(X_tr, y_tr, X_val, y_val, n_estimators=40, early_stopping_rounds=5)
    assert res.name == "xgboost"
    assert "auc" in res.metrics


def test_train_baseline(framed):
    X_tr, y_tr, X_val, y_val = framed
    res = train_baseline(X_tr, y_tr, X_val, y_val)
    assert res.name == "logistic"


def test_calibration_and_pick(framed, tmp_path):
    X_tr, y_tr, X_val, y_val = framed
    results = [
        train_xgboost(X_tr, y_tr, X_val, y_val, n_estimators=40, early_stopping_rounds=5),
        train_baseline(X_tr, y_tr, X_val, y_val),
    ]
    results = [calibrate(r, X_val, y_val) for r in results]
    best = pick_best(results)
    assert best.name in ("xgboost", "logistic")
    path = save_result(best, tmp_path)
    assert path.exists()
    assert (tmp_path / "best_model.txt").exists()


def test_calibration_tracker():
    tracker = CalibrationTracker(n_buckets=10)
    for _ in range(20):
        tracker.record(0.7, True)
    for _ in range(10):
        tracker.record(0.7, False)
    report = tracker.report()
    bucket = [r for r in report if abs(r["bucket_mid"] - 0.75) < 0.06][0]
    assert bucket["samples"] == 30
    assert bucket["empirical_p"] == pytest.approx(2 / 3, rel=1e-3)
    mace = tracker.mean_abs_calibration_error(report)
    assert mace == pytest.approx(0.75 - 2 / 3, rel=1e-3)