"""Walk-forward validation and probabilistic calibration (Phase 4).

Splits by year:
- Train: 2020-2023
- Validate: 2024
- Test: 2025-2026

Calculates comprehensive calibration quality metrics:
- LogLoss
- ROC-AUC
- Brier Score (mean squared error of probability predictions)
- Reliability Curve (calibration bins: predicted probability vs true outcome fraction)
- Expected Calibration Error (ECE)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from ..logging import get_logger
from ..metamodel.build_dataset import build_training_data
from ..metamodel.train import (
    calibrate,
    pick_best,
    save_result,
    train_baseline,
    train_xgboost,
)

log = get_logger(__name__)


@dataclass
class ReliabilityBin:
    bin_index: int
    mean_predicted_prob: float
    true_fraction: float
    count: int


@dataclass
class CalibrationReport:
    brier_score: float
    expected_calibration_error: float
    reliability_curve: list[ReliabilityBin] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "brier_score": self.brier_score,
            "expected_calibration_error": self.expected_calibration_error,
            "reliability_curve": [
                {
                    "bin": b.bin_index,
                    "pred_prob": round(b.mean_predicted_prob, 4),
                    "true_win_rate": round(b.true_fraction, 4),
                    "count": b.count,
                }
                for b in self.reliability_curve
            ],
        }


def compute_calibration_report(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> CalibrationReport:
    """Compute Brier Score, ECE, and reliability curve bins."""
    if len(y_true) == 0 or len(y_prob) == 0:
        return CalibrationReport(brier_score=0.0, expected_calibration_error=0.0)

    # 1. Brier score
    try:
        brier = float(brier_score_loss(y_true, y_prob))
    except Exception:
        brier = float(np.mean((y_prob - y_true) ** 2))

    # 2. Reliability curve
    bins: list[ReliabilityBin] = []
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    total_samples = len(y_true)
    ece = 0.0

    for idx in range(n_bins):
        b_min = bin_edges[idx]
        b_max = bin_edges[idx + 1]
        mask = (y_prob >= b_min) & (y_prob <= b_max if idx == n_bins - 1 else y_prob < b_max)
        count = int(np.sum(mask))
        if count > 0:
            mean_pred = float(np.mean(y_prob[mask]))
            true_frac = float(np.mean(y_true[mask]))
            bins.append(
                ReliabilityBin(
                    bin_index=idx + 1,
                    mean_predicted_prob=mean_pred,
                    true_fraction=true_frac,
                    count=count,
                )
            )
            ece += (count / total_samples) * abs(mean_pred - true_frac)

    return CalibrationReport(
        brier_score=brier,
        expected_calibration_error=float(ece),
        reliability_curve=bins,
    )


@dataclass
class FoldResult:
    fold_name: str
    test_logloss: float
    test_auc: float
    brier_score: float
    ece: float
    n_train: int
    n_test: int
    n_events_test: int
    winner: str
    calibration: CalibrationReport = field(default_factory=lambda: CalibrationReport(0.0, 0.0))


def time_slices(df: pd.DataFrame, train_end: str, val_end: str, test_end: str) -> dict[str, pd.DataFrame]:
    def to_dt(s: str) -> pd.Timestamp:
        return pd.Timestamp(datetime.fromisoformat(s), tz="UTC") if "T" not in s else pd.Timestamp(s)

    out = {}
    for label, (start, end) in {
        "train": (df.index.min(), to_dt(train_end)),
        "validation": (to_dt(train_end) + pd.Timedelta(hours=1), to_dt(val_end)),
        "test": (to_dt(val_end) + pd.Timedelta(hours=1), to_dt(test_end)),
    }.items():
        mask = (df.index >= start) & (df.index <= end)
        out[label] = df.loc[mask]
    return out


def run_fold(name, ohlcv_train, ohlcv_val, ohlcv_test, cot=None,
             horizon_bars: int = 8, out_dir: Path | None = None) -> FoldResult:
    train_df = build_training_data(ohlcv_train, cot, horizon_bars)
    val_df = build_training_data(ohlcv_val, cot, horizon_bars)
    test_df = build_training_data(ohlcv_test, cot, horizon_bars)

    X_train = train_df[[c for c in train_df.columns if c != "label"]]
    y_train = train_df["label"]
    X_val = val_df[[c for c in val_df.columns if c != "label"]]
    y_val = val_df["label"]
    X_test = test_df[[c for c in test_df.columns if c != "label"]]
    y_test = test_df["label"]

    results = [train_xgboost(X_train, y_train, X_val, y_val)]
    try:
        results.append(train_baseline(X_train, y_train, X_val, y_val))
    except Exception as exc:
        log.warning("Baseline failed in %s: %s", name, exc)

    results = [calibrate(r, X_val, y_val) for r in results]
    best = pick_best(results)

    if out_dir is not None:
        save_result(best, out_dir / name)

    proba_test = best.model.predict_proba(X_test)[:, 1]
    y_test_arr = y_test.to_numpy()

    try:
        ll = float(log_loss(y_test, proba_test, labels=[0, 1]))
        auc = float(roc_auc_score(y_test, proba_test))
    except ValueError:
        ll, auc = float("nan"), float("nan")

    cal_rep = compute_calibration_report(y_test_arr, proba_test, n_bins=10)

    log.info("Fold %s -> winner=%s test LL=%.4f AUC=%.3f Brier=%.4f ECE=%.4f",
             name, best.name, ll, auc, cal_rep.brier_score, cal_rep.expected_calibration_error)

    return FoldResult(
        fold_name=name,
        test_logloss=ll,
        test_auc=auc,
        brier_score=cal_rep.brier_score,
        ece=cal_rep.expected_calibration_error,
        n_train=len(X_train),
        n_test=len(X_test),
        n_events_test=int(test_df["label"].size),
        winner=best.name,
        calibration=cal_rep,
    )


def walk_forward(ohlcv, cot=None, *, train_end=..., val_end=..., test_end=...,
                 out_dir: Path | None = None) -> list[FoldResult]:
    slices = time_slices(ohlcv, train_end, val_end, test_end)
    fold = run_fold("wf_main", slices["train"], slices["validation"], slices["test"], cot, out_dir=out_dir)
    return [fold]


def write_fold_report(folds: list[FoldResult], path: Path) -> None:
    data = []
    for f in folds:
        d = vars(f).copy()
        d["calibration"] = f.calibration.to_dict()
        data.append(d)
    path.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )