"""Walk-forward validation (Phase 4).

Splits by year: train on T0..T1, validate T1..T2, test T2..T3 — per the plan
(2020–2023 / 2024 / 2025–2026). Each fold trains from scratch so no test
information leaks into training.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

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
class FoldResult:
    fold_name: str
    test_logloss: float
    test_auc: float
    n_train: int
    n_test: int
    n_events_test: int
    winner: str


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
    except Exception as exc:  # baseline is optional
        log.warning("Baseline failed in %s: %s", name, exc)

    results = [calibrate(r, X_val, y_val) for r in results]
    best = pick_best(results)

    if out_dir is not None:
        save_result(best, out_dir / name)

    from sklearn.metrics import log_loss, roc_auc_score

    proba_test = best.model.predict_proba(X_test)[:, 1]
    try:
        ll = float(log_loss(y_test, proba_test, labels=[0, 1]))
        auc = float(roc_auc_score(y_test, proba_test))
    except ValueError:
        ll, auc = float("nan"), float("nan")

    log.info("Fold %s -> winner=%s test LL=%.4f AUC=%.3f", name, best.name, ll, auc)
    return FoldResult(
        fold_name=name,
        test_logloss=ll,
        test_auc=auc,
        n_train=len(X_train),
        n_test=len(X_test),
        n_events_test=int(test_df["label"].size),
        winner=best.name,
    )


def walk_forward(ohlcv, cot=None, *, train_end=..., val_end=..., test_end=...,
                 out_dir: Path | None = None) -> list[FoldResult]:
    """Single-fold default: train/val/test per the build plan's windows.

    Expand to multi-fold (rolling windows, e.g. every year) once live data
    accumulates — structure is identical: call ``run_fold`` per window.
    """
    slices = time_slices(ohlcv, train_end, val_end, test_end)
    fold = run_fold("wf_main", slices["train"], slices["validation"], slices["test"], cot, out_dir=out_dir)
    return [fold]


def write_fold_report(folds: list[FoldResult], path: Path) -> None:
    path.write_text(
        json.dumps([vars(f) for f in folds], indent=2),
        encoding="utf-8",
    )