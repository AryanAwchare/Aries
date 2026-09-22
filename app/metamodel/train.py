"""Training pipeline for the meta-model.

XGBoost is benchmarked against a logistic-regression baseline. If XGBoost
does not beat the baseline on validation log-loss / AUC, that IS the finding
— the platform then keeps the simpler model. Both are calibrated on the
validation window so probabilities are honest.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..logging import get_logger

log = get_logger(__name__)

try:
    import xgboost as xgb
except Exception:  # pragma: no cover - optional dep
    xgb = None


@dataclass
class TrainResult:
    name: str
    model: object | None = None
    metrics: dict = field(default_factory=dict)
    calibration: str = "none"

    @property
    def logloss(self) -> float:
        return float(self.metrics.get("logloss", float("inf")))

    @property
    def auc(self) -> float:
        return float(self.metrics.get("auc", 0.0))


def _metrics(y_true, y_proba) -> dict:
    from sklearn.metrics import auc, brier_score_loss, log_loss, roc_curve

    if len(np.unique(y_true)) < 2:
        return {"logloss": float("nan"), "auc": float("nan"), "brier": float("nan")}
    return {
        "logloss": float(log_loss(y_true, y_proba, labels=[0, 1])),
        "auc": float(auc(*roc_curve(y_true, y_proba)[:2])),
        "brier": float(brier_score_loss(y_true, y_proba)),
    }


def train_xgboost(X_train, y_train, X_val, y_val, **params) -> TrainResult:
    if xgb is None:
        raise RuntimeError("xgboost is not installed; run: pip install -r requirements.txt")
    kwargs = dict(params or {})
    kwargs.setdefault("n_estimators", 500)
    kwargs.setdefault("max_depth", 4)
    kwargs.setdefault("learning_rate", 0.05)
    kwargs.setdefault("subsample", 0.9)
    kwargs.setdefault("colsample_bytree", 0.8)
    kwargs.setdefault("eval_metric", "logloss")
    kwargs.setdefault("early_stopping_rounds", 25)

    model = xgb.XGBClassifier(
        **{k: v for k, v in kwargs.items() if k not in ("early_stopping_rounds", "eval_metric")},
        early_stopping_rounds=kwargs["early_stopping_rounds"],
        eval_metric=kwargs["eval_metric"],
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    proba_val = model.predict_proba(X_val)[:, 1]
    result = TrainResult(name="xgboost", model=model, metrics=_metrics(y_val, proba_val))
    log.info("XGBoost val: %s", result.metrics)
    return result


def train_baseline(X_train, y_train, X_val, y_val) -> TrainResult:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    proba_val = model.predict_proba(X_val)[:, 1]
    result = TrainResult(name="logistic", model=model, metrics=_metrics(y_val, proba_val))
    log.info("LR baseline val: %s", result.metrics)
    return result


def _fit_platt(proba: np.ndarray, y_true) -> object:
    from sklearn.calibration import _SigmoidCalibration

    cal = _SigmoidCalibration()
    cal.fit(proba, y_true)
    return cal


def _fit_isotonic(proba: np.ndarray, y_true) -> object:
    from sklearn.isotonic import IsotonicRegression

    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(proba, y_true)


class _CalibratedModel:
    """Wraps base model + a fitted 1-D calibrator over its scores."""

    def __init__(self, base, calibrator, method: str) -> None:
        self.base = base
        self.calibrator = calibrator
        self.method = method

    def predict_proba(self, X) -> np.ndarray:
        score = self.base.predict_proba(X)[:, 1]
        calibrated = self.calibrator.predict(score)
        p = np.clip(np.asarray(calibrated, dtype=float).squeeze(), 0.0, 1.0)
        return np.column_stack([1 - p, p])


def calibrate(result: TrainResult, X_val, y_val, method: str = "isotonic") -> TrainResult:
    """Platt or isotonic calibration on validation scores.

    Handles both sklearn APIs: the old ``cv="prefit"`` route and the newer
    direct-calibrator route (sklearn >= 1.6 removed ``cv="prefit"``).
    """
    if result.model is None:
        return result
    fitted = False
    if _sklearn_supports_prefit():
        try:
            from sklearn.calibration import CalibratedClassifierCV

            calibrated = CalibratedClassifierCV(result.model, method=method, cv="prefit").fit(X_val, y_val)
            cal_model = _wrap_if_needed(calibrated)
            fitted = True
        except (ValueError, TypeError, ImportError):
            fitted = False
    if not fitted:
        score = result.model.predict_proba(X_val)[:, 1]
        calib = _fit_isotonic(score, y_val) if method == "isotonic" else _fit_platt(score, y_val)
        cal_model = _CalibratedModel(result.model, calib, method)

    proba_val = cal_model.predict_proba(X_val)[:, 1]
    result.model = cal_model
    result.calibration = method
    result.metrics = _metrics(y_val, proba_val)
    log.info("Calibrated %s (%s): %s", result.name, method, result.metrics)
    return result


def _sklearn_supports_prefit() -> bool:
    import sklearn

    return sklearn.__version__ < "1.6"


def _wrap_if_needed(model) -> object:
    # CalibratedClassifierCV already returns (n,2); wrapper stays a no-op passthrough.
    class _C:
        def __init__(self, m):
            self._m = m

        def predict_proba(self, X):
            return self._m.predict_proba(X)

    return _C(model)


def pick_best(results: list[TrainResult]) -> TrainResult:
    """Choose the better model by validation log-loss (tie-break: AUC)."""
    valid = [r for r in results if np.isfinite(r.logloss)]
    if not valid:
        return results[0]
    return min(valid, key=lambda r: (r.logloss, -r.auc))


def save_result(result: TrainResult, directory: Path) -> Path:
    import joblib

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / f"{result.name}.joblib"
    joblib.dump(result.model, model_path)
    (directory / f"{result.name}.json").write_text(
        json.dumps({"name": result.name, "calibration": result.calibration, "metrics": result.metrics}, indent=2)
    )
    (directory / "best_model.txt").write_text(result.name)
    log.info("Saved %s -> %s", result.name, model_path)
    return model_path