"""Model registry + online predictor for the meta-model.

Also hosts the live calibration monitor (§5 'Model performance monitor'):
it buckets live predictions and realised outcomes so you can watch whether
'70%-confidence' really wins ~70% of the time.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..logging import get_logger

log = get_logger(__name__)


def model_feature_names(model) -> list[str] | None:
    """Best-effort ordered feature list recorded on a fitted estimator.

    Handles the plain sklearn/xgboost estimators and our ``_CalibratedModel``
    wrapper (which exposes the base model as ``.base``).
    """
    for m in (model, getattr(model, "base", None)):
        names = getattr(m, "feature_names_in_", None)
        if names is not None:
            return [str(c) for c in names]
    return None


def read_manifest_features(model_dir, name: str) -> list[str] | None:
    """Ordered training feature list persisted alongside a saved model."""
    manifest = Path(model_dir) / f"{name}.json"
    if not manifest.exists():
        return None
    try:
        data = json.loads(manifest.read_text())
    except (ValueError, OSError):
        return None
    feats = data.get("features")
    return [str(c) for c in feats] if feats else None


def backfill_feature_manifest(model_dir) -> Path | None:
    """Persist the feature list of an already-saved model if the manifest lacks it.

    Pre-feature-manifest models are upgraded in place so live prediction always
    reindexes to the columns the model was actually trained on. Returns the
    manifest path if written, else ``None``.
    """
    import joblib

    model_dir = Path(model_dir)
    best = model_dir / "best_model.txt"
    if not best.exists():
        return None
    name = best.read_text().strip()
    if read_manifest_features(model_dir, name) is not None:
        return None
    model_path = model_dir / f"{name}.joblib"
    if not model_path.exists():
        return None
    feats = model_feature_names(joblib.load(model_path))
    if not feats:
        return None
    manifest = model_dir / f"{name}.json"
    data: dict = {}
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text())
        except (ValueError, OSError):
            data = {}
    data["features"] = feats
    manifest.write_text(json.dumps(data, indent=2))
    log.info("Backfilled feature manifest for '%s' (%d features)", name, len(feats))
    return manifest


class MetaModelPredictor:
    """Loads the best trained model once and scores feature rows.

    Degrades to a neutral ``0.5`` when no trained model exists yet so the
    pipeline is runnable end-to-end on a fresh checkout.
    """

    def __init__(self, model_dir: Path, threshold: float = 0.60) -> None:
        self.model_dir = Path(model_dir)
        self.threshold = threshold
        self._model = None
        self._name = "none"
        self._features: list[str] | None = None
        self._load()

    def _load(self) -> None:
        best = self.model_dir / "best_model.txt"
        if not best.exists():
            log.warning("No trained meta-model found in %s — using neutral P=0.5", self.model_dir)
            return
        self._name = best.read_text().strip()
        path = self.model_dir / f"{self._name}.joblib"
        if not path.exists():
            log.warning("Model file %s missing — using neutral P=0.5", path)
            return
        import joblib

        self._model = joblib.load(path)
        # Prefer the manifest's recorded feature order; fall back to the names
        # the estimator itself carries (legacy models with no manifest field).
        self._features = read_manifest_features(self.model_dir, self._name)
        if self._features is None:
            self._features = model_feature_names(self._model)
        log.info("Loaded meta-model '%s' from %s", self._name, path)

    @property
    def available(self) -> bool:
        return self._model is not None

    def predict(self, feature_row: dict[str, float]) -> float:
        """Calibrated P(trade is profitable) for one feature vector."""
        import pandas as pd

        if self._model is None:
            return 0.5
        cols = self._features or model_feature_names(self._model)
        if cols:
            frame = pd.DataFrame([feature_row]).reindex(columns=cols, fill_value=0.0)
        else:
            frame = pd.DataFrame([feature_row])
        return float(self._model.predict_proba(frame)[0, 1])

    def confirm(self, feature_row: dict[str, float]) -> tuple[bool, float]:
        """Return (confirmed, probability) against the current threshold."""
        p = self.predict(feature_row)
        return (p >= self.threshold), p


class CalibrationTracker:
    """Track predicted-probability vs. realised outcome, in decile buckets."""

    def __init__(self, n_buckets: int = 10) -> None:
        self.n_buckets = n_buckets
        self._buckets: dict[int, dict] = defaultdict(lambda: {"total": 0, "hits": 0})

    def record(self, probability: float, realised_profitable: bool) -> None:
        b = min(int(probability * self.n_buckets), self.n_buckets - 1)
        self._buckets[b]["total"] += 1
        if realised_profitable:
            self._buckets[b]["hits"] += 1

    def report(self) -> list[dict]:
        out = []
        for b in range(self.n_buckets):
            agg = self._buckets[b]
            total = agg["total"]
            empirical = agg["hits"] / total if total else 0.0
            out.append(
                {
                    "bucket_mid": (b + 0.5) / self.n_buckets,
                    "samples": total,
                    "empirical_p": round(empirical, 4),
                }
            )
        return out

    @staticmethod
    def mean_abs_calibration_error(report: list[dict]) -> float:
        populated = [r for r in report if r["samples"] >= 5]
        if not populated:
            return float("nan")
        return float(
            np.mean([abs(r["bucket_mid"] - r["empirical_p"]) for r in populated])
        )


def load_or_default(model_dir: Path, threshold: float) -> MetaModelPredictor:
    return MetaModelPredictor(model_dir, threshold)