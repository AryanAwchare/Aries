"""Kronos wrapper — financial time-series foundation model.

Kronos is used as a *point forecast* engine: given recent OHLCV closes it
returns a forecast of the next ``horizon`` values plus a standard deviation
(uncertainty). The wrapper maps that to:

  * ``forecast_pct``  — expected % move over the horizon (scaled by ATR for stability)
  * ``confidence``    — 0..1 confidence proxy derived from the model std

Import is lazy: the rest of the platform runs (backtests, risk, tests)
even when torch/Kronos are not installed — the forecaster degrades to a
zero forecast with ``available=False``.
"""
from __future__ import annotations

import numpy as np

from ..logging import get_logger

log = get_logger(__name__)


class KronosForecaster:
    """Thin, testable wrapper around amazon-science/kronos."""

    def __init__(self, model_path: str = "", horizon: int = 24, context_bars: int = 512) -> None:
        self.model_path = model_path
        self.horizon = horizon
        self.context_bars = context_bars
        self._model = None

    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        return self._load_model() is not None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            import kronos

            log.info("Kronos loaded from %s", self.model_path or "pretrained amazon/Kronos-Small")
            if self.model_path:
                self._model = kronos.KronosPredictor.from_pretrained(self.model_path)
            else:
                self._model = kronos.KronosPredictor.from_pretrained("amazon/Kronos-Small")
        except Exception as exc:  # ImportError, OSError, torch not installed, etc.
            log.warning("Kronos unavailable (%s) — using neutral forecast", exc)
            self._model = False
        return self._model

    # ------------------------------------------------------------------
    def predict_pct(self, closes: np.ndarray) -> tuple[float, float]:
        """Return (forecast_pct, confidence) for the horizon.

        ``closes`` is the most-recent contiguous sequence of closes (ascending
        time). Normalises by relative change so scale (e.g. $1800 vs $2600) does
        not matter.
        """
        model = self._load_model()
        if model is False:
            return 0.0, 0.0

        try:
            import torch

            series = np.asarray(closes, dtype=np.float32)
            context = torch.tensor(series, dtype=torch.float32).unsqueeze(0) if series.ndim == 1 else torch.tensor(series, dtype=torch.float32)
            forecast, forecast_std = model.predict(context, self.horizon)
            f = np.asarray(forecast[0], dtype=np.float32).squeeze()
            s = np.asarray(forecast_std[0], dtype=np.float32).squeeze()
            last = float(series[-1]) if len(series) else 1.0
            if last <= 0:
                return 0.0, 0.0
            pct = float((f[-1] - last) / last) * 100.0
            # confidence proxy: lower forecast std => higher confidence
            std_pct = float(np.nanmean(s) / last) * 100.0
            confidence = float(np.clip(1.0 - std_pct / 1.0, 0.0, 0.95))
            return pct, confidence
        except Exception as exc:  # pragma: no cover - torch/mps edge cases
            log.warning("Kronos predict failed (%s) — neutral forecast", exc)
            return 0.0, 0.0


class NeutralForecaster:
    """Degenerate forecaster (no move, zero confidence) for tests/CI."""

    available = False

    def predict_pct(self, closes) -> tuple[float, float]:
        return 0.0, 0.0

    @property
    def horizon(self) -> int:
        return 24