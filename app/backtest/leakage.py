"""Leakage checks (Phase 4 requirement).

Two concrete, automatable checks:

  1. Causality of features — the value of each feature at time *t* must not
     depend on data after *t*. Verified by comparing a feature row built from
     the full frame against one built from the frame truncated at *t*.
  2. Kronos pretraining overlap — warn when the backtest test window falls
     inside the model's pretraining window. Kronos's public weights were
     trained on data up to ~2024; treating 2025+ as out-of-sample is the
     conservative assumption. Update ``KRONOS_TRAINING_CUTOFF`` when you know
     the exact version you deploy.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from ..features.build_features import build_feature_row, FEATURE_COLUMNS
from ..logging import get_logger

log = get_logger(__name__)

# Conservative estimate; update to the exact checkpoint you ship.
KRONOS_TRAINING_CUTOFF = date(2024, 6, 1)


@dataclass
class LeakageReport:
    features_causal: bool
    mismatched_features: list[str]
    kronos_test_overlap: bool
    notes: list[str] = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []

    @property
    def clean(self) -> bool:
        return self.features_causal and not self.kronos_test_overlap


def check_feature_causality(ohlcv: pd.DataFrame, at_index: int | None = None) -> tuple[bool, list[str]]:
    """Build the feature row at ``idx`` two ways (full vs truncated) — must match."""
    if "rsi" not in ohlcv.columns:
        from ..features.build_features import calculate_indicator_block

        ohlcv = calculate_indicator_block(ohlcv)

    idx = ohlcv.index[at_index] if at_index is not None else ohlcv.index[-3]
    full = build_feature_row(ohlcv, index=idx)
    truncated = build_feature_row(ohlcv.iloc[: list(ohlcv.index).index(idx) + 1], index=None)
    mismatched = [c for c in FEATURE_COLUMNS if abs(full.get(c, 0) - truncated.get(c, 0)) > 1e-9]
    return (not mismatched), mismatched


def check_kronos_overlap(test_start: str) -> bool:
    """True if the test window overlaps Kronos's pretraining window (bad)."""
    start = datetime.fromisoformat(test_start).date()
    return start < KRONOS_TRAINING_CUTOFF


def run_leakage_report(
    ohlcv: pd.DataFrame, test_start: str, at_index: int | None = None
) -> LeakageReport:
    causal, mismatched = check_feature_causality(ohlcv, at_index)
    overlap = check_kronos_overlap(test_start)

    notes = []
    if not causal:
        notes.append(f"Non-causal features detected: {mismatched}")
    if overlap:
        notes.append(
            f"Test window starts before Kronos training cutoff "
            f"({KRONOS_TRAINING_CUTOFF}) — results may benefit from pretraining leakage."
        )
    report = LeakageReport(causal, mismatched, overlap, notes)
    log.info("Leakage report: causal=%s kronos_overlap=%s", causal, overlap)
    return report