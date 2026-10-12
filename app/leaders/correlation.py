"""Stage D: Leader Correlation Engine.

Computes pairwise trade alignment and correlation between eligible leaders.
Detects when multiple leaders are executing redundant / identical strategies
so the signal aggregator (Phase 5) can treat correlated leaders as a single
cluster vote rather than independent consensus.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Sequence

import numpy as np
import pandas as pd

from ..logging import get_logger
from .models import RawLeaderTrade

log = get_logger(__name__)


def build_position_time_series(
    trades: Sequence[RawLeaderTrade],
    start_time: datetime,
    end_time: datetime,
    freq: str = "15min",
) -> pd.Series:
    """Discretize a leader's trade history into a continuous position time series.

    Values: +1 (long), -1 (short), 0 (flat).
    """
    timeline = pd.date_range(start=start_time, end=end_time, freq=freq, tz="UTC")
    series = pd.Series(0.0, index=timeline)

    for t in trades:
        if not t.opened_at:
            continue
        op_t = t.opened_at if t.opened_at.tzinfo else t.opened_at.replace(tzinfo=timezone.utc)
        cl_t = (t.closed_at if t.closed_at and t.closed_at.tzinfo else (t.closed_at.replace(tzinfo=timezone.utc) if t.closed_at else end_time))

        direction = 1.0 if t.side.lower() == "buy" else -1.0
        mask = (series.index >= op_t) & (series.index <= cl_t)
        series.loc[mask] = direction

    return series


class LeaderCorrelationEngine:
    """Computes correlation matrices and clusters highly-correlated leaders."""

    def __init__(self, high_correlation_threshold: float = 0.80) -> None:
        self.threshold = high_correlation_threshold

    def compute_pairwise_correlation(
        self,
        trades_a: Sequence[RawLeaderTrade],
        trades_b: Sequence[RawLeaderTrade],
        window_days: int = 90,
        now: datetime | None = None,
    ) -> float:
        """Compute Pearson correlation between two leaders' positions over a lookback window."""
        end_time = now or datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=window_days)

        ts_a = build_position_time_series(trades_a, start_time, end_time)
        ts_b = build_position_time_series(trades_b, start_time, end_time)

        # If either trader has zero trades (all flat), correlation is 0.0
        if ts_a.std() == 0 or ts_b.std() == 0:
            return 0.0

        corr = float(ts_a.corr(ts_b))
        return 0.0 if np.isnan(corr) else corr

    def compute_matrix(
        self,
        leaders_trades: dict[str, Sequence[RawLeaderTrade]],
        window_days: int = 90,
    ) -> pd.DataFrame:
        """Generate full correlation matrix for all eligible leaders."""
        leader_ids = list(leaders_trades.keys())
        matrix = pd.DataFrame(1.0, index=leader_ids, columns=leader_ids)

        for i, id_a in enumerate(leader_ids):
            for j, id_b in enumerate(leader_ids):
                if i < j:
                    c = self.compute_pairwise_correlation(
                        leaders_trades[id_a], leaders_trades[id_b], window_days=window_days
                    )
                    matrix.loc[id_a, id_b] = c
                    matrix.loc[id_b, id_a] = c

        return matrix

    def cluster_correlated_leaders(
        self,
        matrix: pd.DataFrame,
        leader_ids: list[str] | None = None,
    ) -> list[list[str]]:
        """Group leaders with correlation >= threshold into single clusters.

        Returns a list of clusters, where each cluster is a list of leader IDs.
        """
        active_ids = leader_ids or list(matrix.index)
        clusters: list[list[str]] = []
        visited: set[str] = set()

        for lid in active_ids:
            if lid in visited:
                continue
            # Find all leaders correlated with this one >= threshold
            cluster = [lid]
            visited.add(lid)
            for other_id in active_ids:
                if other_id not in visited and other_id in matrix.columns:
                    if matrix.loc[lid, other_id] >= self.threshold:
                        cluster.append(other_id)
                        visited.add(other_id)
            clusters.append(cluster)

        return clusters