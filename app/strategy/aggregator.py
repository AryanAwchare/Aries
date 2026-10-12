"""Phase 5: Multi-Factor Signal Aggregator.

Replaces the simple binary "leader AND meta-model" gate with a multi-factor
aggregator combining:
1. Market regime (trending/ranging from features & indicators)
2. Live-tier leader signals, correlation-adjusted (correlated leaders count as 1 vote)
3. Kronos foundation model forecast direction & confidence
4. Calibrated XGBoost meta-model P(profitable)

Outputs a single definitive final_signal (BUY / SELL / NO_TRADE) and logs all
contributing factors to the signals table for complete auditability.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..logging import get_logger

log = get_logger(__name__)


@dataclass
class LeaderVote:
    leader_id: str
    direction: str  # "buy" | "sell"
    score: float = 1.0


@dataclass
class AggregatedSignalResult:
    final_signal: str                     # "BUY" | "SELL" | "NO_TRADE"
    symbol: str
    direction: str                        # "buy" | "sell" | "none"
    confidence: float = 0.0
    xgb_probability: float = 0.0
    kronos_direction: str = "neutral"     # "buy" | "sell" | "neutral"
    kronos_confidence: float = 0.0
    regime: str = "ranging"
    contributing_factors: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


class SignalAggregator:
    """Consolidates market regime, correlation-adjusted leaders, Kronos, and XGBoost."""

    def __init__(
        self,
        prob_threshold: float = 0.60,
        high_correlation_threshold: float = 0.80,
        weight_leaders: float = 0.35,
        weight_kronos: float = 0.25,
        weight_xgboost: float = 0.30,
        weight_regime: float = 0.10,
    ) -> None:
        self.threshold = prob_threshold
        self.corr_threshold = high_correlation_threshold
        self.w_leaders = weight_leaders
        self.w_kronos = weight_kronos
        self.w_xgb = weight_xgboost
        self.w_regime = weight_regime

    def aggregate(
        self,
        symbol: str,
        feature_row: dict[str, float],
        leader_votes: list[LeaderVote],
        correlation_matrix: pd.DataFrame | None = None,
        kronos_forecast_pct: float = 0.0,
        kronos_conf: float = 0.0,
        xgb_probability: float = 0.50,
    ) -> AggregatedSignalResult:
        # 1. Market Regime determination
        trend_str = feature_row.get("trend_strength", 0.0)
        vol_reg = feature_row.get("vol_regime", "ranging")
        if isinstance(vol_reg, (int, float)):
            regime = "trending_bull" if trend_str > 0.05 else ("trending_bear" if trend_str < -0.05 else "ranging")
        else:
            regime = str(vol_reg)

        regime_dir_val = 1.0 if trend_str > 0.05 else (-1.0 if trend_str < -0.05 else 0.0)

        # 2. Correlation-Adjusted Leader Voting
        clusters = self._cluster_leaders(leader_votes, correlation_matrix)
        cluster_votes: list[float] = []
        cluster_details: list[dict[str, Any]] = []

        for cluster in clusters:
            if not cluster:
                continue
            buys = sum(1 for v in cluster if v.direction.lower() == "buy")
            sells = sum(1 for v in cluster if v.direction.lower() == "sell")
            # Cluster consolidated vote is normalized to +1 or -1
            if buys > sells:
                c_vote = 1.0
            elif sells > buys:
                c_vote = -1.0
            else:
                c_vote = 0.0
            cluster_votes.append(c_vote)
            cluster_details.append({
                "leaders": [v.leader_id for v in cluster],
                "buys": buys,
                "sells": sells,
                "consolidated_vote": c_vote,
            })

        net_leader_score = float(np.mean(cluster_votes)) if cluster_votes else 0.0
        leader_dir_str = "buy" if net_leader_score > 0.1 else ("sell" if net_leader_score < -0.1 else "neutral")

        # 3. Kronos Direction & Confidence
        if kronos_forecast_pct > 0.02:
            kronos_dir = "buy"
            kronos_dir_val = 1.0 * min(1.0, kronos_conf + 0.3)
        elif kronos_forecast_pct < -0.02:
            kronos_dir = "sell"
            kronos_dir_val = -1.0 * min(1.0, kronos_conf + 0.3)
        else:
            kronos_dir = "neutral"
            kronos_dir_val = 0.0

        # 4. XGBoost Directional Impact
        # Probability P above 0.5 favors current leader/trend direction
        xgb_dir_val = (xgb_probability - 0.5) * 2.0  # Normalized to [-1.0, 1.0]

        # 5. Composite Score Synthesis
        composite_score = float(
            self.w_leaders * net_leader_score
            + self.w_kronos * kronos_dir_val
            + self.w_xgb * xgb_dir_val
            + self.w_regime * regime_dir_val
        )

        factors = {
            "regime": regime,
            "trend_strength": trend_str,
            "net_leader_score": net_leader_score,
            "leader_clusters_count": len(clusters),
            "leader_cluster_details": cluster_details,
            "kronos_forecast_pct": kronos_forecast_pct,
            "kronos_confidence": kronos_conf,
            "kronos_direction": kronos_dir,
            "xgb_probability": xgb_probability,
            "composite_score": composite_score,
            "weights": {
                "leaders": self.w_leaders,
                "kronos": self.w_kronos,
                "xgboost": self.w_xgb,
                "regime": self.w_regime,
            },
        }

        # Gate evaluations
        if xgb_probability < self.threshold:
            return AggregatedSignalResult(
                final_signal="NO_TRADE",
                symbol=symbol,
                direction="none",
                confidence=abs(composite_score),
                xgb_probability=xgb_probability,
                kronos_direction=kronos_dir,
                kronos_confidence=kronos_conf,
                regime=regime,
                contributing_factors=factors,
                reason=f"xgb_probability_below_threshold:{xgb_probability:.3f}<{self.threshold}",
            )

        if composite_score >= 0.25:
            final_signal = "BUY"
            direction = "buy"
            reason = "consensus_buy"
        elif composite_score <= -0.25:
            final_signal = "SELL"
            direction = "sell"
            reason = "consensus_sell"
        else:
            final_signal = "NO_TRADE"
            direction = "none"
            reason = f"insufficient_consensus_score:{composite_score:.3f}"

        return AggregatedSignalResult(
            final_signal=final_signal,
            symbol=symbol,
            direction=direction,
            confidence=abs(composite_score),
            xgb_probability=xgb_probability,
            kronos_direction=kronos_dir,
            kronos_confidence=kronos_conf,
            regime=regime,
            contributing_factors=factors,
            reason=reason,
        )

    def _cluster_leaders(
        self,
        votes: list[LeaderVote],
        corr_matrix: pd.DataFrame | None,
    ) -> list[list[LeaderVote]]:
        """Group correlated leaders into single clusters so they count as 1 vote."""
        if not votes:
            return []
        if corr_matrix is None or corr_matrix.empty:
            # If no matrix provided, treat each leader as an independent vote
            return [[v] for v in votes]

        votes_by_id = {v.leader_id: v for v in votes}
        visited: set[str] = set()
        clusters: list[list[LeaderVote]] = []

        for v in votes:
            lid = v.leader_id
            if lid in visited:
                continue
            cluster = [v]
            visited.add(lid)
            if lid in corr_matrix.index:
                for other_id, other_vote in votes_by_id.items():
                    if other_id not in visited and other_id in corr_matrix.columns:
                        if corr_matrix.loc[lid, other_id] >= self.corr_threshold:
                            cluster.append(other_vote)
                            visited.add(other_id)
            clusters.append(cluster)

        return clusters