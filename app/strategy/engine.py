"""Strategy decision engine — multi-factor signal aggregator."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from ..config import Settings
from ..db import audit
from ..db.models import Leader, LeaderTrade, Signal, SignalStatus, TradeSide
from ..features.build_features import FEATURE_COLUMNS
from ..leaders.models import LeaderScore
from ..logging import get_logger
from ..metamodel.predict import MetaModelPredictor
from .aggregator import AggregatedSignalResult, LeaderVote, SignalAggregator

log = get_logger(__name__)


@dataclass
class DecisionOutcome:
    verb: str  # "alert" | "copy" | "skip"
    status: SignalStatus
    meta_probability: float = 0.0
    leader_score: LeaderScore | None = None
    kronos_forecast_pct: float = 0.0
    final_signal: str = "NO_TRADE"  # "BUY" | "SELL" | "NO_TRADE"
    reason: str = ""
    aggregated_result: AggregatedSignalResult | None = None


class StrategyEngine:
    """Evaluates multi-factor signals combining Regime, Correlation-adjusted Leaders, Kronos, and XGBoost."""

    def __init__(
        self,
        settings: Settings,
        predictor: MetaModelPredictor,
        require_forecast_alignment: bool = True,
        aggregator: SignalAggregator | None = None,
        min_leader_score: float | None = None,
        meta_threshold: float | None = None,
    ) -> None:
        self.settings = settings
        self.predictor = predictor
        self.require_forecast_alignment = require_forecast_alignment
        self.meta_threshold = meta_threshold if meta_threshold is not None else settings.meta_model_prob_threshold
        self.min_leader_score = min_leader_score if min_leader_score is not None else settings.min_leader_score
        self.aggregator = aggregator or SignalAggregator(
            prob_threshold=self.meta_threshold,
            high_correlation_threshold=0.80,
        )

    def evaluate(
        self,
        event: LeaderTrade,
        leader_score: LeaderScore,
        feature_row: dict[str, float],
        ohlcv: object | None = None,
        correlation_matrix: object | None = None,
    ) -> DecisionOutcome:
        """Run the multi-factor signal aggregator."""
        if event.is_open is False and event.close_price is None:
            return DecisionOutcome("skip", SignalStatus.SKIPPED, reason="no_price")

        # 1. Calibrated XGBoost probability (support fake_p in tests)
        if "fake_p" in feature_row:
            p = float(feature_row["fake_p"])
        else:
            p = self.predictor.predict(feature_row)

        kronos_pct = feature_row.get("kronos_forecast_pct", 0.0)
        kronos_conf = feature_row.get("kronos_conf", 0.5)

        # Legacy Gate / Forecast Conflict check
        side_val = event.side.value if hasattr(event.side, "value") else str(event.side).lower()
        if self.require_forecast_alignment and abs(kronos_pct) > 0.1:
            conflicts = (side_val == "buy" and kronos_pct < 0) or (
                side_val == "sell" and kronos_pct > 0
            )
            if conflicts:
                return DecisionOutcome(
                    "skip",
                    SignalStatus.REJECTED,
                    meta_probability=p,
                    leader_score=leader_score,
                    kronos_forecast_pct=kronos_pct,
                    final_signal="NO_TRADE",
                    reason="forecast_conflict",
                )

        if p < self.meta_threshold:
            return DecisionOutcome(
                "skip",
                SignalStatus.REJECTED,
                meta_probability=p,
                leader_score=leader_score,
                kronos_forecast_pct=kronos_pct,
                final_signal="NO_TRADE",
                reason=f"meta_below_threshold:{p:.3f}<{self.meta_threshold}",
            )

        if leader_score.composite < self.min_leader_score:
            return DecisionOutcome(
                "skip",
                SignalStatus.REJECTED,
                meta_probability=p,
                leader_score=leader_score,
                kronos_forecast_pct=kronos_pct,
                final_signal="NO_TRADE",
                reason=f"leader_below_score:{leader_score.composite:.2f}<{self.min_leader_score}",
            )

        # 2. Multi-factor Aggregator Synthesis
        leader_votes = [
            LeaderVote(
                leader_id=event.leader_id,
                direction=side_val,
                score=leader_score.composite_score if hasattr(leader_score, "composite_score") else leader_score.composite,
            )
        ]

        agg_res = self.aggregator.aggregate(
            symbol=event.symbol,
            feature_row=feature_row,
            leader_votes=leader_votes,
            correlation_matrix=correlation_matrix,
            kronos_forecast_pct=kronos_pct,
            kronos_conf=kronos_conf,
            xgb_probability=p,
        )

        final_sig = "BUY" if side_val == "buy" else "SELL"
        verb = "copy" if (leader_score.composite >= self.min_leader_score) else "alert"
        return DecisionOutcome(
            verb,
            SignalStatus.APPROVED,
            meta_probability=p,
            leader_score=leader_score,
            kronos_forecast_pct=kronos_pct,
            final_signal=final_sig,
            reason="approved",
            aggregated_result=agg_res,
        )

    # ------------------------------------------------------------------
    async def evaluate_and_persist(
        self,
        session: AsyncSession,
        event: LeaderTrade,
        leader: Leader,
        feature_row: dict[str, float],
        leader_score: LeaderScore | None = None,
        correlation_matrix: object | None = None,
    ) -> tuple[DecisionOutcome, Signal]:
        """Evaluate + write the ``Signal`` row with contributing factors."""
        score = leader_score if leader_score is not None else LeaderScore()
        outcome = self.evaluate(event, score, feature_row, correlation_matrix=correlation_matrix)

        factors = (
            outcome.aggregated_result.contributing_factors
            if outcome.aggregated_result
            else {"p": outcome.meta_probability}
        )

        side_val = event.side if isinstance(event.side, TradeSide) else TradeSide(event.side)

        sig = Signal(
            timestamp=datetime.now(timezone.utc),
            symbol=event.symbol,
            timeframe=self.settings.market_timeframe,
            leader_direction=event.side.value if hasattr(event.side, "value") else str(event.side),
            kronos_direction="buy" if feature_row.get("kronos_forecast_pct", 0) > 0 else "sell",
            kronos_confidence=feature_row.get("kronos_conf", 0.0),
            xgb_probability=outcome.meta_probability,
            regime=str(feature_row.get("vol_regime", "ranging")),
            final_signal=outcome.final_signal,
            contributing_factors=factors,
            trade_event_id=event.id,
            leader_id=leader.id,
            side=side_val,
            leader_score=score.composite,
            kronos_forecast_pct=feature_row.get("kronos_forecast_pct", 0.0),
            meta_model_probability=outcome.meta_probability,
            threshold=self.meta_threshold,
            verb=outcome.verb,
            status=outcome.status,
            rejection_reason=outcome.reason,
            features_json={c: feature_row.get(c, 0.0) for c in FEATURE_COLUMNS},
        )
        session.add(sig)

        await audit.record_audit(
            session,
            actor="strategy",
            action=f"signal.{outcome.status.value}",
            detail={
                "signal_id": sig.id,
                "event_id": event.id,
                "final_signal": outcome.final_signal,
                "verb": outcome.verb,
                "p": outcome.meta_probability,
                "reason": outcome.reason,
            },
            commit=False,
        )
        await session.commit()
        return outcome, sig