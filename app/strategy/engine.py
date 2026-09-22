"""Strategy engine — the hybrid gate.

A signal becomes tradeable ONLY when the leader branch and the meta-model
branch both clear their thresholds. Neither one decides alone. Every
decision is persisted as a ``SignalDecision`` and audited.

Rejection reasons are explicit and machine-readable so the dashboard and
logs can show *why* a setup was skipped.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from ..config import Settings
from ..db import audit
from ..db.models import SignalDecision, SignalStatus, TradeEvent, TradeSide, Leader
from ..features.build_features import build_feature_row, FEATURE_COLUMNS  # noqa: F401
from ..leaders.scoring import LeaderScore
from ..logging import get_logger
from ..metamodel.predict import MetaModelPredictor

log = get_logger(__name__)


@dataclass
class DecisionOutcome:
    verb: str                       # "alert" | "copy" | "skip"
    status: SignalStatus
    meta_probability: float = 0.5
    leader_score: LeaderScore = field(default_factory=LeaderScore)
    kronos_forecast_pct: float = 0.0
    reason: str = ""


class StrategyEngine:
    """Purely-functional gate logic (no IO) wrapped around DB persistence."""

    def __init__(
        self,
        settings: Settings,
        predictor: MetaModelPredictor,
        min_leader_score: float | None = None,
        meta_threshold: float | None = None,
        require_forecast_alignment: bool = True,
    ) -> None:
        self.settings = settings
        self.predictor = predictor
        self.min_leader_score = min_leader_score if min_leader_score is not None else settings.min_leader_score
        self.meta_threshold = meta_threshold if meta_threshold is not None else settings.meta_model_prob_threshold
        self.require_forecast_alignment = require_forecast_alignment

    # ------------------------------------------------------------------
    def evaluate(
        self,
        event: TradeEvent,
        leader_score: LeaderScore,
        feature_row: dict[str, float],
        ohlcv: object | None = None,
    ) -> DecisionOutcome:
        """Run the hybrid gate. Pure — no DB/network side effects."""
        if event.is_open is False and event.close_price is None:
            return DecisionOutcome("skip", SignalStatus.SKIPPED, reason="no_price")

        # Gate 1: calibrated meta-model probability.
        p = self.predictor.predict(feature_row)

        kronos_pct = feature_row.get("kronos_forecast_pct", 0.0)
        if self.require_forecast_alignment and abs(kronos_pct) > 0.1:
            conflicts = (event.side == TradeSide.BUY and kronos_pct < 0) or (
                event.side == TradeSide.SELL and kronos_pct > 0
            )
            if conflicts:
                return DecisionOutcome(
                    "skip",
                    SignalStatus.REJECTED,
                    meta_probability=p,
                    leader_score=leader_score,
                    kronos_forecast_pct=kronos_pct,
                    reason="forecast_conflict",
                )

        if p < self.meta_threshold:
            return DecisionOutcome(
                "skip",
                SignalStatus.REJECTED,
                meta_probability=p,
                leader_score=leader_score,
                kronos_forecast_pct=kronos_pct,
                reason=f"meta_below_threshold:{p:.3f}<{self.meta_threshold}",
            )

        # Gate 2: leader quality.
        if leader_score.composite < self.min_leader_score:
            return DecisionOutcome(
                "skip",
                SignalStatus.REJECTED,
                meta_probability=p,
                leader_score=leader_score,
                kronos_forecast_pct=kronos_pct,
                reason=f"leader_below_score:{leader_score.composite:.2f}<{self.min_leader_score}",
            )

        verb = "copy" if leader_score.composite >= self.min_leader_score + 0.5 else "alert"
        return DecisionOutcome(
            verb, SignalStatus.APPROVED, meta_probability=p, leader_score=leader_score,
            kronos_forecast_pct=kronos_pct, reason="approved",
        )

    # ------------------------------------------------------------------
    async def evaluate_and_persist(
        self,
        session: AsyncSession,
        event: TradeEvent,
        leader: Leader,
        feature_row: dict[str, float],
        leader_score: LeaderScore | None = None,
    ) -> tuple[DecisionOutcome, SignalDecision]:
        """Evaluate + write the ``SignalDecision`` row (and audit line)."""
        score = leader_score if leader_score is not None else LeaderScore()

        outcome = self.evaluate(event, score, feature_row)
        decision = SignalDecision(
            trade_event_id=event.id,
            leader_id=leader.id,
            symbol=event.symbol,
            side=event.side,
            leader_score=score.composite,
            kronos_forecast_pct=feature_row.get("kronos_forecast_pct", 0.0),
            meta_model_probability=outcome.meta_probability,
            threshold=self.meta_threshold,
            verb=outcome.verb,
            status=outcome.status,
            rejection_reason=outcome.reason,
            features_json={c: feature_row.get(c, 0.0) for c in FEATURE_COLUMNS},
        )
        session.add(decision)
        await audit.record_audit(
            session,
            actor="strategy",
            action=f"signal.{outcome.status.value}",
            detail={
                "event_id": event.id,
                "verb": outcome.verb,
                "p": outcome.meta_probability,
                "leader_score": score.composite,
                "reason": outcome.reason,
            },
            commit=False,
        )
        await session.commit()
        return outcome, decision