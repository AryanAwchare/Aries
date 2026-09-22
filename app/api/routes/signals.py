"""Recent signal decisions + manual override to re-run the hybrid gate."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.base import Database
from ...db.models import SignalDecision, SignalStatus, TradeEvent
from ...features.build_features import build_feature_row
from ...strategy.engine import LeaderScore

router = APIRouter(prefix="/api/signals", tags=["signals"])


async def _session(request: Request) -> AsyncSession:
    db: Database = request.app.state.db
    async with db.session_factory() as session:
        yield session


@router.get("")
async def list_signals(limit: int = 50, session: AsyncSession = Depends(_session)):
    stmt = select(SignalDecision).order_by(SignalDecision.decided_at.desc()).limit(limit)
    rows = (await session.execute(stmt)).scalars().all()
    return [
        {
            "id": s.id,
            "symbol": s.symbol,
            "side": s.side.value,
            "verb": s.verb,
            "status": s.status.value,
            "p": s.meta_model_probability,
            "leader_score": s.leader_score,
            "threshold": s.threshold,
            "reason": s.rejection_reason,
            "decided_at": s.decided_at,
        }
        for s in rows
    ]


@router.post("/{signal_id}/re-evaluate")
async def re_evaluate(signal_id: str, request: Request, session: AsyncSession = Depends(_session)):
    """Re-run the hybrid gate on a stored decision's underlying trade event.

    Used during tuning (Phase 4) to see how a changed threshold would have
    decided an old setup.
    """
    decision = await session.get(SignalDecision, signal_id)
    if decision is None or decision.trade_event_id is None:
        raise HTTPException(404, "decision or source trade not found")
    event = await session.get(TradeEvent, decision.trade_event_id)
    if event is None:
        raise HTTPException(404, "source trade not found")

    feature = {**decision.features_json}
    feature["side_buy"] = 1 if decision.side.value == "buy" else 0
    feature["side_sell"] = 1 if decision.side.value == "sell" else 0

    from ... import services
    from ...metamodel.predict import MetaModelPredictor
    from ...risk.prop_profiles import load_profile

    settings = request.app.state.settings
    profile = load_profile(settings.BASE_DIR / "app" / "risk" / "prop_profiles" / f"{settings.active_prop_profile}.yaml")
    predictor = MetaModelPredictor(view_model_dir(settings), settings.meta_model_prob_threshold)

    from ...strategy.engine import StrategyEngine

    engine = StrategyEngine(settings, predictor, require_forecast_alignment=False)
    outcome = engine.evaluate(event, LeaderScore(), feature)
    return {"verb": outcome.verb, "status": outcome.status.value, "p": outcome.meta_probability, "reason": outcome.reason}


def view_model_dir(settings):
    return settings.BASE_DIR / "data" / "models"