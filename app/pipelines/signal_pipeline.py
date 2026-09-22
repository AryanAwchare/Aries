"""Live signal pipeline — leader event ⇒ features ⇒ hybrid gate ⇒ risk ⇒ order."""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from ..backtest.leakage import KRONOS_TRAINING_CUTOFF  # noqa: F401  (documented dependency)
from ..config import Settings
from ..data.ohlcv import OHLCVRequest
from ..db import audit
from ..db.models import SignalStatus, TradeEvent
from ..execution.executor import Executor
from ..features.build_features import build_feature_row
from ..forecast.kronos import KronosForecaster
from ..leaders.scoring import LeaderScore
from ..logging import get_logger
from ..metamodel.predict import MetaModelPredictor
from ..risk.engine import AccountState, RiskEngine
from ..strategy.engine import StrategyEngine

log = get_logger(__name__)


class SignalPipeline:
    """Composes the whole live stack for one leader event."""

    def __init__(
        self,
        settings: Settings,
        ohlcv_provider,
        forecaster: KronosForecaster,
        strategy: StrategyEngine,
        risk_engine: RiskEngine,
        executor: Executor,
        cot_vector: dict[str, float] | None = None,
    ) -> None:
        self.settings = settings
        self.ohlcv = ohlcv_provider
        self.forecaster = forecaster
        self.strategy = strategy
        self.risk = risk_engine
        self.executor = executor
        self.cot = cot_vector or {}

    async def process(self, session: AsyncSession, event: TradeEvent, leader) -> str:
        """Run a leader event through the full chain. Returns a status string."""
        timeline = event.opened_at
        ohlcv = await self.ohlcv.get_ohlcv(
            OHLCVRequest(
                symbol=event.symbol,
                timeframe=self.settings.market_timeframe,
                start=timeline.replace(tzinfo=None) - _lookback(self.settings),
                end=timeline.replace(tzinfo=None),
            )
        )
        if ohlcv is None or ohlcv.empty:
            raise RuntimeError(f"no market data for {event.symbol} at {timeline}")

        forecast_pct, conf = self.forecaster.predict_pct(ohlcv["close"].to_numpy())
        feature = build_feature_row(
            ohlcv,
            cot=self.cot,
            kronos_forecast_pct=forecast_pct,
            kronos_conf=conf,
        )
        feature["side_buy"] = 1 if event.side.value == "buy" else 0
        feature["side_sell"] = 1 if event.side.value == "sell" else 0

        outcome, decision = await self.strategy.evaluate_and_persist(
            session, event, leader, feature, leader_score=LeaderScore()
        )

        if outcome.status is not SignalStatus.APPROVED:
            await audit.record_audit(
                session, actor="pipeline", action=f"signal.{outcome.status.value}",
                detail={"reason": outcome.reason, "p": outcome.meta_probability},
            )
            await session.commit()
            return outcome.reason

        if outcome.verb == "alert":
            # Phase 1/3 — alert-only mode: notify, do not place orders.
            await audit.record_audit(
                session, actor="pipeline", action="signal.alert",
                detail={"event_id": event.id, "p": outcome.meta_probability, "verb": outcome.verb},
            )
            await session.commit()
            return "alerted"

        # verb == "copy": continue through risk + execution
        entry_ref = event.open_price or float(ohlcv.iloc[-1]["close"])
        account = AccountState(
            equity=self.risk.profile.capital,
            high_water_mark=self.risk.profile.capital,
            daily_start_balance=self.risk.profile.capital,
        )
        exec_outcome = await self.executor.execute_signal(
            session, decision, account, entry_ref_price=entry_ref, stop_points=5.0
        )
        return "filled" if exec_outcome.order and exec_outcome.order.status == "filled" else exec_outcome.message


def _lookback(settings: Settings) -> object:
    from datetime import timedelta

    return timedelta(hours=72)