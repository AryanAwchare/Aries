"""Replay mode — feed historical bars through the LIVE pipeline at speed.

Unlike a pure backtest (which shortcuts the strategy into simulation code),
replay drives the real StrategyEngine → RiskEngine → Executor(PaperBroker)
chain bar by bar — the same path paper/live will use. It's the dry run
before Paper mode.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pandas as pd

from ..db.base import Database
from ..db.models import SignalDecision, SignalStatus, TradeEvent, TradeSide
from ..execution.base import PaperBroker
from ..execution.executor import Executor
from ..execution.killswitch import KillSwitch
from ..features.build_features import build_feature_row
from ..logging import get_logger
from ..metamodel.build_dataset import candidate_events
from ..risk.engine import AccountState, RiskEngine
from ..strategy.engine import StrategyEngine

log = get_logger(__name__)


@dataclass
class ReplayStats:
    bars_seen: int = 0
    signals_fired: int = 0
    orders_filled: int = 0
    rejected: int = 0


async def run_replay(
    db: Database,
    strategy: StrategyEngine,
    risk_engine: RiskEngine,
    executor: Executor,
    ohlcv: pd.DataFrame,
) -> ReplayStats:
    """Iterate the data, snapshot features at candidate events, and run the
    real strategy + risk + executor chain.

    Because it uses the live components, replay also validates the kill
    switch, rate limiter and audit logging end-to-end.
    """
    from ..leaders.models import LeaderProfile
    from ..leaders.repo import get_or_create_leader

    stats = ReplayStats()
    events_df = candidate_events(ohlcv)
    fake_leader_id = "replay-leader"
    leader_profile = LeaderProfile(id=fake_leader_id, name="Replay Leader")

    equity = risk_engine.profile.capital
    hwm = equity
    daily_start = equity
    daily_pnl = 0.0

    async with db.session_factory() as session:
        for bar_i in range(len(ohlcv)):
            ts = ohlcv.index[bar_i]
            stats.bars_seen += 1
            if bar_i > 0 and ts.normalize() != ohlcv.index[bar_i - 1].normalize():
                daily_start = equity
                daily_pnl = 0.0
            if ts not in events_df.index:
                continue

            side = events_df.loc[ts, "side"]
            bar = ohlcv.iloc[bar_i]
            feature = build_feature_row(
                ohlcv.iloc[: bar_i + 1],
                cot=None,
                kronos_forecast_pct=0.0,
                kronos_conf=0.0,
            )
            feature["side_buy"] = 1 if side == "buy" else 0
            feature["side_sell"] = 1 if side == "sell" else 0

            leader = await get_or_create_leader(session, leader_profile)
            event = TradeEvent(
                leader_id=fake_leader_id,
                external_trade_id=f"replay-{ts.isoformat()}",
                symbol=ohlcv.attrs.get("symbol", "XAUUSD"),
                side=TradeSide(side),
                volume=1.0,
                open_price=float(bar["close"]),
                opened_at=ts.to_pydatetime(),
                is_open=True,
            )
            session.add(event)
            await session.flush()

            from ..strategy.engine import LeaderScore

            outcome, decision = await strategy.evaluate_and_persist(
                session, event, leader, feature, leader_score=LeaderScore()
            )
            stats.signals_fired += 1

            if outcome.status is not SignalStatus.APPROVED:
                stats.rejected += 1
                continue

            account = AccountState(
                equity=equity, high_water_mark=hwm,
                daily_start_balance=daily_start, daily_pnl=daily_pnl,
                open_positions=0, now=ts,
            )
            exec_outcome = await executor.execute_signal(
                session, decision, account,
                entry_ref_price=float(bar["close"]), stop_points=5.0,
            )
            if exec_outcome.order is not None and exec_outcome.order.status == "filled":
                stats.orders_filled += 1

    log.info("Replay complete: %s", stats)
    return stats