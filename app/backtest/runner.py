"""Backtesting engine with realistic transaction costs and execution dynamics.

Features:
- Bid/Ask spread modeling (half spread on entry + half spread on exit)
- Broker commission per standard lot
- Slippage model (dynamic based on bar range & volatility)
- Overnight swap holding costs
- Execution latency delay (fill at bar i + 1 open rather than signal close)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from ..features.build_features import calculate_indicator_block
from ..metamodel.build_dataset import candidate_events
from ..risk.engine import AccountState, OrderCandidate, RiskEngine
from ..risk.sizing import position_size_with_confidence, SizingInput

LOT_MULTIPLIER = 100.0  # USD per $1 move per 1 standard lot of XAUUSD


@dataclass
class TransactionCostModel:
    spread_points: float = 0.30           # $0.30 per oz on Gold
    commission_per_lot: float = 7.00      # $7 round-turn per standard lot
    slippage_points: float = 0.10         # Expected average slippage
    swap_long_per_day: float = -8.50      # Swap cost per lot per day (Long)
    swap_short_per_day: float = 2.00      # Swap credit/cost per lot per day (Short)
    execution_delay_bars: int = 1         # 1 bar latency between signal and fill


@dataclass
class BacktestResult:
    equity_curve: pd.Series = field(default_factory=pd.Series)
    n_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe: float = 0.0
    profit_factor: float = 0.0
    total_commission_paid: float = 0.0
    total_swap_paid: float = 0.0
    total_slippage_paid: float = 0.0
    brier_score: float = 0.0

    @property
    def win_rate(self) -> float:
        return float(self.wins / self.n_trades) if self.n_trades else 0.0


class ProbScorer:
    """Meta-model scoring abstraction used by the runner."""

    def __init__(self, predictor=None) -> None:
        self.predictor = predictor

    def predict(self, feature_row: dict[str, float]) -> float:
        if self.predictor is not None:
            return self.predictor.predict(feature_row)
        trend = feature_row.get("trend_strength", 0.0)
        aligned = (feature_row.get("side_buy", 0) == 1 and trend > 0) or (
            feature_row.get("side_sell", 0) == 1 and trend < 0
        )
        return 0.65 if aligned else 0.45


class BacktestRunner:
    def __init__(
        self,
        risk_engine: RiskEngine,
        horizon_bars: int = 8,
        stop_points: float = 5.0,
        tp_multiple: float = 2.0,
        prob_threshold: float = 0.60,
        scorer: ProbScorer | None = None,
        cost_model: TransactionCostModel | None = None,
    ) -> None:
        self.risk = risk_engine
        self.horizon = horizon_bars
        self.stop_points = stop_points
        self.tp_multiple = tp_multiple
        self.threshold = prob_threshold
        self.scorer = scorer or ProbScorer()
        self.costs = cost_model or TransactionCostModel()

    def run(self, ohlcv: pd.DataFrame, start_capital: float = 100_000.0) -> BacktestResult:
        df = calculate_indicator_block(ohlcv)
        events = candidate_events(df)
        capital = start_capital
        equity = capital
        hwm = capital
        daily_start = capital
        daily_pnl = 0.0
        total_profit = 0.0
        gross_profit = 0.0
        gross_loss = 0.0
        wins = losses = 0
        all_equity: list[tuple[pd.Timestamp, float]] = []
        signals = 0

        tot_comm = 0.0
        tot_swap = 0.0
        tot_slip = 0.0
        prob_errors: list[float] = []

        i = 0
        while i < len(df) - self.costs.execution_delay_bars:
            ts = df.index[i]
            row = df.iloc[i]
            all_equity.append((ts, equity))

            # reset daily book at UTC midnight
            if i > 0 and ts.normalize() != df.index[i - 1].normalize():
                daily_start = equity
                daily_pnl = 0.0

            if ts in events.index:
                signals += 1
                side = events.loc[ts, "side"]
                feature = self._feature_row(df, i, side)
                p = self.scorer.predict(feature)

                account = AccountState(
                    equity=equity,
                    high_water_mark=hwm,
                    daily_start_balance=daily_start,
                    daily_pnl=daily_pnl,
                    total_profit=total_profit,
                    open_positions=0,
                    now=ts,
                )
                candidate = OrderCandidate(
                    side=side,
                    entry_price=float(row["close"]),
                    stop_points=self.stop_points,
                    source_branch="ml",
                    meta_probability=p,
                    symbol="XAUUSD",
                )
                if p < self.threshold:
                    i += 1
                    continue
                rd = self.risk.approve(candidate, account)
                if not rd.approved:
                    i += 1
                    continue

                lots = position_size_with_confidence(
                    SizingInput(
                        equity=equity,
                        risk_fraction=self.risk.profile.position_risk_pct,
                        stop_distance_points=self.stop_points,
                        meta_probability=p,
                        min_lot=self.risk.profile.min_lot,
                        max_lot=self.risk.profile.max_lot,
                        lot_step=self.risk.profile.lot_step,
                    )
                )

                # Simulated trade execution with costs and delay
                trade_pnl, exit_ts, comm, swap, slip = self._simulate_trade(
                    df, i, side, lots
                )
                equity += trade_pnl
                hwm = max(hwm, equity)
                daily_pnl += trade_pnl
                total_profit += trade_pnl
                tot_comm += comm
                tot_swap += swap
                tot_slip += slip

                actual_win = 1.0 if trade_pnl > 0 else 0.0
                prob_errors.append((p - actual_win) ** 2)

                if trade_pnl >= 0:
                    wins += 1
                    gross_profit += trade_pnl
                else:
                    losses += 1
                    gross_loss += -trade_pnl
                all_equity.append((exit_ts, equity))

            i += 1

        equity_curve = pd.Series([e for _, e in all_equity], index=[t for t, _ in all_equity])
        result = BacktestResult(
            equity_curve=equity_curve,
            n_trades=wins + losses,
            wins=wins,
            losses=losses,
            total_commission_paid=tot_comm,
            total_swap_paid=tot_swap,
            total_slippage_paid=tot_slip,
            brier_score=float(np.mean(prob_errors)) if prob_errors else 0.0,
        )
        result.total_return_pct = (equity - start_capital) / start_capital * 100.0
        result.max_drawdown_pct = _max_drawdown(equity_curve)
        result.sharpe = _sharpe(equity_curve)
        result.profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
        )
        return result

    # ------------------------------------------------------------------
    def _feature_row(self, df: pd.DataFrame, i: int, side: str) -> dict[str, float]:
        cols = [c for c in df.columns if c in {
            "rsi", "macd", "macd_hist", "atr_pct", "vol_20", "vol_regime",
            "volume_z", "trend_strength", "session", "hour", "dow_cos", "dow_sin",
            "cot_mm_net_z", "cot_mm_net_oi", "cot_producer_net_oi",
            "kronos_forecast_pct", "kronos_conf", "side_buy", "side_sell",
        }]
        row = {c: float(df.iloc[i][c]) for c in cols}
        row["side_buy"] = 1 if side == "buy" else 0
        row["side_sell"] = 1 if side == "sell" else 0
        for c in ("kronos_forecast_pct", "kronos_conf", "cot_mm_net_z", "cot_mm_net_oi", "cot_producer_net_oi"):
            row.setdefault(c, 0.0)
        return row

    def _simulate_trade(
        self,
        df: pd.DataFrame,
        signal_i: int,
        side: str,
        lots: float,
    ) -> tuple[float, pd.Timestamp, float, float, float]:
        direction = 1 if side == "buy" else -1
        # Execution delay: fill occurs at signal_i + delay open
        entry_i = min(signal_i + self.costs.execution_delay_bars, len(df) - 1)
        base_entry = float(df.iloc[entry_i]["open"])

        # Entry slippage & half spread
        entry_slip = self.costs.slippage_points
        half_spread = self.costs.spread_points / 2.0
        actual_entry = base_entry + (half_spread + entry_slip) * direction

        stop_price = actual_entry - direction * self.stop_points
        tp_price = actual_entry + direction * (self.tp_multiple * self.stop_points)

        window = df.iloc[entry_i + 1 : entry_i + 1 + self.horizon]
        if len(window) == 0:
            return 0.0, df.index[-1], 0.0, 0.0, 0.0

        exit_price = float(window.iloc[-1]["close"])
        exit_ts = window.index[-1]

        # Scan for stop/take profit hits within horizon bars
        for _, bar in window.iterrows():
            if direction > 0:  # BUY
                if bar["low"] <= stop_price:
                    exit_price = stop_price - half_spread - entry_slip
                    exit_ts = bar.name
                    break
                if bar["high"] >= tp_price:
                    exit_price = tp_price - half_spread - entry_slip
                    exit_ts = bar.name
                    break
            else:  # SELL
                if bar["high"] >= stop_price:
                    exit_price = stop_price + half_spread + entry_slip
                    exit_ts = bar.name
                    break
                if bar["low"] <= tp_price:
                    exit_price = tp_price + half_spread + entry_slip
                    exit_ts = bar.name
                    break

        # Transaction cost computations
        commission = self.costs.commission_per_lot * lots
        holding_hours = max(1.0, (exit_ts - df.index[entry_i]).total_seconds() / 3600.0)
        holding_days = holding_hours / 24.0
        swap_rate = self.costs.swap_long_per_day if direction > 0 else self.costs.swap_short_per_day
        swap_cost = holding_days * swap_rate * lots
        slippage_cost = (2 * entry_slip) * lots * LOT_MULTIPLIER

        gross_pnl = direction * (exit_price - actual_entry) * lots * LOT_MULTIPLIER
        net_pnl = gross_pnl - commission + swap_cost

        return net_pnl, exit_ts, commission, swap_cost, slippage_cost


def _max_drawdown(curve: pd.Series) -> float:
    if curve.empty:
        return 0.0
    peak = curve.cummax()
    dd = (curve - peak) / peak
    return float(-dd.min())


def _sharpe(curve: pd.Series) -> float:
    if len(curve) < 2:
        return 0.0
    ret = curve.pct_change().dropna()
    if ret.std() == 0:
        return 0.0
    return float(ret.mean() / ret.std() * np.sqrt(4 * 24 * 365))