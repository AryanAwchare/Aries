"""Vectorised-style backtest of the hybrid signal over historical OHLCV.

Simulates:
  1. candidate events (leader-trade proxy) from the feature block
  2. meta-model probability (real predictor when available, else a neutral/base
     scorer so the harness runs on a fresh checkout)
  3. the risk engine (drawdown, daily-loss, rate limits, sizing)
  4. realistic trade PnL: lots * $100/oz * direction * price change, with stop
     and take-profit honoured on the bar path.

The point of this runner is validation, not realism-perfect fills — Phase 4's
walk-forward results are what gate moving to paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..features.build_features import calculate_indicator_block
from ..metamodel.build_dataset import candidate_events
from ..risk.engine import AccountState, OrderCandidate, RiskEngine
from ..risk.sizing import position_size_with_confidence, SizingInput

LOT_MULTIPLIER = 100.0  # USD per $1 move per 1 lot of XAUUSD


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

    @property
    def win_rate(self) -> float:
        return float(self.wins / self.n_trades) if self.n_trades else 0.0


class ProbScorer:
    """Meta-model scoring abstraction used by the runner.

    Provide a real ``MetaModelPredictor`` when trained; otherwise a neutral
    scorer that keys off trend strength so the pipeline is exercisable.
    """

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
    ) -> None:
        self.risk = risk_engine
        self.horizon = horizon_bars
        self.stop_points = stop_points
        self.tp_multiple = tp_multiple
        self.threshold = prob_threshold
        self.scorer = scorer or ProbScorer()

    def run(self, ohlcv: pd.DataFrame, start_capital: float = 100_000.0) -> BacktestResult:
        df = calculate_indicator_block(ohlcv)
        events = candidate_events(df)
        capital = start_capital
        equity_start = capital
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

        for i in range(len(df)):
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
                    equity=equity, high_water_mark=hwm,
                    daily_start_balance=daily_start, daily_pnl=daily_pnl,
                    total_profit=total_profit, open_positions=0, now=ts,
                )
                candidate = OrderCandidate(
                    side=side, entry_price=float(row["close"]), stop_points=self.stop_points,
                    source_branch="ml", meta_probability=p, symbol="XAUUSD",
                )
                if p < self.threshold:
                    continue
                rd = self.risk.approve(candidate, account)
                if not rd.approved:
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
                trade_pnl, exit_ts = self._simulate_trade(
                    df, i, side, lots, float(row["close"])
                )
                equity += trade_pnl
                hwm = max(hwm, equity)
                daily_pnl += trade_pnl
                total_profit += trade_pnl
                if trade_pnl >= 0:
                    wins += 1
                    gross_profit += trade_pnl
                else:
                    losses += 1
                    gross_loss += -trade_pnl
                all_equity.append((exit_ts, equity))

        equity_curve = pd.Series([e for _, e in all_equity], index=[t for t, _ in all_equity])
        result = BacktestResult(equity_curve=equity_curve, n_trades=wins + losses, wins=wins, losses=losses)
        result.total_return_pct = (equity - start_capital) / start_capital * 100.0
        result.max_drawdown_pct = _max_drawdown(equity_curve)
        result.sharpe = _sharpe(equity_curve)
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
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

    def _simulate_trade(self, df, entry_i, side, lots, entry):
        direction = 1 if side == "buy" else -1
        stop = self.stop_points / entry
        tp = self.tp_multiple * self.stop_points / entry
        window = df.iloc[entry_i + 1: entry_i + 1 + self.horizon]
        if len(window) == 0:
            return 0.0, df.index[-1]
        for _, bar in window.iterrows():
            if direction > 0:
                if bar["low"] <= entry * (1 - stop):
                    return -self.stop_points * lots * LOT_MULTIPLIER, bar.name
                if bar["high"] >= entry * (1 + tp):
                    return tp * entry * lots * LOT_MULTIPLIER, bar.name
            else:
                if bar["high"] >= entry * (1 + stop):
                    return -self.stop_points * lots * LOT_MULTIPLIER, bar.name
                if bar["low"] <= entry * (1 - tp):
                    return tp * entry * lots * LOT_MULTIPLIER, bar.name
        exit_price = window.iloc[-1]["close"]
        return direction * (exit_price - entry) * lots * LOT_MULTIPLIER, window.index[-1]


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
    return float(ret.mean() / ret.std() * np.sqrt(4 * 24 * 365))  # approximated annualisation