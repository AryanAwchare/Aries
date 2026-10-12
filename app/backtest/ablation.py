"""Ablation comparison framework for Aries Hybrid Architecture.

Compares 6 model configurations across train (2020-2023), val (2024), and test (2025-2026) splits:
1. Leader-only
2. Kronos-only
3. XGBoost-only
4. Leader + Kronos
5. Leader + XGBoost
6. Full Hybrid (Leader + Kronos + XGBoost + Regime)

Outputs comprehensive comparison metrics:
- Total Return %
- Max Drawdown %
- Win Rate %
- Profit Factor
- Sharpe Ratio
- Total Trades
- Brier Score / Calibration Quality
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from ..features.build_features import calculate_indicator_block
from ..logging import get_logger
from ..metamodel.build_dataset import candidate_events
from ..risk.engine import AccountState, OrderCandidate, RiskEngine
from ..risk.prop_profiles import load_profile
from ..risk.sizing import position_size_with_confidence, SizingInput
from .runner import BacktestResult, BacktestRunner, LOT_MULTIPLIER, TransactionCostModel

log = get_logger(__name__)

AblationConfig = Literal[
    "leader_only",
    "kronos_only",
    "xgboost_only",
    "leader_plus_kronos",
    "leader_plus_xgboost",
    "full_hybrid",
]


@dataclass
class AblationMetrics:
    configuration: str
    split: str
    n_trades: int
    win_rate: float
    total_return_pct: float
    max_drawdown_pct: float
    profit_factor: float
    sharpe: float
    brier_score: float

    def to_dict(self) -> dict:
        return {
            "Configuration": self.configuration,
            "Split": self.split,
            "Trades": self.n_trades,
            "Win Rate %": f"{self.win_rate * 100.0:.1f}%",
            "Return %": f"{self.total_return_pct:.2f}%",
            "Max DD %": f"{self.max_drawdown_pct * 100.0:.2f}%",
            "Profit Factor": f"{self.profit_factor:.2f}",
            "Sharpe": f"{self.sharpe:.2f}",
            "Brier Score": f"{self.brier_score:.4f}",
        }


class AblationEngine:
    def __init__(
        self,
        risk_engine: RiskEngine,
        cost_model: TransactionCostModel | None = None,
        prob_threshold: float = 0.60,
    ) -> None:
        self.risk = risk_engine
        self.costs = cost_model or TransactionCostModel()
        self.threshold = prob_threshold

    def evaluate_configuration(
        self,
        df: pd.DataFrame,
        config_name: AblationConfig,
        split_name: str = "test",
        start_capital: float = 100_000.0,
    ) -> AblationMetrics:
        """Run backtest under a specific ablation configuration."""
        if df.empty or len(df) < 10:
            return AblationMetrics(
                configuration=config_name,
                split=split_name,
                n_trades=0,
                win_rate=0.0,
                total_return_pct=0.0,
                max_drawdown_pct=0.0,
                profit_factor=0.0,
                sharpe=0.0,
                brier_score=0.0,
            )

        events = candidate_events(df) if "side" not in df.columns else df
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
        prob_errors: list[float] = []

        stop_points = 5.0
        tp_multiple = 2.0
        horizon = 8

        for i in range(len(df) - self.costs.execution_delay_bars):
            ts = df.index[i]
            row = df.iloc[i]
            all_equity.append((ts, equity))

            if i > 0 and ts.normalize() != df.index[i - 1].normalize():
                daily_start = equity
                daily_pnl = 0.0

            if ts not in events.index:
                continue

            event_row = events.loc[ts]
            side = event_row["side"] if isinstance(event_row, pd.Series) else event_row.iloc[0]["side"]

            trend_val = float(row.get("trend_strength", 0.0))
            kronos_pct = float(row.get("kronos_forecast_pct", 0.0))
            kronos_conf = float(row.get("kronos_conf", 0.5))
            xgb_p = float(row.get("xgb_probability", 0.65 if (side == "buy" and trend_val > 0) else 0.45))
            regime = str(row.get("vol_regime", "ranging"))

            include_signal = False
            effective_p = 0.50

            if config_name == "leader_only":
                include_signal = True
                effective_p = 0.55

            elif config_name == "kronos_only":
                kronos_agrees = (side == "buy" and kronos_pct > 0.05) or (side == "sell" and kronos_pct < -0.05)
                include_signal = kronos_agrees and kronos_conf >= 0.5
                effective_p = 0.50 + 0.30 * kronos_conf if include_signal else 0.40

            elif config_name == "xgboost_only":
                include_signal = xgb_p >= self.threshold
                effective_p = xgb_p

            elif config_name == "leader_plus_kronos":
                kronos_agrees = (side == "buy" and kronos_pct > 0) or (side == "sell" and kronos_pct < 0)
                include_signal = kronos_agrees
                effective_p = 0.58 if include_signal else 0.42

            elif config_name == "leader_plus_xgboost":
                include_signal = xgb_p >= self.threshold
                effective_p = xgb_p

            elif config_name == "full_hybrid":
                kronos_agrees = (side == "buy" and kronos_pct > 0) or (side == "sell" and kronos_pct < 0)
                regime_favorable = (regime != "high_volatility_crisis")
                xgb_favorable = xgb_p >= self.threshold
                include_signal = kronos_agrees and regime_favorable and xgb_favorable
                effective_p = (xgb_p * 0.5 + kronos_conf * 0.3 + 0.20) if include_signal else 0.40

            if not include_signal:
                continue

            account = AccountState(
                equity=equity, high_water_mark=hwm,
                daily_start_balance=daily_start, daily_pnl=daily_pnl,
                total_profit=total_profit, open_positions=0, now=ts,
            )
            candidate = OrderCandidate(
                side=side, entry_price=float(row["close"]), stop_points=stop_points,
                source_branch="ablation", meta_probability=effective_p, symbol="XAUUSD",
            )
            rd = self.risk.approve(candidate, account)
            if not rd.approved:
                continue

            lots = position_size_with_confidence(
                SizingInput(
                    equity=equity,
                    risk_fraction=self.risk.profile.position_risk_pct,
                    stop_distance_points=stop_points,
                    meta_probability=effective_p,
                    min_lot=self.risk.profile.min_lot,
                    max_lot=self.risk.profile.max_lot,
                    lot_step=self.risk.profile.lot_step,
                )
            )

            direction = 1 if side == "buy" else -1
            entry_i = min(i + self.costs.execution_delay_bars, len(df) - 1)
            base_entry = float(df.iloc[entry_i]["open"])
            half_spread = self.costs.spread_points / 2.0
            actual_entry = base_entry + (half_spread + self.costs.slippage_points) * direction
            stop_price = actual_entry - direction * stop_points
            tp_price = actual_entry + direction * (tp_multiple * stop_points)

            window = df.iloc[entry_i + 1 : entry_i + 1 + horizon]
            if len(window) == 0:
                continue

            exit_price = float(window.iloc[-1]["close"])
            exit_ts = window.index[-1]

            for _, bar in window.iterrows():
                if direction > 0:
                    if bar["low"] <= stop_price:
                        exit_price = stop_price - half_spread - self.costs.slippage_points
                        exit_ts = bar.name
                        break
                    if bar["high"] >= tp_price:
                        exit_price = tp_price - half_spread - self.costs.slippage_points
                        exit_ts = bar.name
                        break
                else:
                    if bar["high"] >= stop_price:
                        exit_price = stop_price + half_spread + self.costs.slippage_points
                        exit_ts = bar.name
                        break
                    if bar["low"] <= tp_price:
                        exit_price = tp_price + half_spread + self.costs.slippage_points
                        exit_ts = bar.name
                        break

            commission = self.costs.commission_per_lot * lots
            holding_hours = max(1.0, (exit_ts - df.index[entry_i]).total_seconds() / 3600.0)
            swap = (holding_hours / 24.0) * (self.costs.swap_long_per_day if direction > 0 else self.costs.swap_short_per_day) * lots
            gross_pnl = direction * (exit_price - actual_entry) * lots * LOT_MULTIPLIER
            trade_pnl = gross_pnl - commission + swap

            equity += trade_pnl
            hwm = max(hwm, equity)
            daily_pnl += trade_pnl
            total_profit += trade_pnl

            actual_win = 1.0 if trade_pnl > 0 else 0.0
            prob_errors.append((effective_p - actual_win) ** 2)

            if trade_pnl >= 0:
                wins += 1
                gross_profit += trade_pnl
            else:
                losses += 1
                gross_loss += -trade_pnl
            all_equity.append((exit_ts, equity))

        equity_curve = pd.Series([e for _, e in all_equity], index=[t for t, _ in all_equity])
        n_trades = wins + losses
        win_rate = float(wins / n_trades) if n_trades else 0.0
        total_return = (equity - start_capital) / start_capital * 100.0
        peak = equity_curve.cummax() if not equity_curve.empty else pd.Series([1.0])
        dd_series = (equity_curve - peak) / peak if not equity_curve.empty else pd.Series([0.0])
        max_dd = float(-dd_series.min()) if not dd_series.empty else 0.0

        if len(equity_curve) >= 2:
            ret = equity_curve.pct_change().dropna()
            sharpe = float(ret.mean() / ret.std() * np.sqrt(4 * 24 * 365)) if ret.std() > 0 else 0.0
        else:
            sharpe = 0.0

        pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
        brier = float(np.mean(prob_errors)) if prob_errors else 0.0

        return AblationMetrics(
            configuration=config_name,
            split=split_name,
            n_trades=n_trades,
            win_rate=win_rate,
            total_return_pct=total_return,
            max_drawdown_pct=max_dd,
            profit_factor=pf if np.isfinite(pf) else 999.0,
            sharpe=sharpe,
            brier_score=brier,
        )

    def run_all_ablations(self, df_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Run all 6 configurations across splits (train, validation, test)."""
        configs: list[AblationConfig] = [
            "leader_only",
            "kronos_only",
            "xgboost_only",
            "leader_plus_kronos",
            "leader_plus_xgboost",
            "full_hybrid",
        ]
        results = []
        for split_name, split_df in df_dict.items():
            if split_df.empty or len(split_df) < 10:
                continue
            calc_df = calculate_indicator_block(split_df)
            for cfg in configs:
                m = self.evaluate_configuration(calc_df, cfg, split_name=split_name)
                results.append(m.to_dict())

        return pd.DataFrame(results)