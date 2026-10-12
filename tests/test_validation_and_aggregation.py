"""Tests for Phase 4 (AI/Backtest Validation & Costs) and Phase 5 (Signal Aggregator).

Validates:
1. Transaction Cost Model: realistic spread, commission, slippage, swap, and execution latency.
2. Ablation Comparison across all 6 configurations (leader-only, kronos-only, xgboost-only, leader+kronos, leader+xgboost, full hybrid).
3. Walk-Forward Calibration metrics: Brier score and reliability curve with expected calibration error (ECE).
4. Multi-factor Signal Aggregator: Regime, correlation-adjusted leader votes, Kronos, XGBoost -> final_signal.
5. Correlated leaders treated as one vote, not independent consensus.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.ablation import AblationEngine
from app.backtest.runner import BacktestRunner, TransactionCostModel
from app.backtest.walkforward import compute_calibration_report
from app.config import Settings
from app.execution.killswitch import KillSwitch
from app.features.build_features import calculate_indicator_block
from app.risk.engine import RiskEngine
from app.risk.limiter import RateLimiter
from app.risk.prop_profiles import PropProfile
from app.strategy.aggregator import LeaderVote, SignalAggregator

pytestmark = pytest.mark.asyncio


def _synthetic_ohlcv(n_bars: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timeline = pd.date_range(start="2024-01-01 00:00:00", periods=n_bars, freq="15min", tz="UTC")
    price = 2000.0 + np.cumsum(rng.normal(0.1, 1.5, size=n_bars))
    df = pd.DataFrame(
        {
            "open": price,
            "high": price + rng.uniform(0.5, 2.0, size=n_bars),
            "low": price - rng.uniform(0.5, 2.0, size=n_bars),
            "close": price + rng.normal(0.0, 0.5, size=n_bars),
            "volume": rng.uniform(500, 2000, size=n_bars),
        },
        index=timeline,
    )
    df.attrs["symbol"] = "XAUUSD"
    return df


@pytest.fixture()
def risk_engine():
    prof = PropProfile(
        name="test_profile",
        capital=100_000.0,
        position_risk_pct=0.01,
        max_total_drawdown_pct=0.06,
        max_daily_loss_pct=0.03,
        min_lot=0.01,
        max_lot=10.0,
        lot_step=0.01,
    )
    return RiskEngine(prof, KillSwitch(True), RateLimiter())


# ---------------------------------------------------------------- Transaction Costs Test
def test_backtest_runner_applies_realistic_transaction_costs(risk_engine):
    costs = TransactionCostModel(
        spread_points=0.40,
        commission_per_lot=7.00,
        slippage_points=0.15,
        swap_long_per_day=-10.0,
        execution_delay_bars=1,
    )
    runner = BacktestRunner(risk_engine, prob_threshold=0.50, cost_model=costs)
    df = _synthetic_ohlcv(n_bars=300)
    result = runner.run(df)

    # Asserts realistic costs were tracked and deducted
    if result.n_trades > 0:
        assert result.total_commission_paid > 0
        assert result.total_slippage_paid > 0
        assert result.brier_score >= 0.0


# ---------------------------------------------------------------- Ablation Comparison Test
def test_ablation_comparison_all_six_configurations(risk_engine):
    ablation = AblationEngine(risk_engine, prob_threshold=0.55)
    df = _synthetic_ohlcv(n_bars=250)
    calc_df = calculate_indicator_block(df)
    calc_df["kronos_forecast_pct"] = 0.15
    calc_df["kronos_conf"] = 0.70
    calc_df["xgb_probability"] = 0.68

    configs = [
        "leader_only",
        "kronos_only",
        "xgboost_only",
        "leader_plus_kronos",
        "leader_plus_xgboost",
        "full_hybrid",
    ]

    metrics_list = []
    for cfg in configs:
        m = ablation.evaluate_configuration(calc_df, cfg, split_name="test_2025")
        metrics_list.append(m)
        assert m.configuration == cfg
        assert m.split == "test_2025"

    assert len(metrics_list) == 6


# ---------------------------------------------------------------- Calibration & Reliability Curve Test
def test_walkforward_brier_score_and_reliability_curve():
    # Synthetic predictions and outcomes
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    y_prob = np.array([0.9, 0.2, 0.8, 0.7, 0.3, 0.85, 0.1, 0.4, 0.75, 0.95])

    rep = compute_calibration_report(y_true, y_prob, n_bins=5)
    assert 0.0 <= rep.brier_score <= 1.0
    assert 0.0 <= rep.expected_calibration_error <= 1.0
    assert len(rep.reliability_curve) > 0

    d = rep.to_dict()
    assert "brier_score" in d
    assert "expected_calibration_error" in d
    assert "reliability_curve" in d


# ---------------------------------------------------------------- Signal Aggregator Test
def test_signal_aggregator_correlated_leaders_consolidation():
    aggregator = SignalAggregator(prob_threshold=0.60, high_correlation_threshold=0.80)

    # 3 leaders all voting BUY, but Leader 1 and Leader 2 are 95% correlated
    votes = [
        LeaderVote("L1", "buy", score=1.0),
        LeaderVote("L2", "buy", score=1.0),
        LeaderVote("L3", "sell", score=1.0),
    ]

    # Correlation matrix showing L1 and L2 are correlated (0.95)
    corr_df = pd.DataFrame(
        [
            [1.00, 0.95, 0.10],
            [0.95, 1.00, 0.10],
            [0.10, 0.10, 1.00],
        ],
        index=["L1", "L2", "L3"],
        columns=["L1", "L2", "L3"],
    )

    feature_row = {"trend_strength": 0.10, "vol_regime": "trending_bull"}

    result = aggregator.aggregate(
        symbol="XAUUSD",
        feature_row=feature_row,
        leader_votes=votes,
        correlation_matrix=corr_df,
        kronos_forecast_pct=0.10,
        kronos_conf=0.75,
        xgb_probability=0.70,
    )

    # L1 and L2 consolidated into 1 cluster vote (+1.0), L3 is 1 cluster vote (-1.0)
    # Total clusters = 2
    assert result.contributing_factors["leader_clusters_count"] == 2
    assert result.final_signal == "BUY"
    assert result.direction == "buy"
    assert result.xgb_probability == 0.70


def test_signal_aggregator_rejects_low_probability():
    aggregator = SignalAggregator(prob_threshold=0.65)
    votes = [LeaderVote("L1", "buy", score=1.0)]
    feature_row = {"trend_strength": 0.10, "vol_regime": "trending_bull"}

    # XGBoost P is 0.52 < threshold 0.65
    result = aggregator.aggregate(
        symbol="XAUUSD",
        feature_row=feature_row,
        leader_votes=votes,
        kronos_forecast_pct=0.10,
        kronos_conf=0.80,
        xgb_probability=0.52,
    )

    assert result.final_signal == "NO_TRADE"
    assert "below_threshold" in result.reason