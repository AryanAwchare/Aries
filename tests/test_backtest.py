"""End-to-end smoke tests: backtest/replay/execution without real deps."""
from __future__ import annotations

import pytest

from app.backtest.leakage import run_leakage_report
from app.backtest.runner import BacktestRunner, ProbScorer
from app.execution.base import PaperBroker, OrderRequest
from app.execution.killswitch import KillSwitch
from app.risk.engine import RiskEngine
from app.risk.limiter import RateLimiter


def test_paper_broker_fills(risk_engine):
    broker = PaperBroker()

    import asyncio

    async def _go():
        return await broker.place_market_order(
            OrderRequest(symbol="XAUUSD", side="buy", volume=1.0, entry_ref_price=2000.0)
        )

    out = asyncio.run(_go())
    assert out.ok
    assert out.fill_price > 0


def test_backtest_runner_runs_on_synthetic(synthetic_ohlcv, profile):
    risk = RiskEngine(profile, KillSwitch(True), RateLimiter(per_minute=100, per_hour=1000))
    runner = BacktestRunner(risk, scorer=ProbScorer())
    result = runner.run(synthetic_ohlcv, start_capital=profile.capital)
    assert result.n_trades >= 0
    assert isinstance(result.total_return_pct, float)
    assert 0.0 <= result.max_drawdown_pct <= 1.0
    assert result.equity_curve.index[0] == synthetic_ohlcv.index[0]


def test_leakage_report_warns_on_synthetic(synthetic_ohlcv):
    report = run_leakage_report(synthetic_ohlcv, test_start="2024-01-01")
    # Kronos pretraining overlap is expected on 2024 data
    assert report.kronos_test_overlap is True
    assert report.features_causal is True


def test_rate_limiter_reset(profile):
    limiter = RateLimiter(per_minute=1, per_hour=2)
    assert limiter.allowed()
    assert not limiter.allowed()
    limiter.reset()
    assert limiter.allowed()