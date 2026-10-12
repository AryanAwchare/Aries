"""Tests for Phase 3: Leader Intelligence Overhaul.

Covers:
- Stage A: Eligibility Gate (track record verification, symbol, age, trades, martingale/grid filter, drawdown)
- Stage B: Multi-factor composite scoring with configurable weights
- Stage C: Behavior classification (trend_following, scalping, swing, martingale_like, grid, etc.)
- Stage D: Correlation engine (pairwise correlation & cluster grouping)
- Stage E: Two-tier roster promotion and demotion
- Stage F: Live loss-rate circuit breaker demoting mid-cycle on consecutive loss / drawdown breach
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pytest

from app.config import Settings
from app.db.base import Database
from app.leaders.circuit_breaker import LeaderLiveCircuitBreaker
from app.leaders.classifier import classify_behavior
from app.leaders.correlation import LeaderCorrelationEngine
from app.leaders.discovery import Candidate, LeaderRoster, WatchStats
from app.leaders.eligibility import (
    LeaderEligibilityGate,
    detect_grid,
    detect_martingale,
)
from app.leaders.models import LeaderProfile, RawLeaderTrade
from app.leaders.scoring import compute_score

pytestmark = pytest.mark.asyncio


def _make_trade(
    ext_id: str,
    symbol: str = "XAUUSD",
    side: str = "buy",
    volume: float = 0.1,
    profit: float = 50.0,
    open_mins_ago: int = 120,
    dur_mins: int = 30,
) -> RawLeaderTrade:
    now = datetime.now(timezone.utc)
    op = now - timedelta(minutes=open_mins_ago)
    cl = op + timedelta(minutes=dur_mins)
    return RawLeaderTrade(
        external_id=ext_id,
        symbol=symbol,
        side=side,
        volume=volume,
        open_price=2000.0,
        close_price=2005.0 if side == "buy" else 1995.0,
        profit=profit,
        opened_at=op,
        closed_at=cl,
        is_open=False,
    )


# ---------------------------------------------------------------- STAGE A: Eligibility Gate
def test_eligibility_gate_rejection_unverified_track_record():
    gate = LeaderEligibilityGate()
    profile = LeaderProfile(
        id="lead-1",
        name="L1",
        account_age=200,
        verified_track_record=False,  # Unverified!
        verified_trading_privileges=True,
    )
    trades = [_make_trade(f"t{i}", profit=10.0) for i in range(120)]
    res = gate.evaluate(profile, trades)
    assert res.eligible is False
    assert "unverified_track_record" in res.rejection_reasons


def test_eligibility_gate_rejection_non_xauusd():
    gate = LeaderEligibilityGate()
    profile = LeaderProfile(
        id="lead-2",
        name="L2",
        account_age=200,
        verified_track_record=True,
    )
    trades = [_make_trade(f"t{i}", symbol="EURUSD") for i in range(120)]
    res = gate.evaluate(profile, trades)
    assert res.eligible is False
    assert "no_trades_in_XAUUSD" in res.rejection_reasons


def test_eligibility_gate_rejection_underage():
    gate = LeaderEligibilityGate()
    profile = LeaderProfile(
        id="lead-3",
        name="L3",
        account_age=60,  # 60 days < 180 days default
        verified_track_record=True,
    )
    trades = [_make_trade(f"t{i}") for i in range(120)]
    res = gate.evaluate(profile, trades)
    assert res.eligible is False
    assert any("account_age_below_min" in r for r in res.rejection_reasons)


def test_eligibility_gate_rejection_insufficient_trades():
    gate = LeaderEligibilityGate()
    profile = LeaderProfile(
        id="lead-4",
        name="L4",
        account_age=250,
        verified_track_record=True,
    )
    trades = [_make_trade(f"t{i}") for i in range(40)]  # 40 trades < 100 default
    res = gate.evaluate(profile, trades)
    assert res.eligible is False
    assert any("insufficient_closed_trades" in r for r in res.rejection_reasons)


def test_eligibility_gate_rejection_martingale():
    gate = LeaderEligibilityGate()
    profile = LeaderProfile(
        id="lead-5",
        name="L5",
        account_age=250,
        verified_track_record=True,
    )
    # Generate a classic martingale pattern: volume doubles after consecutive losses
    trades = []
    for i in range(110):
        # Loss sequence with escalating size
        if i % 10 in (0, 1, 2):
            vol = 0.1 * (2 ** (i % 10))
            trades.append(_make_trade(f"t{i}", volume=vol, profit=-100.0 * vol))
        else:
            trades.append(_make_trade(f"t{i}", volume=0.1, profit=50.0))

    assert detect_martingale(trades) is True
    res = gate.evaluate(profile, trades)
    assert res.eligible is False
    assert "martingale_pattern_detected" in res.rejection_reasons


def test_eligibility_gate_passes_healthy_leader():
    gate = LeaderEligibilityGate()
    profile = LeaderProfile(
        id="lead-healthy",
        name="Healthy Gold Leader",
        account_age=300,
        verified_track_record=True,
        verified_trading_privileges=True,
    )
    # 150 consistent winning/losing trades in XAUUSD
    trades = []
    for i in range(150):
        pnl = 40.0 if i % 3 != 0 else -20.0
        trades.append(_make_trade(f"t{i}", volume=0.1, profit=pnl, open_mins_ago=1000 - i * 5))

    res = gate.evaluate(profile, trades)
    assert res.eligible is True
    assert len(res.rejection_reasons) == 0


# ---------------------------------------------------------------- STAGE B: Composite Scoring
def test_composite_scoring_custom_weights():
    settings = Settings(
        weight_risk_adjusted_return=0.30,
        weight_profit_factor=0.30,
        weight_drawdown_quality=0.10,
        weight_consistency=0.10,
        weight_expectancy=0.10,
        weight_longevity=0.05,
        weight_recent_performance=0.05,
    )
    profile = LeaderProfile(id="l1", name="L1", account_age=400)
    trades = [_make_trade(f"t{i}", profit=50.0 if i % 4 != 0 else -30.0) for i in range(100)]
    score = compute_score(trades, profile, settings)

    assert score.composite_score > 0.5
    assert score.profit_factor > 1.5
    assert score.win_rate == 0.75
    assert score.expectancy > 0.0


# ---------------------------------------------------------------- STAGE C: Behavior Classification
def test_behavior_classification_scalper_and_swing():
    # Scalper: 5 minute durations
    scalper_trades = [_make_trade(f"t{i}", dur_mins=5) for i in range(20)]
    assert classify_behavior(scalper_trades) == "scalping"

    # Swing: 2880 minute (2 days) durations
    swing_trades = [_make_trade(f"t{i}", dur_mins=2880) for i in range(20)]
    assert classify_behavior(swing_trades) == "swing"


# ---------------------------------------------------------------- STAGE D: Correlation Engine
def test_correlation_engine_clusters_aligned_leaders():
    corr_engine = LeaderCorrelationEngine(high_correlation_threshold=0.80)
    now = datetime.now(timezone.utc)

    # Leader A and B take identical trades at identical times
    trades_a = [
        RawLeaderTrade(f"a_{i}", "XAUUSD", "buy", 0.1, 2000.0, opened_at=now - timedelta(days=i), closed_at=now - timedelta(days=i) + timedelta(hours=2), profit=10.0, is_open=False)
        for i in range(1, 20)
    ]
    trades_b = [
        RawLeaderTrade(f"b_{i}", "XAUUSD", "buy", 0.1, 2000.0, opened_at=now - timedelta(days=i), closed_at=now - timedelta(days=i) + timedelta(hours=2), profit=10.0, is_open=False)
        for i in range(1, 20)
    ]
    # Leader C trades opposite / different times
    trades_c = [
        RawLeaderTrade(f"c_{i}", "XAUUSD", "sell", 0.1, 2000.0, opened_at=now - timedelta(days=i, hours=5), closed_at=now - timedelta(days=i, hours=3), profit=10.0, is_open=False)
        for i in range(1, 20)
    ]

    corr_ab = corr_engine.compute_pairwise_correlation(trades_a, trades_b)
    assert corr_ab > 0.90

    matrix = corr_engine.compute_matrix({"A": trades_a, "B": trades_b, "C": trades_c})
    clusters = corr_engine.cluster_correlated_leaders(matrix)

    # A and B should be in the same cluster
    cluster_with_a = next(c for c in clusters if "A" in c)
    assert "B" in cluster_with_a


# ---------------------------------------------------------------- STAGE E & F: Two-Tier Roster & Circuit Breaker
async def test_two_tier_roster_promotion_and_demotion(tmp_path):
    roster_path = tmp_path / "test_roster.json"
    roster = LeaderRoster(roster_path, max_live=2, max_watch=10, min_observations=2)

    roster.add("winner1", name="W1", verified_track_record=True)
    roster.add("winner2", name="W2", verified_track_record=True)
    roster.add("loser1", name="L1", verified_track_record=True)
    roster.add("unverified", name="U1", verified_track_record=False)

    stats = [
        WatchStats(account_id="winner1", gain_pct=25.0, drawdown_pct=3.0, trades=150, verified_track_record=True),
        WatchStats(account_id="winner2", gain_pct=20.0, drawdown_pct=4.0, trades=120, verified_track_record=True),
        WatchStats(account_id="loser1", gain_pct=-10.0, drawdown_pct=35.0, trades=80, verified_track_record=True),
        WatchStats(account_id="unverified", gain_pct=40.0, drawdown_pct=2.0, trades=200, verified_track_record=False),
    ]

    # Two observations to qualify for rotation
    roster.observe(stats)
    roster.observe(stats)

    res = roster.rotate()
    assert "winner1" in roster.live_ids()
    assert "winner2" in roster.live_ids()
    assert "loser1" not in roster.live_ids()
    assert "unverified" not in roster.live_ids()  # Blocked by track record requirement


async def test_live_circuit_breaker_mid_cycle_demotion(tmp_path):
    roster_path = tmp_path / "cb_roster.json"
    settings = Settings(leader_consecutive_loss_limit=3, leader_live_drawdown_limit_pct=5.0)
    roster = LeaderRoster(roster_path, max_live=2, max_watch=10, min_observations=1)

    roster.add("lead-live", name="Live Leader")
    roster.set_live(["lead-live"])
    assert "lead-live" in roster.live_ids()

    db_path = tmp_path / "cb_test.db"
    settings.database_url = f"sqlite+aiosqlite:///{db_path}"
    db = Database(settings)
    await db.create_all()

    cb = LeaderLiveCircuitBreaker(roster, settings)

    async with db.session_factory() as session:
        # Loss 1
        t1 = _make_trade("t1", profit=-50.0)
        demoted1 = await cb.update_trade_and_check(session, "lead-live", t1)
        assert demoted1 is False
        assert "lead-live" in roster.live_ids()

        # Loss 2
        t2 = _make_trade("t2", profit=-60.0)
        demoted2 = await cb.update_trade_and_check(session, "lead-live", t2)
        assert demoted2 is False
        assert "lead-live" in roster.live_ids()

        # Loss 3 -> Hits limit (3 consecutive losses) -> Immediate Demotion!
        t3 = _make_trade("t3", profit=-40.0)
        demoted3 = await cb.update_trade_and_check(session, "lead-live", t3)
        assert demoted3 is True
        assert "lead-live" not in roster.live_ids()
        assert "lead-live" in roster.watch_ids()

    await db.dispose()