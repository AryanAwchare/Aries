"""Risk engine + sizing tests — the safety-critical paths."""
from __future__ import annotations

import pytest

from app.execution.killswitch import KillSwitch
from app.risk.engine import AccountState, OrderCandidate, RiskEngine
from app.risk.limiter import RateLimiter
from app.risk.sizing import (
    SizingInput,
    confidence_multiplier,
    position_size,
    position_size_with_confidence,
)


def _candidate(source="ml"):
    return OrderCandidate(side="buy", entry_price=2000.0, stop_points=5.0, source_branch=source)


def test_kill_switch_blocks_everything(risk_engine, healthy_account):
    risk_engine.kill_switch.disable()
    d = risk_engine.approve(_candidate(), healthy_account)
    assert not d.approved and d.reason_code == "kill_switch"
    risk_engine.kill_switch.enable()


def test_copy_trading_forbidden_when_disallowed(profile, healthy_account):
    profile.copy_trading_allowed = False
    engine = RiskEngine(profile, KillSwitch(True), RateLimiter(per_minute=100, per_hour=1000))
    d = engine.approve(_candidate(source="leader"), healthy_account)
    assert not d.approved and d.reason_code == "copy_trading_forbidden"
    # pure-ML branch still fine under a no-copy profile
    d = engine.approve(_candidate(source="ml"), healthy_account)
    assert d.approved


def test_drawdown_emergency_stop(profile, risk_engine):
    account = AccountState(
        equity=90_000.0, high_water_mark=100_000.0,
        daily_start_balance=100_000.0,
    )
    d = risk_engine.approve(_candidate(), account)
    assert not d.approved and d.reason_code == "drawdown_emergency_stop"


def test_operating_ceiling_blocks_before_firm_limit(profile, risk_engine):
    # firm limit 10%, emergency 6%, operating ceiling 4% -> equity at 95.5k is a 4.5% dd
    account = AccountState(
        equity=95_500.0, high_water_mark=100_000.0,
        daily_start_balance=100_000.0,
    )
    d = risk_engine.approve(_candidate(), account)
    assert not d.approved and d.reason_code == "drawdown_operating_ceiling"


def test_daily_loss_soft_limit(profile, risk_engine):
    account = AccountState(
        equity=99_000.0, high_water_mark=100_000.0,
        daily_start_balance=100_000.0, daily_pnl=-2_500.0,  # -2.5% day
    )
    d = risk_engine.approve(_candidate(), account)
    assert not d.approved and d.reason_code == "daily_loss_soft_limit"


def test_daily_loss_firm_breach(profile, risk_engine):
    account = AccountState(
        equity=99_000.0, high_water_mark=100_000.0,
        daily_start_balance=100_000.0, daily_pnl=-3_500.0,  # -3.5% > firm 3%
    )
    d = risk_engine.approve(_candidate(), account)
    assert not d.approved and d.reason_code == "daily_loss_firm_breach"


def test_profit_target_stops_new_entries(profile, risk_engine):
    account = AccountState(
        equity=112_000.0, high_water_mark=112_000.0,
        daily_start_balance=100_000.0,
    )
    d = risk_engine.approve(_candidate(), account)
    assert not d.approved and d.reason_code == "profit_target_reached"


def test_rate_limit_enforced_independently(profile, healthy_account):
    strict = RiskEngine(profile, KillSwitch(True), RateLimiter(per_minute=2, per_hour=100))
    assert strict.approve(_candidate(), healthy_account).approved
    assert strict.approve(_candidate(), healthy_account).approved
    d = strict.approve(_candidate(), healthy_account)
    assert not d.approved and d.reason_code == "order_rate_limit"


def test_max_open_positions(profile, risk_engine):
    account = AccountState(
        equity=100_000.0, high_water_mark=100_000.0,
        daily_start_balance=100_000.0, open_positions=3,
    )
    d = risk_engine.approve(_candidate(), account)
    assert not d.approved and d.reason_code == "max_open_positions"


def test_full_approval_returns_sized_volume(risk_engine, healthy_account):
    d = risk_engine.approve(_candidate(), healthy_account)
    assert d.approved
    assert d.volume > 0 and d.volume <= profile_max_lot()
    assert d.reason_code == "approved"


def profile_max_lot():
    return 10.0


def test_sizing_risk_first():
    inp = SizingInput(
        equity=100_000.0, risk_fraction=0.01, stop_distance_points=5.0,
        meta_probability=0.5,
    )
    lots = position_size(inp)
    # risk $1000 across $5 stop * $100/lot = 2 lots
    assert lots == pytest.approx(2.0, rel=1e-6)


def test_confidence_scales_size():
    assert confidence_multiplier(0.5) == pytest.approx(0.5)
    assert confidence_multiplier(0.75) == pytest.approx(1.25)
    assert confidence_multiplier(0.95) <= 1.5


def test_sizing_negative_stop_returns_zero():
    inp = SizingInput(equity=100_000.0, risk_fraction=0.01, stop_distance_points=-1.0)
    assert position_size_with_confidence(inp) == 0.0