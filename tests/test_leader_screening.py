"""Phase 2.1 leader discovery + gold screening tests (offline)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.config import Settings
from app.leaders.discovery import LeaderRoster, WatchStats
from app.leaders.models import RawLeaderTrade
from app.leaders.screening import (
    GoldScreenConfig,
    discover_leaders,
    load_seed_ids,
    screen_history,
)

AS_OF = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _trades(
    n: int,
    *,
    symbol: str = "XAUUSD",
    profit_cycle: tuple[float, ...] = (120.0, 120.0, -80.0),
    volume: float = 1.0,
    span_days: int = 400,
) -> list[RawLeaderTrade]:
    out: list[RawLeaderTrade] = []
    for i in range(n):
        opened = AS_OF - timedelta(days=span_days) + timedelta(days=i * (span_days / max(n, 1)))
        profit = profit_cycle[i % len(profit_cycle)]
        out.append(
            RawLeaderTrade(
                external_id=str(i),
                symbol=symbol,
                side="buy",
                volume=volume,
                open_price=2000.0,
                close_price=2001.0,
                profit=profit,
                opened_at=opened,
                closed_at=opened + timedelta(hours=3),
                is_open=False,
            )
        )
    return out


def test_clean_account_qualifies():
    trades = _trades(250)
    result = screen_history(trades, GoldScreenConfig(), as_of=AS_OF)
    assert result.qualified, result.reasons
    assert result.reasons == []
    assert result.metrics["closed_trades"] == 250
    assert result.metrics["profit_factor"] > 1.5
    assert result.composite > 0.0


def test_rejects_young_account():
    trades = _trades(250, span_days=120)
    result = screen_history(trades, GoldScreenConfig(), as_of=AS_OF)
    assert not result.qualified
    assert any(r.startswith("account_too_young") for r in result.reasons)


def test_rejects_insufficient_trades():
    trades = _trades(50)
    result = screen_history(trades, GoldScreenConfig(), as_of=AS_OF)
    assert not result.qualified
    assert any(r.startswith("insufficient_trades") for r in result.reasons)


def test_rejects_demo_when_real_required():
    trades = _trades(250)
    result = screen_history(
        trades, GoldScreenConfig(require_real_account=True), is_real_account=False, as_of=AS_OF
    )
    assert not result.qualified
    assert "demo_account" in result.reasons


def test_rejects_high_drawdown():
    trades = _trades(250)
    result = screen_history(
        trades, GoldScreenConfig(), reported_drawdown_pct=35.0, as_of=AS_OF
    )
    assert not result.qualified
    assert any(r.startswith("drawdown_too_high") for r in result.reasons)


def test_rejects_martingale_pattern():
    volumes = [2.0**i for i in range(12)]
    out: list[RawLeaderTrade] = []
    for i, v in enumerate(volumes):
        opened = AS_OF - timedelta(days=300) + timedelta(days=i)
        out.append(
            RawLeaderTrade(
                external_id=str(i),
                symbol="XAUUSD",
                side="buy",
                volume=v,
                open_price=2000.0,
                close_price=1990.0,
                profit=-100.0,
                opened_at=opened,
                closed_at=opened + timedelta(hours=1),
                is_open=False,
            )
        )
    result = screen_history(out, GoldScreenConfig(min_closed_trades=10), as_of=AS_OF)
    assert not result.qualified
    assert "martingale_pattern" in result.reasons


def test_load_seed_ids_parses_file(tmp_path):
    p = tmp_path / "seeds.txt"
    p.write_text("# comment\n10515575, Gold Trade Pro\n\n11340280\n10515575\n", encoding="utf-8")
    assert load_seed_ids(p) == ["10515575", "11340280"]
    assert load_seed_ids(tmp_path / "missing.txt") == []


class _FakeClient:
    """Returns a clean, verified account for CLEAN and nothing for others."""

    def __init__(self, good_id: str = "CLEAN") -> None:
        self.good_id = good_id

    async def get_account_stats(self, leader_id: str) -> WatchStats:
        if leader_id == self.good_id:
            return WatchStats(
                account_id=leader_id, name="Clean", drawdown_pct=8.0,
                verified_track_record=True, raw={"type": "Real"},
            )
        return WatchStats(account_id=leader_id, verified_track_record=True, raw={})

    async def get_history(self, leader_id: str) -> list[RawLeaderTrade]:
        return _trades(250) if leader_id == self.good_id else []

    async def get_watched_accounts(self) -> list[WatchStats]:
        return [WatchStats(account_id=self.good_id, verified_track_record=True, raw={})]


async def test_discover_leaders_persists_qualified(tmp_path):
    settings = Settings(
        leader_min_discovery_trades=200,
        leader_min_account_age_days=365,
        leader_seed_file=str(tmp_path / "none.txt"),
        myfxbook_leaders="",
    )
    roster = LeaderRoster(tmp_path / "roster.json")
    client = _FakeClient()

    result = await discover_leaders(settings, client, roster, apply=True, as_of=AS_OF)

    assert "CLEAN" in result["qualified"]
    assert result["applied"] is True
    c = roster.get("CLEAN")
    assert c is not None
    assert c.is_eligible
    assert c.last_composite > 0.0
    # persisted to disk
    reloaded = LeaderRoster(tmp_path / "roster.json")
    assert reloaded.get("CLEAN") is not None


async def test_discover_leaders_dry_run_does_not_write(tmp_path):
    settings = Settings(
        leader_min_discovery_trades=200,
        leader_seed_file=str(tmp_path / "none.txt"),
        myfxbook_leaders="",
    )
    roster = LeaderRoster(tmp_path / "roster.json")
    result = await discover_leaders(settings, _FakeClient(), roster, apply=False, as_of=AS_OF)
    assert "CLEAN" in result["qualified"]
    assert roster.get("CLEAN") is None
    assert not (tmp_path / "roster.json").exists()
