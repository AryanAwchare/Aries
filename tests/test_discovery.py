"""Two-tier leader roster: persistence, ranking, and rotation."""
from __future__ import annotations

from app.leaders.discovery import (
    LeaderRoster,
    RotationResult,
    WatchStats,
    parse_watch_stats,
    rank_watch_stats,
)


def _stats(account_id: str, gain: float, dd: float, trades: int = 10, name: str = "") -> WatchStats:
    return WatchStats(account_id=account_id, name=name or account_id, gain_pct=gain, drawdown_pct=dd, trades=trades)


def test_rank_rewards_gain_and_punishes_drawdown():
    good = _stats("a", gain=10.0, dd=2.0)
    bad = _stats("b", gain=-5.0, dd=40.0)
    assert rank_watch_stats(good) > rank_watch_stats(bad)
    assert 0.0 <= rank_watch_stats(good) <= 1.0


def test_roster_add_and_persist(tmp_path):
    path = tmp_path / "roster.json"
    roster = LeaderRoster(path, max_live=2, max_watch=10, min_observations=1)
    assert roster.add("111", name="GoldFox")
    assert not roster.add("111")  # already known
    assert roster.live_ids() == []

    reloaded = LeaderRoster(path, max_live=2, max_watch=10, min_observations=1)
    assert reloaded.get("111") is not None
    assert reloaded.get("111").name == "GoldFox"


def test_observe_grows_pool(tmp_path):
    roster = LeaderRoster(tmp_path / "roster.json", max_live=1, max_watch=10, min_observations=1)
    roster.observe([_stats("111", 8.0, 3.0), _stats("222", 2.0, 5.0)])
    assert "111" in roster.watch_ids()
    assert "222" in roster.watch_ids()
    assert roster.get("111").observations == 1


def test_rotation_promotes_top_performers_and_demotes_losers(tmp_path):
    roster = LeaderRoster(tmp_path / "roster.json", max_live=2, max_watch=10, min_observations=1)
    roster.add("winner")
    roster.add("loser")
    roster.add("stable")

    # Two scans so everyone meets min observations.
    roster.observe([_stats("winner", 20.0, 1.0), _stats("loser", -10.0, 30.0), _stats("stable", 5.0, 4.0)])
    roster.observe([_stats("winner", 18.0, 1.0), _stats("loser", -12.0, 32.0), _stats("stable", 5.0, 4.0)])

    result = roster.rotate()
    assert isinstance(result, RotationResult)
    assert "winner" in roster.live_ids()
    assert "stable" in roster.live_ids()
    assert "loser" in roster.watch_ids()
    assert len(roster.live_ids()) == 2


def test_promotion_requires_min_observations(tmp_path):
    roster = LeaderRoster(tmp_path / "r.json", max_live=1, max_watch=10, min_observations=3)
    roster.add("fresh")
    roster.observe([_stats("fresh", 50.0, 1.0)])  # only one scan
    result = roster.rotate()
    assert roster.live_ids() == []  # not enough history yet
    assert result.promoted == []


def test_parse_watch_stats_defensive():
    stats = parse_watch_stats({"id": "123", "name": "A", "gain": "4.2", "drawdown": None, "trades": "42"})
    assert stats.account_id == "123"
    assert stats.gain_pct == 4.2
    assert stats.drawdown_pct == 0.0
    assert stats.trades == 42


def test_observe_persists_scores(tmp_path):
    roster = LeaderRoster(tmp_path / "roster.json", max_live=1, max_watch=10, min_observations=3)
    roster.add("perf")
    roster.observe([_stats("perf", 30.0, 1.0)])
    assert roster.get("perf").watch_score > 0.5

    reloaded = LeaderRoster(tmp_path / "roster.json", max_live=1, max_watch=10, min_observations=3)
    assert reloaded.get("perf").watch_score == roster.get("perf").watch_score
    assert reloaded.get("perf").observations == 2


def test_dry_run_observe_does_not_rotate(tmp_path):
    roster = LeaderRoster(tmp_path / "r.json", max_live=1, max_watch=10, min_observations=1)
    roster.observe([_stats("a", 50.0, 1.0)])
    # observe() alone must never change tiers — rotate() is the only mutator
    assert roster.live_ids() == []
    assert len(roster.watch_ids()) == 1