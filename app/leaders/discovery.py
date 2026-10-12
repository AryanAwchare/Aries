"""Two-tier leader discovery + roster management.

Tier architecture:
- WATCH TIER: broad candidate pool (30-50+ accounts). Updated via cheap
  get-watched-accounts bulk calls on a LEADER_RESCORE_INTERVAL_HOURS schedule.
- LIVE TIER: top N performers passing Stage A eligibility (max_live / LIVE_LEADER_POOL_SIZE).
  Only live-tier leaders are polled on the rapid MYFXBOOK_POLL_INTERVAL_SECONDS cycle.
"""
from __future__ import annotations

import itertools
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class WatchStats:
    """Aggregate stats from the cheap bulk call (one row per watched account)."""

    account_id: str
    name: str = ""
    gain_pct: float = 0.0
    drawdown_pct: float = 0.0
    trades: int = 0
    updated_at: str = ""
    verified_track_record: bool = True
    verified_trading_privileges: bool = False
    strategy_type: str = "unknown"
    raw: dict = field(default_factory=dict)


@dataclass
class Candidate:
    id: str
    name: str = ""
    source: str = "manual"          # manual | watchlist | seed
    url: str = ""
    audited: bool = False
    tier: str = "watch"             # watch | live | demoted
    added_at: str = ""
    last_seen_at: str = ""
    observations: int = 0           # number of times it appeared in scans
    watch_score: float = 0.0        # aggregate-based rank (watch tier)
    last_composite: float = 0.0     # trade-based score last computed (live tier)
    verified_track_record: bool = True
    verified_trading_privileges: bool = False
    strategy_type: str = "unknown"
    is_eligible: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Candidate":
        fields = cls.__dataclass_fields__
        return cls(**{k: d.get(k, fields[k].default if fields[k].default is not field else None) for k in fields if k in d})


@dataclass
class RotationResult:
    promoted: list[str] = field(default_factory=list)
    demoted: list[str] = field(default_factory=list)
    note: str = ""


class LeaderRoster:
    """Persistent two-tier roster (JSON file)."""

    def __init__(
        self,
        path: Path,
        max_live: int = 5,
        max_watch: int = 50,
        min_observations: int = 3,
    ) -> None:
        self.path = Path(path)
        self.max_live = max(1, max_live)
        self.max_watch = max(self.max_live, max_watch)
        self.min_observations = max(1, min_observations)
        self._candidates: dict[str, Candidate] = {}
        self.load()

    # ------------------------------------------------------------------ I/O
    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            log.warning("roster %s unreadable — starting empty", self.path)
            return
        self._candidates = {c.id: c for c in (Candidate.from_dict(d) for d in data.get("candidates", []))}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "updated_at": _now_iso(),
            "live_count": len(self.live_ids()),
            "watch_count": len(self.watch_ids()),
            "candidates": [c.to_dict() for c in self._candidates.values()],
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    # ------------------------------------------------------------- lookups
    def get(self, leader_id: str) -> Candidate | None:
        return self._candidates.get(leader_id)

    def live_ids(self) -> list[str]:
        return sorted(c.id for c in self._candidates.values() if c.tier == "live")

    def watch_ids(self) -> list[str]:
        return sorted(c.id for c in self._candidates.values() if c.tier == "watch")

    def all_ids(self) -> list[str]:
        return sorted(self._candidates)

    def add(
        self,
        leader_id: str,
        name: str = "",
        source: str = "manual",
        url: str = "",
        verified_track_record: bool = True,
        verified_trading_privileges: bool = False,
    ) -> bool:
        """Add a candidate to the watch tier. Returns False if already known."""
        leader_id = leader_id.strip()
        if not leader_id or leader_id in self._candidates:
            return False
        now = _now_iso()
        self._candidates[leader_id] = Candidate(
            id=leader_id,
            name=name,
            source=source,
            url=url,
            added_at=now,
            last_seen_at=now,
            observations=1,
            verified_track_record=verified_track_record,
            verified_trading_privileges=verified_trading_privileges,
        )
        self.save()
        return True

    def remove(self, leader_id: str) -> bool:
        removed = self._candidates.pop(leader_id, None) is not None
        if removed:
            self.save()
        return removed

    # ----------------------------------------------------------- rotation
    def observe(self, stats: list[WatchStats]) -> None:
        """Record aggregate stats for every watched account. Grows the pool."""
        for s in stats:
            candidate = self._candidates.get(s.account_id)
            now = _now_iso()
            score = rank_watch_stats(s)
            if candidate is None:
                self._candidates[s.account_id] = Candidate(
                    id=s.account_id,
                    name=s.name,
                    source="watchlist",
                    added_at=now,
                    last_seen_at=now,
                    observations=1,
                    watch_score=score,
                    verified_track_record=s.verified_track_record,
                    verified_trading_privileges=s.verified_trading_privileges,
                    strategy_type=s.strategy_type,
                )
            else:
                candidate.name = s.name or candidate.name
                candidate.last_seen_at = now
                candidate.observations += 1
                candidate.watch_score = score
                candidate.verified_track_record = s.verified_track_record
                candidate.verified_trading_privileges = s.verified_trading_privileges
                candidate.strategy_type = s.strategy_type or candidate.strategy_type
        self.save()

    def rotate(self) -> RotationResult:
        """Promote top performers to live, demote those who lost their spot."""
        result = RotationResult()
        # Only consider candidates that meet minimum observation count and are eligible
        promotable = [
            c
            for c in self._candidates.values()
            if c.observations >= self.min_observations and c.is_eligible and c.verified_track_record
        ]
        # Sort by composite / watch score
        promotable.sort(key=lambda c: (max(c.last_composite, c.watch_score), c.observations), reverse=True)

        reserved: set[str] = set()
        for c in promotable[: max(0, self.max_live)]:
            if c.tier != "live":
                result.promoted.append(c.id)
            c.tier = "live"
            reserved.add(c.id)

        for c in self._candidates.values():
            if c.tier == "live" and c.id not in reserved:
                c.tier = "watch"
                result.demoted.append(c.id)

        if result.promoted or result.demoted:
            self.save()
        result.note = f"live={len(self.live_ids())}/{self.max_live} watch={len(self.watch_ids())}"
        return result

    def set_live(self, leader_ids: list[str]) -> None:
        """Explicit override — set exactly these accounts as the live tier."""
        wanted = set(leader_ids)
        for leader_id, c in self._candidates.items():
            c.tier = "live" if leader_id in wanted else "watch"
        self.save()


def rank_watch_stats(s: WatchStats) -> float:
    """Composite rank from bulk aggregate stats (0..1)."""
    gain = max(0.0, min(1.0, s.gain_pct / 5.0))
    dd_penalty = max(0.0, min(1.0, s.drawdown_pct / 10.0))
    volume = max(0.0, min(1.0, s.trades / 50.0))
    return float(0.6 * gain + 0.3 * (1.0 - dd_penalty) + 0.1 * volume)


def parse_watch_stats(raw: dict) -> WatchStats:
    """Defensive parser for a get-watched-accounts row."""
    # Myfxbook tracks verification flags
    track_rec = bool(raw.get("verifiedTrackRecord", raw.get("tracking", True)))
    privs = bool(raw.get("verifiedTradingPrivileges", raw.get("privileges", False)))
    return WatchStats(
        account_id=str(raw.get("id", raw.get("accountId", ""))),
        name=str(raw.get("name", raw.get("login", ""))),
        gain_pct=_f(raw.get("gain")),
        drawdown_pct=_f(raw.get("drawdown")),
        trades=int(_f(raw.get("trades"))),
        updated_at=str(raw.get("updatedAt", "")),
        verified_track_record=track_rec,
        verified_trading_privileges=privs,
        raw=raw,
    )


def _f(v) -> float:
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def watched_account_ids(raw_rows: list[dict]) -> list[str]:
    """Bulk-call rows -> stable account id."""
    ids = []
    for r in raw_rows:
        aid = r.get("id") or r.get("accountId")
        if aid:
            ids.append(str(aid))
    return list(dict.fromkeys(itertools.chain(ids)))