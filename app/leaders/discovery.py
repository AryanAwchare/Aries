"""Leader discovery + two-tier rotation.

Myfxbook has **no webhooks and no discovery API**, which shapes this module:

* Finding leaders is a *one-time* manual step on myfxbook.com (Top Systems /
  Follow pages). ``scan-leaders`` then auto-selects the best performers, so you
  never hand-pick "the top 3-5" — the roster does it from the whole watch tier.
* Live event detection (``get-history`` / ``get-open-trades``) is per-account
  and rate-limited, so only the small **live tier** is polled every cycle.
  The rest of the pool is ranked with one cheap bulk ``get-watched-accounts``
  call and only graduates after enough observations.

Tiers
-----
watch  : whole candidate pool (up to ``max_watch``). Ranked cheaply via bulk
         aggregate stats; each account must appear on your Myfxbook watch list
         (Follow) or be explicitly added with ``scan-leaders --add``.
live   : top ``max_live`` promotable candidates — the only accounts polled for
         actual open/closed trades each cycle.

Rotation is deliberate: promotion requires ``min_observations`` worth of data,
and underperformers are demoted back to watch.
"""
from __future__ import annotations

import json
import itertools
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
    raw: dict = field(default_factory=dict)


@dataclass
class Candidate:
    id: str
    name: str = ""
    source: str = "manual"          # manual | watchlist | seed
    url: str = ""
    audited: bool = False
    tier: str = "watch"             # watch | live
    added_at: str = ""
    last_seen_at: str = ""
    observations: int = 0           # number of times it appeared in scans
    watch_score: float = 0.0        # aggregate-based rank (watch tier)
    last_composite: float = 0.0     # trade-based score last computed (live tier)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Candidate":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})


@dataclass
class RotationResult:
    promoted: list[str] = field(default_factory=list)
    demoted: list[str] = field(default_factory=list)
    note: str = ""


class LeaderRoster:
    """Persistent two-tier roster (JSON file). Thread-agnostic; cheap file I/O.

    Rotation is idempotent and deterministic given the same stats, so it can
    run as a scheduled daily job or on-demand without double-side effects.
    """

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
            "version": 1,
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

    def add(self, leader_id: str, name: str = "", source: str = "manual", url: str = "") -> bool:
        """Add a candidate to the watch tier. Returns False if already known."""
        leader_id = leader_id.strip()
        if not leader_id or leader_id in self._candidates:
            return False
        now = _now_iso()
        self._candidates[leader_id] = Candidate(
            id=leader_id, name=name, source=source, url=url,
            added_at=now, last_seen_at=now, observations=1,
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
        """Record aggregate stats for every watched account. Grows the pool.

        Non-destructive to tiers; persists updated watch scores so rankings are
        durable between scans. Rotate() is the only operation that moves tiers.
        """
        for s in stats:
            candidate = self._candidates.get(s.account_id)
            now = _now_iso()
            if candidate is None:
                self._candidates[s.account_id] = Candidate(
                    id=s.account_id, name=s.name, source="watchlist",
                    added_at=now, last_seen_at=now, observations=1,
                    watch_score=rank_watch_stats(s),
                )
            else:
                candidate.name = s.name or candidate.name
                candidate.last_seen_at = now
                candidate.observations += 1
                candidate.watch_score = rank_watch_stats(s)
        self.save()

    def rotate(self) -> RotationResult:
        """Promote top performers to live, demote those who lost their spot."""
        result = RotationResult()
        promotable = [
            c
            for c in self._candidates.values()
            if c.tier == "watch" and c.observations >= self.min_observations
        ]
        promotable.sort(key=lambda c: (c.watch_score, c.observations), reverse=True)
        reserved: set[str] = set()
        for c in promotable[: max(0, self.max_live)]:
            c.tier = "live"
            reserved.add(c.id)
            result.promoted.append(c.id)

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
    """Composite rank from bulk aggregate stats (0..1).

    Rewards positive gains, punishes drawdown, and gently favours accounts
    with enough volume to be measured. Bounded — never NaN.
    """
    gain = max(0.0, min(1.0, s.gain_pct / 5.0))
    dd_penalty = max(0.0, min(1.0, s.drawdown_pct / 10.0))
    volume = max(0.0, min(1.0, s.trades / 50.0))
    return float(0.6 * gain + 0.3 * (1.0 - dd_penalty) + 0.1 * volume)


def parse_watch_stats(raw: dict) -> WatchStats:
    """Defensive parser for a ``get-watched-accounts`` row (schema varies)."""
    return WatchStats(
        account_id=str(raw.get("id", raw.get("accountId", ""))),
        name=str(raw.get("name", raw.get("login", ""))),
        gain_pct=_f(raw.get("gain")),
        drawdown_pct=_f(raw.get("drawdown")),
        trades=int(_f(raw.get("trades"))),
        updated_at=str(raw.get("updatedAt", "")),
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
    """Bulk-call rows -> stable account id (``id`` first, ``accountId`` fallback)."""
    ids = []
    for r in raw_rows:
        aid = r.get("id") or r.get("accountId")
        if aid:
            ids.append(str(aid))
    return list(dict.fromkeys(itertools.chain(ids)))