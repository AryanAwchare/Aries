"""Leader polling + two-tier roster refresh (no scheduler dependency).

Kept dependency-light so the CLI (``scan-leaders``, ``poll-leaders``) can reuse
the exact same code the scheduler jobs call — one path, one behaviour.
"""
from __future__ import annotations

from ..config import Settings
from ..db import audit
from ..logging import get_logger
from .discovery import LeaderRoster
from .models import LeaderProfile
from .myfxbook import MyfxbookClient
from .repo import get_or_create_leader, store_daily_score, upsert_trade_event
from .scoring import compute_score, today_utc

log = get_logger(__name__)


def load_roster(settings: Settings) -> LeaderRoster:
    return LeaderRoster(
        settings.roster_path,
        max_live=settings.max_live_leaders,
        max_watch=settings.max_watch_leaders,
        min_observations=settings.leader_min_observations,
    )


def live_tier(settings: Settings) -> list[str]:
    roster = load_roster(settings)
    live = roster.live_ids()
    if live or roster.all_ids():
        return live
    return settings.leader_ids  # pre-roster fallback: the curated env list


async def poll_leader_trades(settings: Settings, session_factory, myfxbook: MyfxbookClient) -> int:
    """Poll the **live tier only** for new XAUUSD trades, persist + alert.

    Only ``max_live`` leaders are touched per cycle so per-account API requests
    stay bounded and well under Myfxbook's rate limits.
    """
    leader_ids = live_tier(settings)
    inserted = 0
    for leader_id in leader_ids:
        try:
            open_trades = await myfxbook.get_open_trades(leader_id)
            closed = await myfxbook.get_history(leader_id)
        except Exception as exc:  # network / API hiccup — do not kill the loop
            log.warning("Leader %s poll failed: %s", leader_id, exc)
            continue

        profile = LeaderProfile(id=leader_id, name=f"leader-{leader_id}", url="", audited=True)
        async with session_factory() as session:
            leader = await get_or_create_leader(session, profile)
            for trade in [*open_trades, *closed]:
                if trade.symbol != settings.market_symbol:
                    continue
                if await upsert_trade_event(session, leader.id, trade) is not None:
                    inserted += 1
            await session.commit()
    log.info("Leader poll finished: %d new %s events from %d live leaders",
             inserted, settings.market_symbol, len(leader_ids))
    return inserted


async def refresh_leader_roster(
    settings: Settings, myfxbook: MyfxbookClient, session_factory, apply: bool = True
) -> dict:
    """Two-tier discovery: bulk scan the watch list, score it, rotate to live.

    One cheap ``get-watched-accounts`` call replaces N per-account polls for
    the whole candidate pool. With ``apply=False`` scores are observed
    (persisted) but the tiers are left untouched — a dry run. Returns a
    summary of promotions/demotions.
    """
    roster = load_roster(settings)
    try:
        watched = await myfxbook.get_watched_accounts()
    except Exception as exc:
        log.warning("Watch-list scan failed: %s", exc)
        return {"error": str(exc), "watched": 0, "promoted": [], "demoted": []}

    roster.observe(watched)
    rotation = roster.rotate() if apply else None
    if rotation is None:
        return {
            "watched": len(watched),
            "live": roster.live_ids(),
            "promoted": [],
            "demoted": [],
            "note": "dry-run (add --apply to promote/demote)",
        }
    roster.save()
    for leader_id in rotation.promoted:
        # Backfill the leader profile so the API/DB know it even before the
        # first trade poll.
        async with session_factory() as session:
            await get_or_create_leader(session, LeaderProfile(id=leader_id, name=leader_id, audited=True))
            await session.commit()
    log.info("roster refreshed: watched=%d promoted=%s demoted=%s (%s)",
             len(watched), rotation.promoted, rotation.demoted, rotation.note)
    return {
        "watched": len(watched),
        "live": roster.live_ids(),
        "promoted": rotation.promoted,
        "demoted": rotation.demoted,
        "note": rotation.note,
    }


async def run_daily_leader_scoring(settings: Settings, session_factory, myfxbook: MyfxbookClient) -> None:
    """Daily profit-factor / win-rate / drawdown / consistency per live leader.

    Trade-history scoring is per-account, so it runs on the live tier only.
    """
    day = today_utc()
    for leader_id in live_tier(settings):
        try:
            closed = await myfxbook.get_history(leader_id)
        except Exception as exc:
            log.warning("Leader %s scoring failed: %s", leader_id, exc)
            continue
        gold_closed = [t for t in closed if t.symbol == settings.market_symbol and not t.is_open]
        score = compute_score(gold_closed)
        async with session_factory() as session:
            await store_daily_score(session, leader_id, day, score)
            await audit.record_audit(
                session, actor="scheduler", action="leader.score",
                detail={"leader_id": leader_id, "composite": score.composite,
                        "profit_factor": _finite(score.profit_factor), "win_rate": score.win_rate},
            )
            await session.commit()
    log.info("Daily leader scoring complete for %s", day)


def _finite(x: float) -> float:
    import math

    return x if math.isfinite(x) else 0.0