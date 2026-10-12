"""Leader polling + two-tier roster refresh with multi-stage intelligence.

Orchestrates:
- Stage A: Eligibility Gate
- Stage B: Composite Scoring
- Stage C: Behavior Classification
- Stage D: Pairwise Correlation
- Stage E: Watch / Live Tier Management
- Stage F: Real-time Circuit Breaker
"""
from __future__ import annotations

from ..config import Settings
from ..db import audit
from ..logging import get_logger
from .circuit_breaker import LeaderLiveCircuitBreaker
from .classifier import classify_behavior
from .correlation import LeaderCorrelationEngine
from .discovery import LeaderRoster
from .eligibility import LeaderEligibilityGate
from .models import LeaderProfile, RawLeaderTrade
from .myfxbook import MyfxbookClient
from .repo import get_or_create_leader, store_daily_score, upsert_trade_event
from .scoring import compute_score, today_utc

log = get_logger(__name__)


def load_roster(settings: Settings) -> LeaderRoster:
    return LeaderRoster(
        settings.roster_path,
        max_live=settings.live_leader_pool_size or settings.max_live_leaders,
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
    """Poll the **live tier only** for new XAUUSD trades, persist + evaluate circuit breaker."""
    roster = load_roster(settings)
    circuit_breaker = LeaderLiveCircuitBreaker(roster, settings)
    leader_ids = roster.live_ids() or settings.leader_ids
    inserted = 0

    for leader_id in leader_ids:
        try:
            open_trades = await myfxbook.get_open_trades(leader_id)
            closed = await myfxbook.get_history(leader_id)
        except Exception as exc:  # network / API hiccup
            log.warning("Leader %s poll failed: %s", leader_id, exc)
            continue

        profile = LeaderProfile(
            id=leader_id,
            name=f"leader-{leader_id}",
            url="",
            audited=True,
            verified_track_record=True,
            verified_trading_privileges=True,
            status="live",
        )

        async with session_factory() as session:
            leader = await get_or_create_leader(session, profile)
            all_trades = [*open_trades, *closed]

            # Classify behavior (Stage C)
            if closed:
                b_class = classify_behavior(closed)
                leader.strategy_type = b_class

            for trade in all_trades:
                if trade.symbol != settings.market_symbol:
                    continue
                ev = await upsert_trade_event(session, leader.id, trade)
                if ev is not None:
                    inserted += 1
                    # Evaluate Stage F Circuit Breaker on new closed trade
                    if not trade.is_open and trade.profit is not None:
                        await circuit_breaker.update_trade_and_check(session, leader_id, trade)

            await session.commit()

    log.info("Leader poll finished: %d new %s events from %d live leaders",
             inserted, settings.market_symbol, len(leader_ids))
    return inserted


async def refresh_leader_roster(
    settings: Settings, myfxbook: MyfxbookClient, session_factory, apply: bool = True
) -> dict:
    """Two-tier discovery & multi-stage rescore.

    Runs on LEADER_RESCORE_INTERVAL_HOURS schedule.
    1. Bulk scans the watch list via get_watched_accounts()
    2. Runs Stage A eligibility filtering
    3. Runs Stage B composite scoring
    4. Rotates top N eligible performers to live tier
    """
    roster = load_roster(settings)
    eligibility_gate = LeaderEligibilityGate(settings)

    try:
        watched = await myfxbook.get_watched_accounts()
    except Exception as exc:
        log.warning("Watch-list scan failed: %s", exc)
        return {"error": str(exc), "watched": 0, "promoted": [], "demoted": []}

    roster.observe(watched)

    # For candidates with sufficient observations, check Stage A eligibility
    for leader_id in roster.all_ids():
        cand = roster.get(leader_id)
        if not cand or cand.observations < settings.leader_min_observations:
            continue

        # In live operation, we fetch history to verify stage A eligibility & score
        try:
            history = await myfxbook.get_history(leader_id)
        except Exception:
            history = []

        prof = LeaderProfile(
            id=leader_id,
            name=cand.name,
            account_age=cand.observations * 30,  # approximate from observations if not in meta
            verified_track_record=cand.verified_track_record,
            verified_trading_privileges=cand.verified_trading_privileges,
        )

        if history:
            res = eligibility_gate.evaluate(prof, history)
            cand.is_eligible = res.eligible
            if res.eligible:
                score = compute_score(history, prof, settings)
                cand.last_composite = score.composite_score
                cand.strategy_type = classify_behavior(history)
            else:
                cand.last_composite = 0.0
        else:
            # Fall back to aggregate watch score if history call was empty
            cand.is_eligible = cand.verified_track_record and (cand.watch_score > 0.3)

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
        cand = roster.get(leader_id)
        async with session_factory() as session:
            await get_or_create_leader(
                session,
                LeaderProfile(
                    id=leader_id,
                    name=cand.name if cand else leader_id,
                    audited=True,
                    verified_track_record=cand.verified_track_record if cand else True,
                    status="live",
                ),
            )
            await audit.record_audit(
                session,
                actor="roster",
                action="leader.promoted_to_live",
                detail={"leader_id": leader_id, "score": cand.last_composite if cand else 0.0},
            )
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
    """Scheduled composite leader scoring."""
    day = today_utc()
    for leader_id in live_tier(settings):
        try:
            closed = await myfxbook.get_history(leader_id)
        except Exception as exc:
            log.warning("Leader %s scoring failed: %s", leader_id, exc)
            continue
        gold_closed = [t for t in closed if t.symbol == settings.market_symbol and not t.is_open]
        score = compute_score(gold_closed, settings=settings)
        async with session_factory() as session:
            await store_daily_score(session, leader_id, day, score)
            await audit.record_audit(
                session, actor="scheduler", action="leader.score",
                detail={
                    "leader_id": leader_id,
                    "composite": score.composite,
                    "profit_factor": _finite(score.profit_factor),
                    "win_rate": score.win_rate,
                    "expectancy": score.expectancy,
                    "longevity": score.longevity,
                },
            )
            await session.commit()
    log.info("Multi-stage leader scoring complete for %s", day)


def _finite(x: float) -> float:
    import math
    return x if math.isfinite(x) else 0.0