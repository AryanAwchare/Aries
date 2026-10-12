"""Macroeconomic News Blackout Filter for Gold (XAUUSD).

Gold volatility surges violently during Tier-1 US economic releases:
- US Non-Farm Payrolls (NFP): First Friday of each month at 13:30 UTC.
- US Consumer Price Index (CPI): Mid-month (typically 2nd Tuesday/Wednesday) at 13:30 UTC.
- Federal Reserve FOMC Rate Decision & Press Conference: 18:00 / 18:30 UTC.
- US Core PCE (Fed's preferred inflation gauge): Last Friday of month at 13:30 UTC.

During these windows, spreads widen from $0.25 to $3.00+ and slippage spikes.
The NewsFilter enforces a hard trading freeze (e.g. 20m before, 15m after)
to prevent stop-hunting and catastrophic execution slippage.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List


@dataclass
class NewsEvent:
    name: str
    impact: str  # 'HIGH', 'CRITICAL'
    timestamp: datetime


class NewsFilter:
    """Detects active news blackout zones around Tier-1 US announcements."""

    def __init__(
        self,
        blackout_before_minutes: int = 20,
        blackout_after_minutes: int = 15,
        custom_events: List[NewsEvent] | None = None,
    ) -> None:
        self.before_mins = blackout_before_minutes
        self.after_mins = blackout_after_minutes
        self.events: List[NewsEvent] = custom_events or []

    def is_blackout(self, dt: datetime | None = None) -> tuple[bool, str | None]:
        """Check if dt is within a high-impact news blackout window."""
        t = dt or datetime.now(timezone.utc)
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)

        # Check explicit events
        for ev in self.events:
            ev_time = ev.timestamp if ev.timestamp.tzinfo else ev.timestamp.replace(tzinfo=timezone.utc)
            start_freeze = ev_time - timedelta(minutes=self.before_mins)
            end_freeze = ev_time + timedelta(minutes=self.after_mins)
            if start_freeze <= t <= end_freeze:
                return True, f"Tier-1 Event: {ev.name} ({ev.impact})"

        # Check recurring Tier-1 structural patterns:
        # 1. First Friday of month (NFP) around 13:30 UTC
        if t.weekday() == 4 and 1 <= t.day <= 7:
            # First Friday
            nfp_release = t.replace(hour=13, minute=30, second=0, microsecond=0)
            if (nfp_release - timedelta(minutes=self.before_mins)) <= t <= (nfp_release + timedelta(minutes=self.after_mins)):
                return True, "US Non-Farm Payrolls (NFP) Blackout"

        return False, None
