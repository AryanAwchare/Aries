"""Stage C: Leader Behavior Classifier.

Classifies each leader into one of:
- trend_following
- mean_reversion
- breakout
- scalping
- swing
- grid
- martingale_like
- news_sensitive
- high_frequency

Based on holding time distribution, position sizing after losses,
trade frequency, and entry clustering around economic news.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Sequence

import numpy as np

from .models import RawLeaderTrade


def classify_behavior(trades: Sequence[RawLeaderTrade]) -> str:
    """Classify leader trading style from historical trade records."""
    closed = [t for t in trades if not t.is_open and t.opened_at is not None and t.closed_at is not None]
    if not closed:
        return "trend_following"  # Default assumption

    # 1. Check for toxic behaviors first: martingale and grid
    from .eligibility import detect_grid, detect_martingale

    if detect_martingale(trades):
        return "martingale_like"
    if detect_grid(trades):
        return "grid"

    # 2. Analyze holding durations
    durations_seconds = []
    news_hour_entries = 0  # 12:00-15:00 UTC (US/EU major news releases)
    for t in closed:
        if t.opened_at and t.closed_at:
            dur = (t.closed_at - t.opened_at).total_seconds()
            durations_seconds.append(max(0.0, dur))
        if t.opened_at:
            h = t.opened_at.hour
            if h in (12, 13, 14, 15):
                news_hour_entries += 1

    dur_arr = np.array(durations_seconds) if durations_seconds else np.array([3600.0])
    median_duration = float(np.median(dur_arr))
    mean_duration = float(np.mean(dur_arr))

    # 3. High Frequency (< 5 min median duration or > 50 trades/day)
    if median_duration < 300:
        return "high_frequency"

    # 4. Scalping (< 30 min median holding duration)
    if median_duration < 1800:
        return "scalping"

    # 5. Swing (> 24 hours median duration)
    if median_duration > 86400:
        return "swing"

    # 6. News Sensitive (> 65% entries during news hours)
    if len(closed) > 10 and (news_hour_entries / len(closed)) > 0.65:
        return "news_sensitive"

    # 7. Holding between 30m and 24h: analyze profit skew & win rate
    profits = np.array([t.profit or 0.0 for t in closed])
    wins = profits[profits > 0]
    losses = profits[profits < 0]
    win_rate = len(wins) / len(profits) if len(profits) > 0 else 0.5
    avg_win = float(wins.mean()) if len(wins) else 1.0
    avg_loss = float(abs(losses.mean())) if len(losses) else 1.0
    payoff_ratio = avg_win / max(avg_loss, 1e-4)

    # High win rate, low payoff ratio -> Mean reversion
    if win_rate > 0.65 and payoff_ratio < 1.0:
        return "mean_reversion"

    # Low win rate, high payoff ratio -> Breakout
    if win_rate < 0.45 and payoff_ratio > 1.8:
        return "breakout"

    # Default robust classification
    return "trend_following"