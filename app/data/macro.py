"""Macroeconomic data provider: US Dollar Index (DXY) & 10-Year Treasury Yields (TNX).

Gold is fundamentally driven by:
1. Inverse correlation with the US Dollar Index (DXY): As USD rises, Gold falls.
2. Inverse correlation with Real Yields / 10-Year US Treasury Yields (^TNX):
   As bond yields rise, capital rotates out of zero-yielding gold into fixed income.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd

from ..logging import get_logger

log = get_logger(__name__)


@dataclass
class MacroContext:
    timestamp: datetime
    dxy_close: float
    dxy_ret_4: float
    us10y_yield: float
    us10y_yield_ret_4: float
    macro_bias: float  # [-1.0, 1.0]: positive = bullish for gold, negative = bearish for gold


class MacroProvider:
    """Fetches and caches DXY and US10Y data via yfinance for gold fundamental context."""

    def __init__(self, cache_dir: Path | str = "data/raw") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._dxy_df: pd.DataFrame | None = None
        self._tnx_df: pd.DataFrame | None = None

    def fetch_latest(self, period: str = "5d", interval: str = "15m") -> None:
        """Fetch latest intraday data for DXY and TNX."""
        try:
            import yfinance as yf

            dxy = yf.Ticker("DX-Y.NYB").history(period=period, interval=interval)
            tnx = yf.Ticker("^TNX").history(period=period, interval=interval)

            if not dxy.empty:
                dxy.index = dxy.index.tz_convert("UTC") if dxy.index.tz else dxy.index.tz_localize("UTC")
                self._dxy_df = dxy.rename(columns=str.lower)
            if not tnx.empty:
                tnx.index = tnx.index.tz_convert("UTC") if tnx.index.tz else tnx.index.tz_localize("UTC")
                self._tnx_df = tnx.rename(columns=str.lower)
            log.info("Macro data updated: DXY bars=%d, TNX bars=%d", len(dxy), len(tnx))
        except Exception as exc:
            log.warning("Could not fetch live macro data (%s) — using neutral macro context", exc)

    def get_context(self, dt: datetime | None = None) -> MacroContext:
        """Calculate stationary macro features for a specific bar timestamp."""
        t = dt or datetime.now(timezone.utc)
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)

        # Default fallback
        dxy_close = 104.0
        dxy_ret_4 = 0.0
        us10y_yield = 4.25
        us10y_ret_4 = 0.0

        if self._dxy_df is not None and not self._dxy_df.empty:
            sub = self._dxy_df.loc[self._dxy_df.index <= t]
            if len(sub) >= 5:
                dxy_close = float(sub["close"].iloc[-1])
                prev_dxy = float(sub["close"].iloc[-5])
                dxy_ret_4 = (dxy_close - prev_dxy) / prev_dxy if prev_dxy > 0 else 0.0

        if self._tnx_df is not None and not self._tnx_df.empty:
            sub = self._tnx_df.loc[self._tnx_df.index <= t]
            if len(sub) >= 5:
                us10y_yield = float(sub["close"].iloc[-1])
                prev_tnx = float(sub["close"].iloc[-5])
                us10y_ret_4 = (us10y_yield - prev_tnx) / prev_tnx if prev_tnx > 0 else 0.0

        # Negative correlation: when DXY and 10Y drop, gold bias is positive (bullish)
        # Macro bias: positive when both are falling, negative when both are rising
        macro_bias = float(-1.0 * (dxy_ret_4 * 50.0 + us10y_ret_4 * 20.0))
        macro_bias = max(-1.0, min(1.0, macro_bias))

        return MacroContext(
            timestamp=t,
            dxy_close=dxy_close,
            dxy_ret_4=round(dxy_ret_4, 5),
            us10y_yield=round(us10y_yield, 3),
            us10y_yield_ret_4=round(us10y_ret_4, 5),
            macro_bias=round(macro_bias, 3),
        )
