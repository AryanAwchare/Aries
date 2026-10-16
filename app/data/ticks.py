"""Tick data ingestion + microstructure aggregation (Phase 1.2).

Ticks are the rawest market feed and the only way to compute true
microstructure features — bid/ask spread, order-flow imbalance, trade
intensity, price impact. Two sources:

* :func:`fetch_ticks_mt5` — the local MetaTrader 5 terminal via the
  ``MetaTrader5`` package (Windows only, requires a running/logged-in
  terminal). Uses ``copy_ticks_range`` (``COPY_TICKS_ALL``).
* :func:`load_ticks_csv` / :func:`save_ticks` — archived ticks on disk.

:func:`ticks_to_bars` aggregates ticks to OHLCV **plus** per-bar
microstructure columns, and :func:`tick_microstructure_features` adds rolling
estimators (Roll effective spread, Kyle lambda, OFI, VPIN proxy).

Note: many MetaQuotes-Demo / CFD feeds report ``volume == 0`` for gold. Every
estimator here therefore has a **count-based** fallback and never divides by
volume, so it works on volume-less feeds.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..logging import get_logger

log = get_logger(__name__)

# Fields returned by MT5 COPY_TICKS_ALL.
TICK_COLUMNS = ["time", "bid", "ask", "last", "volume", "time_msc", "flags", "volume_real"]

_TF_FREQ = {"M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min", "H1": "1h", "H4": "4h", "D1": "1D"}


def _freq(timeframe: str) -> str:
    return _TF_FREQ.get((timeframe or "M1").upper(), "1min")


def mt5_available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("MetaTrader5") is not None
    except Exception:  # pragma: no cover
        return False


# ---------------------------------------------------------------------------
# MT5 live tick fetch
# ---------------------------------------------------------------------------
def fetch_ticks_mt5(
    symbol: str,
    start: datetime,
    end: datetime,
    *,
    terminal_path: str | None = None,
    timeout_ms: int = 20000,
    shutdown: bool = True,
) -> pd.DataFrame:
    """Download ticks for ``symbol`` between ``start`` and ``end`` (UTC).

    Raises ``RuntimeError`` if the terminal cannot be reached or the copy
    call returns ``None``.
    """
    import MetaTrader5 as mt5  # Windows-only; imported lazily

    kwargs: dict = {"timeout": timeout_ms}
    if terminal_path:
        kwargs["path"] = terminal_path
    if not mt5.initialize(**kwargs):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        ticks = mt5.copy_ticks_range(symbol, start, end, mt5.COPY_TICKS_ALL)
        if ticks is None:
            raise RuntimeError(f"copy_ticks_range({symbol}) returned None: {mt5.last_error()}")
        df = pd.DataFrame(ticks)
    finally:
        if shutdown:
            mt5.shutdown()
    if df.empty:
        return pd.DataFrame(columns=TICK_COLUMNS)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df


# ---------------------------------------------------------------------------
# Disk I/O
# ---------------------------------------------------------------------------
def save_ticks(ticks: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = ticks.copy()
    if "time" in out.columns and pd.api.types.is_datetime64_any_dtype(out["time"]):
        out["time"] = out["time"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    out.to_csv(path, index=False)
    log.info("saved %d ticks -> %s", len(out), path)
    return path


def load_ticks_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


# ---------------------------------------------------------------------------
# Aggregation: ticks -> bars + microstructure
# ---------------------------------------------------------------------------
def _tick_rule_sign(mid: pd.Series) -> pd.Series:
    """+1 uptick, -1 downtick, 0 unchanged (forward-filled, then 0)."""
    diff = mid.diff()
    sign = np.sign(diff)
    return pd.Series(sign, index=mid.index).replace(0.0, np.nan).ffill().fillna(0.0)


def ticks_to_bars(ticks: pd.DataFrame, timeframe: str = "M1") -> pd.DataFrame:
    """Aggregate ticks into OHLCV + per-bar microstructure columns.

    OHLCV is derived from the **mid** price ``(bid+ask)/2``; ``volume`` is the
    tick count (a robust proxy when the feed reports zero volume).
    """
    if ticks is None or ticks.empty:
        return pd.DataFrame()

    df = ticks.copy()
    if "time" not in df.columns or not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").set_index("time")
    has_quotes = "bid" in df.columns and "ask" in df.columns
    mid = (df["bid"] + df["ask"]) / 2.0 if has_quotes else df.get("last", df.iloc[:, 0])
    df["mid"] = mid
    df["_sign"] = _tick_rule_sign(df["mid"])
    if has_quotes:
        df["_spread"] = (df["ask"] - df["bid"]).clip(lower=0.0)
        df["_spread_pct"] = df["_spread"] / df["mid"].replace(0, np.nan)

    freq = _freq(timeframe)
    g = df.resample(freq)
    bars = pd.DataFrame(
        {
            "open": g["mid"].first(),
            "high": g["mid"].max(),
            "low": g["mid"].min(),
            "close": g["mid"].last(),
            "volume": g["mid"].count().astype(float),
        }
    )
    bars["tick_count"] = g["mid"].count().astype(float)
    bars["buy_ticks"] = g["_sign"].apply(lambda s: float((s > 0).sum()))
    bars["sell_ticks"] = g["_sign"].apply(lambda s: float((s < 0).sum()))
    denom = (bars["buy_ticks"] + bars["sell_ticks"]).replace(0, np.nan)
    bars["ofi"] = ((bars["buy_ticks"] - bars["sell_ticks"]) / denom).fillna(0.0)
    # Realised vol within the bar (mid-return std), scaled to bar length.
    bars["bar_realized_vol"] = g["mid"].apply(lambda s: float(s.pct_change().std()) if len(s) > 2 else 0.0)
    if has_quotes:
        bars["spread_mean"] = g["_spread"].mean()
        bars["spread_pct"] = g["_spread_pct"].mean()
    else:
        bars["spread_mean"] = 0.0
        bars["spread_pct"] = 0.0

    return bars.dropna(subset=["open", "close"]).reset_index().rename(columns={"time": "timestamp"}).set_index("timestamp")


def tick_microstructure_features(bars: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """Add rolling microstructure estimators to a tick-derived bar frame.

    * ``vpin``            — |OFI| rolled (count-based VPIN proxy).
    * ``roll_spread``     — Roll (1984) effective spread from return autocov.
    * ``kyle_lambda``     — price impact: rolling slope of mid-return on OFI.
    * ``intensity_z``     — z-score of tick count (trade intensity).
    * ``spread_z``        — z-score of relative spread (liquidity stress).
    """
    out = bars.copy()
    if out.empty:
        return out

    ofi = out.get("ofi", pd.Series(0.0, index=out.index)).astype(float)
    out["vpin"] = ofi.abs().rolling(window, min_periods=max(5, window // 5)).mean().fillna(0.0)

    # Roll effective spread: 2*sqrt(max(0, -cov(r_t, r_{t-1})))
    ret = np.log(out["close"].replace(0, np.nan)).diff()
    cov = ret.rolling(window, min_periods=max(5, window // 5)).cov(ret.shift(1)).fillna(0.0)
    out["roll_spread"] = 2.0 * np.sqrt(np.clip(-cov, 0.0, None))

    # Kyle lambda proxy: slope of return on signed order flow (rolling).
    x = ofi.rolling(window, min_periods=max(5, window // 5)).mean()
    xy = (ret.fillna(0.0) * ofi).rolling(window, min_periods=max(5, window // 5)).mean()
    xx = (ofi * ofi).rolling(window, min_periods=max(5, window // 5)).mean()
    out["kyle_lambda"] = (xy - ret.fillna(0.0).rolling(window, min_periods=max(5, window // 5)).mean() * x) / (
        xx - x * x
    ).replace(0, np.nan)
    out["kyle_lambda"] = out["kyle_lambda"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _z(s: pd.Series) -> pd.Series:
        m = s.rolling(window, min_periods=max(5, window // 5)).mean()
        sd = s.rolling(window, min_periods=max(5, window // 5)).std().replace(0, np.nan)
        return ((s - m) / sd).fillna(0.0)

    out["intensity_z"] = _z(out.get("tick_count", pd.Series(1.0, index=out.index)).astype(float))
    out["spread_z"] = _z(out.get("spread_pct", pd.Series(0.0, index=out.index)).astype(float))
    return out


@dataclass
class TickCoverage:
    symbol: str
    start: datetime | None
    end: datetime | None
    ticks: int

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "start": None if self.start is None else self.start.isoformat(),
            "end": None if self.end is None else self.end.isoformat(),
            "ticks": self.ticks,
        }


def tick_coverage(ticks: pd.DataFrame, symbol: str = "") -> TickCoverage:
    if ticks is None or ticks.empty:
        return TickCoverage(symbol, None, None, 0)
    t = pd.to_datetime(ticks["time"], utc=True)
    return TickCoverage(symbol, t.min().to_pydatetime(), t.max().to_pydatetime(), len(ticks))


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
