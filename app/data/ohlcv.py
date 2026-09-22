"""OHLCV data providers.

``get_ohlcv(symbol, timeframe, start, end)`` returns a pandas DataFrame
indexed by UTC timestamp with columns ``open, high, low, close, volume``.
Providers:

* ``UVProvider``/``MetaApiProvider`` — live MT5 bars via MetaApi (async).
* ``CSVProvider`` — deterministic local files, the default for backtest/replay.
* ``SyntheticProvider`` — fixed-seed random walk, for tests/demos only.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from ..logging import get_logger

log = get_logger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass
class OHLCVRequest:
    symbol: str
    timeframe: str        # "M15", "H1", "D1", ...
    start: datetime
    end: datetime
    path: str | None = None  # resolved later by providers that need it


class OHLCVProvider(ABC):
    @abstractmethod
    async def get_ohlcv(self, request: OHLCVRequest) -> pd.DataFrame: ...


# ---------------------------------------------------------------------------
# CSV provider — deterministic, used by backtest/replay
# ---------------------------------------------------------------------------
class CSVProvider(OHLCVProvider):
    """Reads bars from a CSV file with a UTC ``timestamp`` column."""

    def __init__(self, directory: str) -> None:
        self.directory = directory

    async def get_ohlcv(self, request: OHLCVRequest) -> pd.DataFrame:
        import aiofiles  # noqa: F401  (optional accelerator; falls back to sync read)

        # NOTE: pyarrow-backed feather is preferable for big files; keep CSV
        # simple and readable for now.
        path = request.path or f"{self.directory}/{request.symbol}_{request.timeframe}.csv"
        df = pd.read_csv(path, parse_dates=["timestamp"])
        df = df.set_index("timestamp").tz_localize("UTC")
        mask = (df.index >= request.start) & (df.index <= request.end)
        return df.loc[mask][OHLCV_COLUMNS]

    async def write_jsonl(self) -> None:  # pragma: no cover - placeholder
        raise NotImplementedError


# ---------------------------------------------------------------------------
# MetaApi provider — live MT5 bars over the same SDK as execution
# ---------------------------------------------------------------------------
class MetaApiProvider(OHLCVProvider):
    """Fetches bars from a MetaApi cloud account."""

    def __init__(self, token: str, account_id: str) -> None:
        self.token = token
        self.account_id = account_id
        self._connection = None

    async def _connect(self):
        if self._connection is None:
            from metaapi_cloud_sdk import MetaApi  # lazy import (heavy)

            self._api = MetaApi(self.token)
            self._connection = await self._api.connect()
            self._account = self._connection.get_account(self.account_id)
            await self._account.wait_synchronized()
        return self._account

    async def get_ohlcv(self, request: OHLCVRequest) -> pd.DataFrame:
        account = await self._connect()
        market = await account.get_market_data()
        # MetaApi get_historical_candles returns a list of dicts; mapping to a
        # DataFrame keeps the same shape as the CSV provider.
        candles = await market.get_historical_candles(
            request.symbol,
            request.timeframe,
            start_time=request.start,
            end_time=request.end,
        )
        rows = []
        for c in candles:
            rows.append(
                {
                    "timestamp": pd.Timestamp(c["time"], unit="ms", tz="UTC"),
                    "open": c["open"],
                    "high": c["high"],
                    "low": c["low"],
                    "close": c["close"],
                    "volume": c.get("tickVolume", c.get("volume", 0)),
                }
            )
        df = pd.DataFrame(rows).set_index("timestamp")
        return df[OHLCV_COLUMNS]


# ---------------------------------------------------------------------------
# Synthetic provider — deterministic random walk (CI / demos / coverage)
# ---------------------------------------------------------------------------
@dataclass
class SyntheticProvider(OHLCVProvider):
    seed: int = 42
    start_price: float = 2000.0
    bar_pct: float = 0.002
    _cache: dict = field(default_factory=dict)

    async def get_ohlcv(self, request: OHLCVRequest) -> pd.DataFrame:
        key = (request.symbol, request.timeframe, request.start, request.end)
        if key in self._cache:
            return self._cache[key]

        rng = __import__("numpy").random.default_rng(self.seed)
        duration = (request.end - request.start).total_seconds()
        step = self._step_seconds(request.timeframe)
        n = max(int(duration / step), 1)
        # deterministic, but varies per symbol/timeframe hashed into seed:
        rng = __import__("numpy").random.default_rng(self.seed + hash(request.symbol) % 10_000)

        drift = rng.normal(0, self.bar_pct, n).cumsum() + rng.normal(0, self.bar_pct * 0.1)
        close = self.start_price * (1 + drift)
        open_ = close * (1 + rng.normal(0, self.bar_pct * 0.2, n))
        high = pd.Series(close).rolling(2, min_periods=1).max().to_numpy() * (
            1 + rng.uniform(0, self.bar_pct * 0.3, n)
        )
        low = pd.Series(close).rolling(2, min_periods=1).min().to_numpy() * (
            1 - rng.uniform(0, self.bar_pct * 0.3, n)
        )
        idx = pd.date_range(request.start, periods=n, freq=self._freq(request.timeframe), tz="UTC")
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": rng.integers(10, 100, n)},
            index=idx,
        )
        self._cache[key] = df
        return df

    @staticmethod
    def _step_seconds(tf: str) -> float:
        return {"M15": 15 * 60, "H1": 3600, "D1": 86_400}.get(tf, 60 * 60)

    @staticmethod
    def _freq(tf: str) -> str:
        return {"M15": "15min", "H1": "h", "D1": "D"}.get(tf, "h")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)