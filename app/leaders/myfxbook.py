"""Myfxbook API polling client (read-only).

Phase 1 uses this to poll curated, audited gold providers and normalise
their open/close events into ``TradeEvent``-shaped data. The client is
decoupled from the DB so it can also run in backtest/replay with fixture
data.

API surface used:
  * login.json                          -> session
  * get-history.json?id=<leaderId>      -> closed trades
  * get-open-trades.json?id=<leaderId>  -> currently open trades
"""
from __future__ import annotations

from datetime import datetime

import httpx

from ..logging import get_logger
from .models import RawLeaderTrade

log = get_logger(__name__)

BASE_URL = "https://www.myfxbook.com/api"


class MyfxbookClient:
    def __init__(
        self,
        email: str,
        password: str,
        base_url: str = BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        self._email = email
        self._password = password
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session_key: str | None = None

    async def _login(self, client: httpx.AsyncClient) -> str:
        if self._session_key:
            return self._session_key
        resp = await client.get(
            f"{self._base_url}/login.json",
            params={"email": self._email, "password": self._password},
        )
        data = resp.json()
        if not data.get("authenticated"):
            raise PermissionError(f"Myfxbook login failed: {data.get('error')}")
        self._session_key = data["session"]
        return self._session_key

    async def _get(self, endpoint: str, **params) -> list[dict]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            session = await self._login(client)
            resp = await client.get(
                f"{self._base_url}/{endpoint}",
                params={"session": session, **params},
            )
            data = resp.json()
            if not data.get("error"):
                return data.get("history", data.get("openTrades", []))
            log.warning("Myfxbook error on %s: %s", endpoint, data.get("error"))
            return []

    async def get_history(self, leader_id: str) -> list[RawLeaderTrade]:
        rows = await self._get("get-history.json", id=leader_id)
        return [self._normalise(r) for r in rows]

    async def get_open_trades(self, leader_id: str) -> list[RawLeaderTrade]:
        rows = await self._get("get-open-trades.json", id=leader_id)
        out = []
        for r in rows:
            t = self._normalise(r)
            t.is_open = True
            out.append(t)
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _normalise(fields: dict) -> RawLeaderTrade:
        side = str(fields.get("action", "")).lower()
        if side not in ("buy", "sell"):
            side = "buy" if float(fields.get("volume", 0) or 0) >= 0 else "sell"
        return RawLeaderTrade(
            external_id=str(fields.get("id", "")),
            symbol=str(fields.get("symbol", "XAUUSD")).upper(),
            side=side,
            volume=abs(float(fields.get("volume", 0) or 0)),
            open_price=float(fields.get("openPrice", 0) or 0),
            close_price=_opt_float(fields.get("closePrice")),
            profit=_opt_float(fields.get("profit")),
            lots=_opt_float(fields.get("lots")),
            pips=_opt_float(fields.get("pips")),
            opened_at=_parse_myfx_time(fields.get("openTime")),
            closed_at=_parse_myfx_time(fields.get("closeTime")),
            is_open=False,
            raw=fields,
        )


def _opt_float(v) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _parse_myfx_time(v) -> datetime | None:
    """Myfxbook returns epoch ms. Tolerates other formats."""
    if not v:
        return None
    try:
        return datetime.fromtimestamp(int(v) / 1000)
    except (TypeError, ValueError, OSError):
        return None