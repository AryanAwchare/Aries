"""Execution client protocol + concrete implementations.

Same code path for every mode — only the credentials and the profile differ.
``PaperBroker`` must be the default for tests and replay.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class OrderRequest:
    symbol: str
    side: str            # "buy" | "sell"
    volume: float
    entry_ref_price: float
    stop_loss: float | None = None
    take_profit: float | None = None


@dataclass
class OrderResult:
    ok: bool
    broker_order_id: str = ""
    fill_price: float = 0.0
    error: str = ""


class ExecutionClient(ABC):
    name: str = "abstract"

    @abstractmethod
    async def place_market_order(self, request: OrderRequest) -> OrderResult: ...


class PaperBroker(ExecutionClient):
    """Fills immediately at the reference price with a tiny simulated slip."""

    name = "paper"

    def __init__(self, slippage_pct: float = 0.0002, rng_seed: int = 7) -> None:
        self.slippage_pct = slippage_pct
        self._rng = np.random.default_rng(rng_seed)

    async def place_market_order(self, request: OrderRequest) -> OrderResult:
        slip = request.entry_ref_price * self.slippage_pct * self._rng.normal()
        fill = request.entry_ref_price + (slip if request.side == "buy" else -slip)
        return OrderResult(ok=True, broker_order_id=f"paper-{fill:.2f}", fill_price=float(fill))


class MetaApiClient(ExecutionClient):
    """Real MT5 placement through MetaApi (same cloud for demo/prop/live)."""

    name = "metaapi"

    def __init__(self, token: str, account_id: str, **create_options) -> None:
        self.token = token
        self.account_id = account_id
        self.create_options = create_options or {}
        self._connection = None

    async def _connect(self):
        if self._connection is None:
            from metaapi_cloud_sdk import MetaApi  # lazy import (heavy)

            self._api = MetaApi(self.token)
            self._connection = await self._api.connect()
            self._account = self._connection.get_account(self.account_id)
            await self._account.wait_synchronized()
        return self._account

    async def place_market_order(self, request: OrderRequest) -> OrderResult:
        account = await self._connect()
        lot_side = request.volume if request.side == "buy" else -request.volume
        try:
            trade = await account.create_market_order(
                request.symbol,
                lot_side,
                request.stop_loss,
                request.take_profit,
                comment="gold-hybrid-platform",
            )
            return OrderResult(
                ok=True,
                broker_order_id=str(trade.get("id", "")),
                fill_price=float(trade.get("price", 0.0)),
            )
        except Exception as exc:  # metaapi raises typed SDK errors
            return OrderResult(ok=False, error=str(exc))