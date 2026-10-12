"""Execution client protocol + concrete implementations.

Same code path for every mode — only the credentials and the profile differ.
``PaperBroker`` must be the default for tests and replay.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..logging import get_logger
from .validator import OrderValidator, SymbolSpecification, ValidationResult

log = get_logger(__name__)


@dataclass
class OrderRequest:
    symbol: str
    side: str            # "buy" | "sell"
    volume: float
    entry_ref_price: float
    stop_loss: float | None = None
    take_profit: float | None = None
    client_order_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class OrderResult:
    ok: bool
    broker_order_id: str = ""
    fill_price: float = 0.0
    error: str = ""
    client_order_id: str = ""


class ExecutionClient(ABC):
    name: str = "abstract"

    def __init__(self, validator: OrderValidator | None = None) -> None:
        self.validator = validator or OrderValidator()
        self._processed_orders: dict[str, OrderResult] = {}

    def is_duplicate(self, client_order_id: str) -> bool:
        return bool(client_order_id and client_order_id in self._processed_orders)

    def get_processed_result(self, client_order_id: str) -> OrderResult | None:
        return self._processed_orders.get(client_order_id)

    def record_processed_result(self, client_order_id: str, result: OrderResult) -> None:
        if client_order_id:
            self._processed_orders[client_order_id] = result

    def validate_request(self, request: OrderRequest) -> ValidationResult:
        return self.validator.validate(request)

    @abstractmethod
    async def place_market_order(self, request: OrderRequest) -> OrderResult: ...

    async def execute_with_retry(
        self,
        request: OrderRequest,
        max_attempts: int = 3,
        initial_backoff: float = 0.5,
        backoff_multiplier: float = 2.0,
        timeout_seconds: float = 30.0,
    ) -> OrderResult:
        """Executes market order with exponential backoff and hard deadline timeout."""
        start_time = time.monotonic()
        last_error = ""

        # Check idempotency first
        if request.client_order_id and self.is_duplicate(request.client_order_id):
            cached = self.get_processed_result(request.client_order_id)
            log.warning("Duplicate order submission detected for client_order_id: %s. Returning existing result.", request.client_order_id)
            return cached or OrderResult(ok=False, error="Duplicate order rejected", client_order_id=request.client_order_id)

        # Validate order pre-flight
        val = self.validate_request(request)
        if not val.valid:
            res = OrderResult(ok=False, error=val.error, client_order_id=request.client_order_id)
            self.record_processed_result(request.client_order_id, res)
            return res

        for attempt in range(1, max_attempts + 1):
            elapsed = time.monotonic() - start_time
            if elapsed >= timeout_seconds:
                err = f"Execution hard timeout exceeded ({elapsed:.1f}s >= {timeout_seconds:.1f}s)"
                log.error("Order %s failed: %s", request.client_order_id, err)
                res = OrderResult(ok=False, error=err, client_order_id=request.client_order_id)
                self.record_processed_result(request.client_order_id, res)
                return res

            try:
                remaining_time = max(0.5, timeout_seconds - elapsed)
                result = await asyncio.wait_for(self.place_market_order(request), timeout=remaining_time)
                if result.ok:
                    result.client_order_id = request.client_order_id
                    self.record_processed_result(request.client_order_id, result)
                    return result

                last_error = result.error
                # If error is a permanent rejection (e.g. invalid params), do not retry
                if "invalid" in last_error.lower() or "rejected" in last_error.lower() or "validation" in last_error.lower():
                    result.client_order_id = request.client_order_id
                    self.record_processed_result(request.client_order_id, result)
                    return result

                log.warning("Transient order failure on attempt %d/%d for %s: %s", attempt, max_attempts, request.client_order_id, last_error)
            except asyncio.TimeoutError:
                last_error = f"Attempt {attempt} timed out"
                log.warning("Order attempt %d/%d timed out for %s", attempt, max_attempts, request.client_order_id)
            except Exception as exc:
                last_error = str(exc)
                log.warning("Order attempt %d/%d raised exception for %s: %s", attempt, max_attempts, request.client_order_id, exc)

            if attempt < max_attempts:
                sleep_duration = min(initial_backoff * (backoff_multiplier ** (attempt - 1)), 5.0)
                await asyncio.sleep(sleep_duration)

        res = OrderResult(ok=False, error=f"Execution failed after {max_attempts} attempts: {last_error}", client_order_id=request.client_order_id)
        self.record_processed_result(request.client_order_id, res)
        return res


class PaperBroker(ExecutionClient):
    """Fills immediately at the reference price with a tiny simulated slip."""

    name = "paper"

    def __init__(self, slippage_pct: float = 0.0002, rng_seed: int = 7, validator: OrderValidator | None = None) -> None:
        super().__init__(validator=validator)
        self.slippage_pct = slippage_pct
        self._rng = np.random.default_rng(rng_seed)

    async def place_market_order(self, request: OrderRequest) -> OrderResult:
        if not request.client_order_id:
            request.client_order_id = uuid.uuid4().hex

        # Pre-execution validation
        val = self.validate_request(request)
        if not val.valid:
            res = OrderResult(ok=False, error=val.error, client_order_id=request.client_order_id)
            self.record_processed_result(request.client_order_id, res)
            return res

        slip = request.entry_ref_price * self.slippage_pct * self._rng.normal()
        fill = request.entry_ref_price + (slip if request.side == "buy" else -slip)
        res = OrderResult(
            ok=True,
            broker_order_id=f"paper-{fill:.2f}",
            fill_price=float(fill),
            client_order_id=request.client_order_id,
        )
        self.record_processed_result(request.client_order_id, res)
        return res


class MetaApiClient(ExecutionClient):
    """Real MT5 placement through MetaApi (same cloud for demo/prop/live)."""

    name = "metaapi"

    def __init__(self, token: str, account_id: str, validator: OrderValidator | None = None, **create_options) -> None:
        super().__init__(validator=validator)
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
        if not request.client_order_id:
            request.client_order_id = uuid.uuid4().hex

        val = self.validate_request(request)
        if not val.valid:
            res = OrderResult(ok=False, error=val.error, client_order_id=request.client_order_id)
            self.record_processed_result(request.client_order_id, res)
            return res

        try:
            account = await self._connect()
            lot_side = request.volume if request.side == "buy" else -request.volume
            trade = await account.create_market_order(
                request.symbol,
                lot_side,
                request.stop_loss,
                request.take_profit,
                comment=f"aries-{request.client_order_id[:8]}",
            )
            res = OrderResult(
                ok=True,
                broker_order_id=str(trade.get("id", "")),
                fill_price=float(trade.get("price", 0.0)),
                client_order_id=request.client_order_id,
            )
            self.record_processed_result(request.client_order_id, res)
            return res
        except Exception as exc:  # metaapi raises typed SDK errors
            err = str(exc)
            res = OrderResult(ok=False, error=err, client_order_id=request.client_order_id)
            self.record_processed_result(request.client_order_id, res)
            return res