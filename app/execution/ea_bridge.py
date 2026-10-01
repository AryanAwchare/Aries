"""MT5/MT4 bridge: the local terminal's EA polls the platform over HTTP.

Why this exists: the platform must not depend on a geo-blocked cloud SDK
(MetaApi). Instead the MetaTrader terminal runs on the user's own machine (or
VPS) with a small Expert Advisor that:

  1. GET  /ea/next-order  ->  next order to execute (or NONE)
  2. runs it in MT5        ->  OrderSend buy/sell volume lots
  3. POST /ea/execution    ->  reports the fill back to the platform

The protocol is deliberately plain text (pipe-delimited) so the .mq5 EA needs
no JSON parser. HTTP on 127.0.0.1 is allowed by MetaTrader's WebRequest
without a security cache; an ``ea_api_key`` can still be required on the
platform side (passed as a query string constant by the EA).

``EaBroker`` implements ``ExecutionClient`` so the strategy executor can place
orders exactly as it does with PaperBroker/MetaApiClient — `place_market_order`
writes an ``EaOrder`` row and waits (up to ``EA_ORDER_TIMEOUT_SECONDS``) for the
terminal to report a fill.
"""
from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, Query, Request, Response
from sqlalchemy import select, update

from ..config import Settings
from ..db.models import EaOrder, TradeSide
from .base import ExecutionClient, OrderRequest, OrderResult

router = APIRouter(prefix="/ea", tags=["ea-bridge"])


# ---------------------------------------------------------------- client
class EaBroker(ExecutionClient):
    """Places orders through the local MetaTrader terminal."""

    name = "ea"

    def __init__(self, settings: Settings, session_factory) -> None:
        self.settings = settings
        self.session_factory = session_factory

    async def place_market_order(self, request: OrderRequest) -> OrderResult:
        external_id = uuid.uuid4().hex
        async with self.session_factory() as session:
            row = EaOrder(
                external_id=external_id,
                symbol=request.symbol,
                side=TradeSide(request.side),
                volume=request.volume,
                stop_loss=request.stop_loss,
                take_profit=request.take_profit,
            )
            session.add(row)
            await session.commit()

        deadline = asyncio.get_running_loop().time() + self.settings.ea_order_timeout_seconds
        while asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(1.0)
            async with self.session_factory() as session:
                row = (await session.execute(
                    select(EaOrder).where(EaOrder.external_id == external_id)
                )).scalar_one_or_none()
                if row is None:
                    continue
                if row.status == "filled":
                    return OrderResult(ok=True, broker_order_id=row.broker_order_id or "",
                                       fill_price=row.fill_price or 0.0)
                if row.status == "rejected":
                    return OrderResult(ok=False, error=row.error)
        return OrderResult(ok=False, error=f"ea timeout after {self.settings.ea_order_timeout_seconds}s")


# ---------------------------------------------------------------- helpers
def _auth_ok(settings: Settings, key: str | None) -> bool:
    return not settings.ea_api_key or key == settings.ea_api_key


# ---------------------------------------------------------------- endpoints
def _text(value: str) -> Response:
    return Response(content=value, media_type="text/plain")


@router.get("/ping")
async def ea_ping(request: Request):
    settings = request.app.state.settings
    return {"ok": True, "mode": settings.trading_mode.value,
            "driver": settings.execution_driver, "key_required": bool(settings.ea_api_key)}


@router.post("/test-order")
async def test_order(request: Request, key: str | None = Query(default=None)) -> Response:
    """Queue a minimal 0.01-lot market order to verify the EA link end-to-end.

    The EA should claim it within ~1s and open 0.01 lots in the terminal.
    Check the result with GET /ea/order/<external_id>.
    """
    settings: Settings = request.app.state.settings
    factory = request.app.state.db.session_factory
    if not _auth_ok(settings, key):
        return _text("AUTHFAIL")
    external = uuid.uuid4().hex
    async with factory() as session:
        session.add(EaOrder(
            external_id=external, symbol=settings.market_symbol,
            side=TradeSide.BUY, volume=0.01,
        ))
        await session.commit()
    return _text(f"QUEUED|{external}")


@router.get("/order/{external_id}")
async def order_status(external_id: str, request: Request, key: str | None = Query(default=None)) -> Response:
    """Plain-text status of one bridge order: STATUS|<status>|<fill>|<error>."""
    settings: Settings = request.app.state.settings
    factory = request.app.state.db.session_factory
    if not _auth_ok(settings, key):
        return _text("AUTHFAIL")
    async with factory() as session:
        row = (await session.execute(
            select(EaOrder).where(EaOrder.external_id == external_id)
        )).scalar_one_or_none()
    if row is None:
        return _text("ERR:unknown-order")
    fill = "" if row.fill_price is None else f"{row.fill_price:.6g}"
    return _text(f"STATUS|{row.status}|{fill}|{row.error}")


@router.get("/next-order")
async def next_order(request: Request, key: str | None = Query(default=None)) -> Response:
    """Claim the oldest pending order for the terminal to execute.

    Returns pipe-delimited plain text so the EA can parse it without a JSON
    library:
        ORDER|<external_id>|<symbol>|<side>|<volume>|<stop_loss>|<take_profit>
        NONE
    """
    settings: Settings = request.app.state.settings
    factory = request.app.state.db.session_factory
    if not _auth_ok(settings, key):
        return _text("AUTHFAIL")
    import datetime as _dt

    async with factory() as session:
        row = (await session.execute(
            select(EaOrder).where(EaOrder.status == "pending")
            .order_by(EaOrder.created_at).limit(1)
        )).scalar_one_or_none()
        if row is None:
            return _text("NONE")
        claimed = await session.execute(
            update(EaOrder)
            .where(EaOrder.id == row.id, EaOrder.status == "pending")
            .values(status="dispatched", dispatched_at=_dt.datetime.now(_dt.timezone.utc))
        )
        if claimed.rowcount != 1:
            return _text("NONE")
        await session.commit()
        sl = "" if row.stop_loss is None else f"{row.stop_loss:.6g}"
        tp = "" if row.take_profit is None else f"{row.take_profit:.6g}"
        return _text(f"ORDER|{row.external_id}|{row.symbol}|{row.side.value}|{row.volume:.6g}|{sl}|{tp}")


@router.post("/execution")
async def report_execution(request: Request, key: str | None = Query(default=None)) -> Response:
    """Receive the terminal's result for a previously dispatched order.

    Plain-text body (POST):
        RESULT|<external_id>|1|<broker_order_id>|<fill_price>
        RESULT|<external_id>|0|||<error message>
    Also accepts JSON:
        {"external_id": "...", "ok": true, "broker_order_id": "..", "fill_price": 1.0}
    """
    settings: Settings = request.app.state.settings
    factory = request.app.state.db.session_factory
    if not _auth_ok(settings, key):
        return _text("AUTHFAIL")

    raw = (await request.body()).decode("utf-8", errors="replace").strip()
    import datetime as _dt

    if raw.startswith("{"):
        import json as _json
        payload = _json.loads(raw)
        external_id = str(payload.get("external_id", ""))
        ok = bool(payload.get("ok"))
        broker_id = str(payload.get("broker_order_id", ""))
        fill = payload.get("fill_price")
        error = str(payload.get("error", ""))
    else:
        parts = raw.split("|")
        if len(parts) < 5 or parts[0] != "RESULT":
            return _text("ERR:bad-format")
        external_id = parts[1]
        ok = parts[2] == "1"
        broker_id = parts[3]
        fill = float(parts[4]) if parts[4] else None
        error = "|".join(parts[5:]) if not ok else ""

    async with factory() as session:
        row = (await session.execute(
            select(EaOrder).where(EaOrder.external_id == external_id)
        )).scalar_one_or_none()
        if row is None:
            return _text("ERR:unknown-order")
        if ok:
            row.status = "filled"
            row.broker_order_id = broker_id
            row.fill_price = fill
        else:
            row.status = "rejected"
            row.error = error
        row.updated_at = _dt.datetime.now(_dt.timezone.utc)
        await session.commit()
    return _text("OK")