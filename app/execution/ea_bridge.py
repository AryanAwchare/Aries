"""Local MetaTrader 5 execution bridge.

Provides an HTTP bridge for a local MT5 terminal running the ``ea_bridge.mq5``
Expert Advisor. The EA periodically polls for pending orders and posts back
fill reports. Aries execution code talks to ``EaBroker`` and places market
orders exactly as it does with PaperBroker/MetaApiClient — `place_market_order`
writes an ``EaOrder`` row and waits (up to ``EA_ORDER_TIMEOUT_SECONDS``) for the
terminal to report a fill.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import json as _json
import uuid

from fastapi import APIRouter, Query, Request, Response, status
from sqlalchemy import select, update

from ..config import Settings
from ..db.models import EaOrder, TradeSide
from ..logging import get_logger
from .base import ExecutionClient, OrderRequest, OrderResult
from .validator import OrderValidator

log = get_logger(__name__)

router = APIRouter(prefix="/ea", tags=["ea-bridge"])


# ---------------------------------------------------------------- client
class EaBroker(ExecutionClient):
    """Places orders through the local MetaTrader terminal."""

    name = "ea"

    def __init__(self, settings: Settings, session_factory, validator: OrderValidator | None = None) -> None:
        super().__init__(validator=validator)
        self.settings = settings
        self.session_factory = session_factory

    async def place_market_order(self, request: OrderRequest) -> OrderResult:
        external_id = request.client_order_id or uuid.uuid4().hex
        request.client_order_id = external_id

        # Pre-execution symbol validation
        val = self.validate_request(request)
        if not val.valid:
            log.warning("EaBroker rejected invalid order [%s]: %s", external_id, val.error)
            res = OrderResult(ok=False, error=val.error, client_order_id=external_id)
            self.record_processed_result(external_id, res)
            return res

        async with self.session_factory() as session:
            # Check for existing order with same external_id for idempotency
            existing = (await session.execute(
                select(EaOrder).where(EaOrder.external_id == external_id)
            )).scalar_one_or_none()

            if existing is not None:
                if existing.status == "filled":
                    return OrderResult(
                        ok=True,
                        broker_order_id=existing.broker_order_id or "",
                        fill_price=existing.fill_price or 0.0,
                        client_order_id=external_id,
                    )
                if existing.status == "rejected":
                    return OrderResult(ok=False, error=existing.error, client_order_id=external_id)
            else:
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
                    res = OrderResult(
                        ok=True,
                        broker_order_id=row.broker_order_id or "",
                        fill_price=row.fill_price or 0.0,
                        client_order_id=external_id,
                    )
                    self.record_processed_result(external_id, res)
                    return res
                if row.status == "rejected":
                    res = OrderResult(ok=False, error=row.error, client_order_id=external_id)
                    self.record_processed_result(external_id, res)
                    return res

        err = f"ea timeout after {self.settings.ea_order_timeout_seconds}s"
        log.error("EaBroker order %s timed out waiting for fill report", external_id)
        res = OrderResult(ok=False, error=err, client_order_id=external_id)
        self.record_processed_result(external_id, res)
        return res


# ---------------------------------------------------------------- helpers
def _auth_ok(settings: Settings, key: str | None) -> bool:
    return not settings.ea_api_key or key == settings.ea_api_key


# ---------------------------------------------------------------- endpoints
def _text(value: str, status_code: int = 200) -> Response:
    return Response(content=value, media_type="text/plain", status_code=status_code)


@router.get("/ping")
async def ea_ping(request: Request):
    settings = request.app.state.settings
    return {
        "ok": True,
        "mode": settings.trading_mode.value,
        "driver": settings.execution_driver,
        "key_required": bool(settings.ea_api_key),
    }


@router.post("/test-order")
async def test_order(request: Request, key: str | None = Query(default=None)) -> Response:
    """Queue a minimal 0.01-lot market order to verify the EA link end-to-end.

    The EA should claim it within ~1s and open 0.01 lots in the terminal.
    Check the result with GET /ea/order/<external_id>.
    """
    settings: Settings = request.app.state.settings
    factory = request.app.state.db.session_factory
    if not _auth_ok(settings, key):
        return _text("AUTHFAIL", status_code=401)
    external = uuid.uuid4().hex
    async with factory() as session:
        session.add(EaOrder(
            external_id=external,
            symbol=settings.market_symbol,
            side=TradeSide.BUY,
            volume=0.01,
        ))
        await session.commit()
    return _text(f"QUEUED|{external}")


@router.get("/order/{external_id}")
async def order_status(external_id: str, request: Request, key: str | None = Query(default=None)) -> Response:
    """Plain-text status of one bridge order: STATUS|<status>|<fill>|<error>."""
    settings: Settings = request.app.state.settings
    factory = request.app.state.db.session_factory
    if not _auth_ok(settings, key):
        return _text("AUTHFAIL", status_code=401)
    async with factory() as session:
        row = (await session.execute(
            select(EaOrder).where(EaOrder.external_id == external_id)
        )).scalar_one_or_none()
    if row is None:
        return _text("ERR:unknown-order", status_code=404)
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
        return _text("AUTHFAIL", status_code=401)

    try:
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
    except Exception as exc:
        log.error("Failed in /ea/next-order: %s", exc)
        return _text("ERR:internal-error", status_code=500)


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
        return _text("AUTHFAIL", status_code=401)

    try:
        raw = (await request.body()).decode("utf-8", errors="replace").strip()
    except Exception as exc:
        log.error("Failed to read execution report request body: %s", exc)
        return _text("ERR:read-failure", status_code=400)

    try:
        if raw.startswith("{"):
            payload = _json.loads(raw)
            external_id = str(payload.get("external_id", "")).strip()
            ok = bool(payload.get("ok"))
            broker_id = str(payload.get("broker_order_id", "")).strip()
            fill_val = payload.get("fill_price")
            fill = float(fill_val) if fill_val is not None and str(fill_val).strip() != "" else None
            error = str(payload.get("error", ""))
        else:
            parts = raw.split("|")
            if len(parts) < 5 or parts[0] != "RESULT":
                log.warning("Malformed plain-text execution report: %r", raw)
                return _text("ERR:bad-format", status_code=400)
            external_id = parts[1].strip()
            ok = parts[2] == "1"
            broker_id = parts[3].strip()
            fill = float(parts[4]) if parts[4].strip() else None
            error = "|".join(parts[5:]) if not ok else ""
    except Exception as exc:
        log.error("Error parsing execution report payload: %s (raw=%r)", exc, raw)
        return _text(f"ERR:parse-error:{exc}", status_code=400)

    if not external_id:
        log.warning("Execution report missing external_id (raw=%r)", raw)
        return _text("ERR:missing-id", status_code=400)

    try:
        async with factory() as session:
            row = (await session.execute(
                select(EaOrder).where(EaOrder.external_id == external_id)
            )).scalar_one_or_none()
            if row is None:
                log.warning("Execution report for unknown order: %s", external_id)
                return _text("ERR:unknown-order", status_code=404)

            # Idempotency check: if order is already processed, log and acknowledge OK
            if row.status in ("filled", "rejected"):
                log.info("Duplicate execution report received for order %s (current status: %s)", external_id, row.status)
                return _text("OK")

            if ok:
                row.status = "filled"
                row.broker_order_id = broker_id
                row.fill_price = fill
                log.info("Execution acknowledged: order %s FILLED @ %s (broker ticket=%s)", external_id, fill, broker_id)
            else:
                row.status = "rejected"
                row.error = error
                log.warning("Execution acknowledged: order %s REJECTED (%s)", external_id, error)

            row.updated_at = _dt.datetime.now(_dt.timezone.utc)
            await session.commit()
        return _text("OK")
    except Exception as exc:
        log.error("Unhandled error processing execution report for %s: %s", external_id, exc, exc_info=True)
        return _text(f"ERR:db-error:{exc}", status_code=500)