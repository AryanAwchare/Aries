"""Health + mode-awareness endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Request

from ...services import get_copy_trading_toggle, get_kill_switch

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request):
    settings = request.app.state.settings
    return {
        "status": "ok",
        "trading_mode": settings.trading_mode.value,
        "active_prop_profile": settings.active_prop_profile,
        "symbol": settings.market_symbol,
        "timeframe": settings.market_timeframe,
        "kill_switch": get_kill_switch().enabled,
        "copy_trading_enabled": get_copy_trading_toggle().enabled,
    }