"""Trading-mode endpoints. Modes can only be changed explicitly, and the
platform refuses to resolve to ``live`` implicitly — this endpoint is the
one place that may *request* a switch, and it still validates.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ...config import BASE_DIR, TradingMode
from ...risk.prop_profiles import load_profile, load_profile_dir

router = APIRouter(prefix="/api/modes", tags=["modes"])

_ALLOWED = {m.value for m in TradingMode}


class ModePayload(BaseModel):
    mode: str


@router.get("")
async def get_mode(request: Request):
    return {"trading_mode": request.app.state.settings.trading_mode.value}


@router.put("")
async def set_mode(payload: ModePayload, request: Request):
    mode = payload.mode
    if mode not in _ALLOWED:
        raise HTTPException(422, f"invalid mode; allowed: {sorted(_ALLOWED)}")
    candidates = load_profile_dir(BASE_DIR / "app" / "risk" / "prop_profiles")
    if mode in (TradingMode.PROP_EVAL.value, TradingMode.LIVE.value) and not request.app.state.settings.metaapi_token:
        raise HTTPException(409, f"mode '{mode}' requires METAAPI_TOKEN and account credentials")
    if mode == TradingMode.LIVE.value and request.app.state.profile.name == "default_demo":
        raise HTTPException(409, "live mode refused while an internal/demo profile is active")

    # Persist the mode into the environment so restarts keep it.
    request.app.state.settings.trading_mode = TradingMode(mode)
    return {"trading_mode": mode}