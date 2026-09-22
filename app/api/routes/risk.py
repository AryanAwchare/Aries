"""Risk-engine status, prop profile info, kill switch control."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from ... import services
from ...config import BASE_DIR
from ...db import audit
from ...db.base import Database
from ...risk.prop_profiles import load_profile, load_profile_dir

router = APIRouter(prefix="/api/risk", tags=["risk"])


class KillSwitchPayload(BaseModel):
    enabled: bool


async def _session(request: Request) -> AsyncSession:
    db: Database = request.app.state.db
    async with db.session_factory() as session:
        yield session


@router.get("/status")
async def risk_status(request: Request):
    settings = request.app.state.settings
    profile_dir = BASE_DIR / "app" / "risk" / "prop_profiles"
    profile = load_profile(profile_dir / f"{settings.active_prop_profile}.yaml")
    return {
        "profile": settings.active_prop_profile,
        "capital": profile.capital,
        "profit_target": profile.profit_target,
        "daily_loss_firm": profile.daily_loss_limit_pct,
        "daily_loss_internal_soft": profile.daily_loss_soft_limit_pct,
        "drawdown_firm": profile.max_drawdown_pct,
        "drawdown_emergency": profile.emergency_stop_drawdown_pct,
        "drawdown_operating_ceiling": profile.operating_ceiling_drawdown_pct,
        "copy_trading_allowed": profile.copy_trading_allowed,
        "copy_trading_enabled": services.get_copy_trading_toggle().enabled,
        "max_orders_per_minute": profile.max_orders_per_minute,
        "max_orders_per_hour": profile.max_orders_per_hour,
    }


@router.get("/profiles")
async def profiles():
    listed = load_profile_dir(BASE_DIR / "app" / "risk" / "prop_profiles")
    return [{"name": p.name, "description": p.description, "copy_trading_allowed": p.copy_trading_allowed} for p in listed.values()]


@router.get("/killswitch")
async def get_killswitch():
    ks = services.get_kill_switch()
    return {"enabled": ks.enabled}


@router.put("/killswitch")
async def set_killswitch(payload: KillSwitchPayload, request: Request, session: AsyncSession = Depends(_session)):
    ks = services.get_kill_switch()
    enabled = payload.enabled
    if enabled:
        ks.enable()
    else:
        ks.disable()
    await audit.record_audit(
        session, actor="user", action=("killswitch.on" if enabled else "killswitch.off"),
        detail={"trading_mode": request.app.state.settings.trading_mode.value},
    )
    await session.commit()
    return {"enabled": ks.enabled}


@router.put("/copy-trading/off")
async def copy_trading_off(request: Request, session: AsyncSession = Depends(_session)):
    """Single-click, immediate, no-confirmation OFF."""
    toggle = services.get_copy_trading_toggle()
    toggle.disable()
    await audit.record_audit(session, actor="user", action="copy_trading.off", detail={})
    await session.commit()
    return {"copy_trading_enabled": toggle.enabled}


@router.post("/copy-trading/on")
async def copy_trading_on(request: Request, session: AsyncSession = Depends(_session)):
    """Step 1 of 2 — request enabling. Returns a one-time confirmation token,
    or null if the active profile forbids copy-trading."""
    settings = request.app.state.settings
    profile = load_profile(BASE_DIR / "app" / "risk" / "prop_profiles" / f"{settings.active_prop_profile}.yaml")
    toggle = services.get_copy_trading_toggle()
    token = toggle.request_enable(profile.copy_trading_allowed)
    if token is None:
        raise HTTPException(409, "copy-trading is not permitted under the active prop profile")
    await audit.record_audit(session, actor="user", action="copy_trading.enable.requested",
                             detail={"profile": settings.active_prop_profile})
    await session.commit()
    return {"confirmation_token": token, "profile_copy_trading_allowed": profile.copy_trading_allowed}


@router.put("/copy-trading/on")
async def copy_trading_on_confirm(
    confirmation_token: str, request: Request, session: AsyncSession = Depends(_session)
):
    """Step 2 of 2 — second explicit confirmation. Only now does ON take effect."""
    toggle = services.get_copy_trading_toggle()
    ok = toggle.confirm_enable(confirmation_token)
    if not ok:
        raise HTTPException(400, "confirmation token missing, expired, or already used")
    await audit.record_audit(session, actor="user", action="copy_trading.on",
                             detail={"trading_mode": request.app.state.settings.trading_mode.value})
    await session.commit()
    return {"copy_trading_enabled": toggle.enabled}