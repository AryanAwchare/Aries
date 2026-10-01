"""FastAPI application entrypoint.

Startup loads settings, ensures the DB schema, builds the shared services
(risk engine with active prop profile, kill switch, predictors) and — unless
disabled — starts the background scheduler.

The platform never defaults to live: ``Settings`` validates the mode and any
unset/empty/invalid value resolves to ``paper``.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import services
from .config import BASE_DIR, get_settings
from .logging import get_logger, setup_logging
from .risk.prop_profiles import load_profile

log = get_logger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(log_dir=BASE_DIR / "logs")
    settings = get_settings()
    app.state.settings = settings

    db = services.get_db()
    await db.create_all()
    app.state.db = db

    profile_dir = BASE_DIR / "app" / "risk" / "prop_profiles"
    app.state.profile = load_profile(profile_dir / f"{settings.active_prop_profile}.yaml")
    app.state.kill_switch = services.get_kill_switch()
    app.state.copy_trading = services.get_copy_trading_toggle()

    if settings.kill_switch_enabled:
        app.state.kill_switch.enable()
    log.info("Platform booted in mode=%s profile=%s", settings.trading_mode.value, settings.active_prop_profile)

    yield

    await db.dispose()


app = FastAPI(title="Gold Hybrid Copy-Trading Platform", version="0.1.0", lifespan=lifespan)

from .api.routes import health, leaders, modes, risk, signals  # noqa: E402
from .execution.ea_bridge import router as ea_router  # noqa: E402

app.include_router(health.router)
app.include_router(leaders.router)
app.include_router(signals.router)
app.include_router(risk.router)
app.include_router(modes.router)
app.include_router(ea_router)