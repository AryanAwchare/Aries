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

    if settings.ops_api_key:
        log.info("Control-plane API key configured: mutating endpoints require OPS_API_KEY")
    else:
        log.critical(
            "OPS_API_KEY is not set: mutating control-plane endpoints (order, config, "
            "leaders, kill switch, copy-trading) are UNAUTHENTICATED. Set OPS_API_KEY "
            "before exposing the API beyond localhost."
        )

    db = services.get_db()
    await db.create_all()
    app.state.db = db

    profile_dir = BASE_DIR / "app" / "risk" / "prop_profiles"
    app.state.profile = load_profile(profile_dir / f"{settings.active_prop_profile}.yaml")
    app.state.kill_switch = services.get_kill_switch()
    app.state.copy_trading = services.get_copy_trading_toggle()

    if settings.kill_switch_enabled:
        app.state.kill_switch.enable()

    scheduler = None
    if settings.myfxbook_email:
        from .leaders.myfxbook import MyfxbookClient
        from .scheduler.jobs import build_scheduler

        myfxbook = MyfxbookClient(settings.myfxbook_email, settings.myfxbook_password)
        scheduler = build_scheduler(settings, db.session_factory, myfxbook)
        scheduler.start()
        app.state.scheduler = scheduler
        log.info("Background scheduler started")
    else:
        app.state.scheduler = None
    log.info("Platform booted in mode=%s profile=%s", settings.trading_mode.value, settings.active_prop_profile)

    yield

    if app.state.scheduler is not None:
        scheduler.shutdown(wait=False)
    await db.dispose()


app = FastAPI(title="Gold Hybrid Copy-Trading Platform", version="0.1.0", lifespan=lifespan)

from fastapi.responses import FileResponse, HTMLResponse  # noqa: E402
from .api.routes import dashboard, health, leaders, modes, ops, risk, signals  # noqa: E402
from .execution.ea_bridge import router as ea_router  # noqa: E402

app.include_router(dashboard.router)
app.include_router(health.router)
app.include_router(leaders.router)
app.include_router(signals.router)
app.include_router(risk.router)
app.include_router(modes.router)
app.include_router(ops.router)
app.include_router(ea_router)

_static_index = BASE_DIR / "app" / "static" / "index.html"

@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard():
    if _static_index.exists():
        return FileResponse(_static_index)
    return HTMLResponse("<h1>Aries Platform Running</h1><p><a href='/docs'>Swagger API Docs</a></p>")