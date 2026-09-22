"""App-level singleton services (kill switch, settings, database).

Kept deliberately tiny; long-lived state lives here so the FastAPI app,
scheduler jobs and backtest harness share one truth.
"""
from __future__ import annotations

import uuid
from threading import Lock

from .config import Settings, get_settings
from .db.base import Database
from .execution.killswitch import get_kill_switch

_settings: Settings | None = None
_db: Database | None = None


_kill_switch = get_kill_switch()


def get_settings_cached() -> Settings:
    global _settings
    if _settings is None:
        _settings = get_settings()
    return _settings


def get_db() -> Database:
    global _db
    if _db is None:
        _db = Database(get_settings_cached())
    return _db


def get_kill_switch() -> KillSwitch:
    return _kill_switch


class CopyTradingToggle:
    """Phase 7 — whether the leader-signal branch may feed the strategy engine.

    * Defaults to OFF.
    * OFF is instant (single call, no confirmation).
    * ON requires two explicit confirmations in sequence (first call returns a
      one-time token; the second call consumes it). If the active prop profile
      forbids copy-trading, enabling is rejected outright — mirroring the
      risk engine's compatibility gate.
    """

    def __init__(self, initially_enabled: bool = False) -> None:
        self._enabled = initially_enabled
        self._pending_token: str | None = None
        self._lock = Lock()

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def request_enable(self, profile_allows_copy: bool) -> str | None:
        """First confirmation. Returns a one-time token, or None if refused."""
        with self._lock:
            if not profile_allows_copy:
                return None
            self._pending_token = uuid.uuid4().hex
            return self._pending_token

    def confirm_enable(self, token: str | None) -> bool:
        """Second confirmation. Consumes the token; ON only after both steps."""
        with self._lock:
            if token is None or self._pending_token is None:
                return False
            if token != self._pending_token:
                return False
            self._enabled = True
            self._pending_token = None
            return True

    def disable(self) -> None:
        """Single, immediate, no-confirmation OFF."""
        with self._lock:
            self._enabled = False
            self._pending_token = None


_copy_trading = CopyTradingToggle()


def get_copy_trading_toggle() -> CopyTradingToggle:
    return _copy_trading