"""Single access point for the local MetaTrader 5 terminal.

Every direct ``MetaTrader5`` call in the platform goes through this module so
the terminal connection is **always** torn down. The previous code called
``mt5.initialize()`` in several places without a matching ``mt5.shutdown()``,
which leaks terminal handles across requests.

Usage::

    from app.data.mt5_client import MT5_AVAILABLE, session

    with session() as mt5:
        account = mt5.account_info()

``session`` yields the ``MetaTrader5`` module bound to a live connection and
guarantees ``mt5.shutdown()`` on exit (success, error, or early return).
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ..logging import get_logger

log = get_logger(__name__)

try:  # Windows-only package
    import MetaTrader5 as mt5

    MT5_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on host platform
    mt5 = None
    MT5_AVAILABLE = False

DEFAULT_TERMINAL_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
DEFAULT_TIMEOUT_MS = 20000


def is_available() -> bool:
    """True when the ``MetaTrader5`` package is importable."""
    return MT5_AVAILABLE


@contextmanager
def session(
    *,
    path: str | None = None,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    login: int | None = None,
    password: str | None = None,
    server: str | None = None,
) -> Iterator[Any]:
    """Open a terminal connection and guarantee shutdown.

    Raises ``RuntimeError`` if the package is missing or ``initialize`` fails.
    """
    if not MT5_AVAILABLE:
        raise RuntimeError("MetaTrader5 package is not installed")

    kwargs: dict[str, Any] = {"timeout": timeout_ms}
    if path:
        kwargs["path"] = path
    if login is not None:
        kwargs["login"] = int(login)
    if password:
        kwargs["password"] = password
    if server:
        kwargs["server"] = server

    if not mt5.initialize(**kwargs):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        yield mt5
    finally:
        mt5.shutdown()
