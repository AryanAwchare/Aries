"""Process-wide emergency stop.

When disabled, the risk engine refuses every order regardless of signal
strength. The state is process-local; a distributed deployment should back
this with a shared KV store.
"""
from __future__ import annotations

import threading


class KillSwitch:
    def __init__(self, initially_enabled: bool = True) -> None:
        self._enabled = initially_enabled
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def enable(self) -> None:
        with self._lock:
            self._enabled = True

    def disable(self) -> None:
        with self._lock:
            self._enabled = False


_default = KillSwitch()


def get_kill_switch() -> KillSwitch:
    return _default