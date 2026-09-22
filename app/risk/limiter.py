"""In-memory order-frequency limiter.

Prop firms routinely cap automated order activity (orders/min, orders/hr).
This limiter is enforced independently of any strategy logic — it is the
one thing that must never be skipped.
"""
from __future__ import annotations

import threading
import time
from collections import deque


class RateLimiter:
    def __init__(self, per_minute: int = 4, per_hour: int = 20) -> None:
        self.per_minute = per_minute
        self.per_hour = per_hour
        self._stamps: deque[float] = deque()
        self._lock = threading.Lock()

    def _drop_older_than(self, cutoff: float) -> None:
        while self._stamps and self._stamps[0] < cutoff:
            self._stamps.popleft()

    def allowed(self, count: int = 1) -> bool:
        now = time.time()
        with self._lock:
            if self.per_minute > 0:
                self._drop_older_than(now - 60)
                if len(self._stamps) + count > self.per_minute:
                    return False
            if self.per_hour > 0:
                self._drop_older_than(now - 3600)
                if sum(1 for s in self._stamps if s >= now - 3600) + count > self.per_hour:
                    return False
            for _ in range(count):
                self._stamps.append(now)
            return True

    def reset(self) -> None:
        with self._lock:
            self._stamps.clear()