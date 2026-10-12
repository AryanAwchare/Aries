"""Asynchronous non-blocking persistence worker for Supabase / PostgreSQL.

Ensures the critical trading engine execution path (Myfxbook/Kronos -> Engine -> MT5)
never blocks on Supabase network latency, connection pool waits, or DB outages.
Writes are queued in-memory and processed asynchronously by a background worker.
If the database is down or unresponsive, execution proceeds uninterrupted and
errors are recorded to local fallback logs.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ..logging import get_logger

log = get_logger(__name__)


@dataclass
class WriteTask:
    action_name: str
    coro_fn: Callable[..., Coroutine[Any, Any, Any]]
    args: tuple
    kwargs: dict
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)


class AsyncDatabaseWriter:
    """Non-blocking queue-based background writer for fire-and-forget telemetry & persistence."""

    def __init__(self, max_queue_size: int = 10000) -> None:
        self._queue: asyncio.Queue[WriteTask] = asyncio.Queue(maxsize=max_queue_size)
        self._worker_task: asyncio.Task | None = None
        self._running: bool = False
        self._dropped_count: int = 0
        self._processed_count: int = 0
        self._error_count: int = 0

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    def start(self) -> None:
        if not self._running:
            self._running = True
            self._worker_task = asyncio.create_task(self._worker_loop())
            log.info("AsyncDatabaseWriter background worker started")

    async def stop(self) -> None:
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
            log.info("AsyncDatabaseWriter stopped (processed=%d, errors=%d, dropped=%d)",
                     self._processed_count, self._error_count, self._dropped_count)

    def enqueue(self, action_name: str, coro_fn: Callable[..., Coroutine[Any, Any, Any]], *args, **kwargs) -> bool:
        """Enqueue a write task in a non-blocking fire-and-forget manner."""
        task = WriteTask(action_name=action_name, coro_fn=coro_fn, args=args, kwargs=kwargs)
        try:
            self._queue.put_nowait(task)
            return True
        except asyncio.QueueFull:
            self._dropped_count += 1
            log.warning("AsyncDatabaseWriter queue full (%d items)! Dropping write task: %s", self._queue.maxsize, action_name)
            return False

    async def _worker_loop(self) -> None:
        while self._running:
            try:
                task = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                await task.coro_fn(*task.args, **task.kwargs)
                self._processed_count += 1
            except Exception as exc:
                self._error_count += 1
                log.error("AsyncDatabaseWriter failed executing %s: %s", task.action_name, exc)
            finally:
                self._queue.task_done()


_global_async_writer: AsyncDatabaseWriter | None = None


def get_async_writer() -> AsyncDatabaseWriter:
    global _global_async_writer
    if _global_async_writer is None:
        _global_async_writer = AsyncDatabaseWriter()
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                _global_async_writer.start()
        except RuntimeError:
            pass  # No running loop yet
    return _global_async_writer