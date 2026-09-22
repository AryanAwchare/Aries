"""Audit trail helpers — one append-only row per notable event."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from .models import AuditLogEntry


async def record_audit(
    session: AsyncSession,
    actor: str,
    action: str,
    detail: dict | None = None,
    commit: bool = True,
) -> AuditLogEntry:
    entry = AuditLogEntry(
        ts=datetime.now(timezone.utc),
        actor=actor,
        action=action,
        detail=detail or {},
    )
    session.add(entry)
    if commit:
        await session.commit()
    return entry