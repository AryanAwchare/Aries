"""Audit and risk event logging helpers."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from .models import RiskEvent, SystemEvent


async def record_audit(
    session: AsyncSession,
    actor: str,
    action: str,
    detail: dict | None = None,
    commit: bool = True,
) -> SystemEvent:
    entry = SystemEvent(
        timestamp=datetime.now(timezone.utc),
        component=actor,
        event=action,
        detail=detail or {},
    )
    session.add(entry)
    if commit:
        await session.commit()
    return entry


async def record_risk_event(
    session: AsyncSession,
    event: str,
    severity: str = "INFO",
    drawdown: float = 0.0,
    daily_loss: float = 0.0,
    action: str = "",
    detail: dict | None = None,
    commit: bool = True,
) -> RiskEvent:
    entry = RiskEvent(
        timestamp=datetime.now(timezone.utc),
        event=event,
        severity=severity,
        drawdown=drawdown,
        daily_loss=daily_loss,
        action=action,
        detail=detail or {},
    )
    session.add(entry)
    if commit:
        await session.commit()
    return entry