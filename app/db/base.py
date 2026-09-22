"""Async database engine/session management (SQLAlchemy 2.0)."""
from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from ..config import Settings


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def create_sessionmaker(settings: Settings, echo: bool = False) -> async_sessionmaker[AsyncSession]:
    engine = create_async_engine(settings.database_url, echo=echo, future=True)
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Database:
    """Thin wrapper that owns engine + sessionmaker for one app."""

    def __init__(self, settings: Settings, echo: bool = False) -> None:
        self.settings = settings
        self.engine = create_async_engine(settings.database_url, echo=echo, future=True)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)

    async def create_all(self) -> None:
        # Import models so they register with the metadata.
        from . import models  # noqa: F401

        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def dispose(self) -> None:
        await self.engine.dispose()

    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        async with self.session_factory() as session:
            yield session