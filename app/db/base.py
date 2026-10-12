"""Async database engine/session management (SQLAlchemy 2.0)."""
from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from ..config import Settings


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def create_sessionmaker(settings: Settings, echo: bool = False) -> async_sessionmaker[AsyncSession]:
    url = settings.effective_db_url
    engine_kwargs = {"echo": echo, "future": True}
    if not url.startswith("sqlite"):
        engine_kwargs["pool_pre_ping"] = True
        if "asyncpg" in url:
            engine_kwargs["connect_args"] = {"statement_cache_size": 0}
    engine = create_async_engine(url, **engine_kwargs)
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Database:
    """Thin wrapper that owns engine + sessionmaker for one app."""

    def __init__(self, settings: Settings, echo: bool = False) -> None:
        self.settings = settings
        url = settings.effective_db_url
        engine_kwargs = {"echo": echo, "future": True}
        if not url.startswith("sqlite"):
            engine_kwargs["pool_pre_ping"] = True
            if "asyncpg" in url:
                engine_kwargs["connect_args"] = {"statement_cache_size": 0}
        self.engine = create_async_engine(url, **engine_kwargs)
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