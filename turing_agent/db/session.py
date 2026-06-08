"""
Async database session management.

Provides get_session() context manager, init_db(), and close_db().
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from turing_agent.db.engine import dispose_engine, get_engine

_session_factory: async_sessionmaker[AsyncSession] | None = None


def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the async session factory."""
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        _session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager that yields a database session.

    Usage:
        async with get_session() as session:
            result = await session.execute(query)
    """
    factory = _get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Initialize database: create engine, verify connectivity."""
    engine = get_engine()
    async with engine.connect() as conn:
        await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
    _get_session_factory()


async def close_db() -> None:
    """Close all database connections and dispose engine."""
    global _session_factory
    _session_factory = None
    await dispose_engine()
