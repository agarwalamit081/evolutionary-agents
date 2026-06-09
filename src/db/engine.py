"""
SQLAlchemy async engine for PostgreSQL with asyncpg driver.

Creates the engine from settings and provides a module-level accessor.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from src.config.settings import get_settings

_engine: AsyncEngine | None = None


def get_engine() -> AsyncEngine:
    """Get or create the async database engine singleton."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database.database_url,
            pool_size=settings.database.database_pool_size,
            max_overflow=settings.database.database_max_overflow,
            pool_timeout=settings.database.database_pool_timeout,
            pool_recycle=settings.database.database_pool_recycle,
            echo=False,
        )
    return _engine


async def dispose_engine() -> None:
    """Dispose the engine and close all connections."""
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
