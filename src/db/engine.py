"""
SQLAlchemy async engine for PostgreSQL with asyncpg driver.

Creates the engine from settings and provides a module-level accessor.
"""

from __future__ import annotations

from loguru import logger
from sqlalchemy import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from src.config.settings import get_settings

_engine: AsyncEngine | None = None


def get_engine() -> AsyncEngine:
    """Get or create the async database engine singleton."""
    global _engine
    if _engine is None:
        settings = get_settings()
        database_url = settings.database.database_url
        _engine = create_async_engine(
            database_url,
            pool_size=settings.database.database_pool_size,
            max_overflow=settings.database.database_max_overflow,
            pool_timeout=settings.database.database_pool_timeout,
            pool_recycle=settings.database.database_pool_recycle,
            echo=False,
        )
        # Log WHICH database we resolved to every run (§2): a host-local
        # Postgres on 5432 can silently shadow the docker container, so prior
        # runs may have written to a throwaway DB. make_url() parses safely and
        # we log only host/port/database/driver — NEVER the password (the raw
        # connection string is deliberately not logged).
        try:
            url = make_url(database_url)
            logger.info(
                "Database engine created → {}:{}/{} (driver={})".format(
                    url.host,
                    url.port,
                    url.database,
                    url.drivername,
                )
            )
        except Exception:  # best-effort: a logging hiccup must never block engine creation
            logger.debug("Could not parse database_url for startup log")
    return _engine


async def dispose_engine() -> None:
    """Dispose the engine and close all connections."""
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
