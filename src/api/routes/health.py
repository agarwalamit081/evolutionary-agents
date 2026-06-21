"""Health check routes — liveness and readiness endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe — is the service running?"""
    return {"status": "alive", "service": "turing-agent"}


@router.get("/ready")
async def ready() -> dict[str, Any]:
    """Readiness probe — is the service ready to accept requests?

    Checks database and Redis connectivity.
    """
    checks: dict[str, bool] = {}

    # Check PostgreSQL
    try:
        import sqlalchemy as sa
        from src.db.session import get_engine

        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(sa.text("SELECT 1"))
        checks["postgresql"] = True
    except Exception:
        checks["postgresql"] = False

    # Check Redis
    try:
        import redis.asyncio as aioredis
        from src.config import get_settings

        settings = get_settings()
        client = aioredis.from_url(settings.redis.redis_url)
        await client.ping()  # type: ignore[union-attr]
        await client.aclose()
        checks["redis"] = True
    except Exception:
        checks["redis"] = False

    all_ready = all(checks.values())
    return {
        "status": "ready" if all_ready else "degraded",
        "checks": checks,
    }
