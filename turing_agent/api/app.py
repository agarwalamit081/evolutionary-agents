"""FastAPI application factory for the Turing Agent."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from turing_agent.config import get_settings
from turing_agent.observability.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    settings = get_settings()
    setup_logging(settings.logging)
    logger.info("Turing Agent API starting up")
    yield
    logger.info("Turing Agent API shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance with routes registered.
    """
    app = FastAPI(
        title="Turing Agent API",
        description="Self-evolving AI agent built with LangGraph",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Register routes
    from turing_agent.api.routes.health import router as health_router
    app.include_router(health_router, tags=["health"])

    try:
        from turing_agent.api.routes.agent import router as agent_router
        app.include_router(agent_router, prefix="/api/v1/agent", tags=["agent"])
    except ImportError:
        logger.warning("Agent routes not available yet")

    return app
