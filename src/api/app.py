"""FastAPI application factory for the Turing Agent."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from src.config import get_settings
from src.observability.logging import setup_logging


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
    from src.api.routes.health import router as health_router
    app.include_router(health_router, tags=["health"])

    try:
        from src.api.routes.agent import API_PREFIX, router as agent_router
        # Prefixed mount (backward compat) — /api/v1/agent/run, /runs/{id},
        # /runs/{id}/cancel. ``enqueue_run``'s ``status_url`` points here.
        app.include_router(agent_router, prefix=API_PREFIX, tags=["agent"])
        # Root mount so the documented paths (/run, /runs/{id},
        # /runs/{id}/cancel) resolve without the prefix too — the cancel route
        # previously 404'd at root (the agent router carries no router-level
        # prefix, so mounting it at root exposes its routes verbatim).
        app.include_router(agent_router, tags=["agent"])
    except ImportError:
        logger.warning("Agent routes not available yet")

    # D10: operator tool-edit → review → approve HITL routes. Import-wrapped so
    # the API still boots if an optional dep (e.g. the safety stack) is absent.
    try:
        from src.api.routes.tool import API_PREFIX as TOOL_PREFIX
        from src.api.routes.tool import router as tool_router
        app.include_router(tool_router, prefix=TOOL_PREFIX, tags=["tools"])
    except ImportError:
        logger.warning("Tool edit routes not available yet")

    return app
