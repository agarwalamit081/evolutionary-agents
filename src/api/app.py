"""FastAPI application factory for the Turing Agent."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import Response
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


async def _metrics() -> Response:
    """Prometheus scrape endpoint.

    Returns the default registry rendered by ``prometheus_client``; an empty
    body when ``prometheus_client`` is absent (the endpoint stays up but reports
    no metrics). Observability-only — never raises into a scrape.
    """
    from src.observability.metrics import metrics_response

    body, content_type = metrics_response()
    return Response(content=body, media_type=content_type)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance with routes registered.
    """
    settings = get_settings()
    app = FastAPI(
        title="Turing Agent API",
        description="Self-evolving AI agent built with LangGraph",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Observability: OTel tracing. The process-global provider + asyncpg/httpx
    # auto-instrumentors initialize once (opt-in via OTEL_ENABLED); the FastAPI
    # request-span instrumentor is app-scoped. Both are idempotent + best-effort
    # (a missing exporter/instrumentor disables only itself, never app boot).
    # Configured here so spans are ready before the first request arrives.
    obs = settings.observability
    if obs.otel_enabled:
        from src.observability.tracing import setup_tracing

        setup_tracing(
            service_name=obs.otel_service_name,
            endpoint=obs.otel_endpoint,
            sampling_rate=obs.otel_sampling_rate,
        )
    from src.observability.tracing import instrument_fastapi_app

    instrument_fastapi_app(app)

    # Prometheus scrape endpoint. Registered unconditionally — it returns an
    # empty body when prometheus_client is absent or metrics are off, so it is
    # harmless; meaningful only when the recorder call sites have fired.
    app.add_api_route("/metrics", _metrics, tags=["observability"])

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

    # Phase 5 — operator dashboard (server-rendered HTML at /dashboard) + the
    # vendored CSS/JS at /dashboard-static. Mounted at the root (a UI, not a
    # programmatic API). Import-wrapped so the API still boots if jinja2 is
    # absent (the JSON API stays up even if the dashboard cannot render).
    try:
        from pathlib import Path

        from starlette.staticfiles import StaticFiles

        from src.api.routes.dashboard import router as dashboard_router
        app.include_router(dashboard_router, tags=["dashboard"])
        # The dashboard binds 0.0.0.0 (host port 8800) and is gated opt-in:
        # empty DASHBOARD_API_KEY = open. Warn loudly when it ships open so an
        # accidentally-exposed operator UI is never silent.
        if not settings.dashboard.api_key:
            logger.warning(
                "Dashboard mounted WITHOUT an auth gate (DASHBOARD_API_KEY unset) "
                "— set DASHBOARD_API_KEY or bind/firewall port 8800 to restrict access."
            )
        static_dir = Path(__file__).resolve().parent / "static"
        if static_dir.is_dir():
            app.mount(
                "/dashboard-static",
                StaticFiles(directory=str(static_dir)),
                name="dashboard-static",
            )
    except ImportError:
        logger.warning("Dashboard routes not available (jinja2 missing?)")

    return app
