"""Tests for src.api.app — FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI

from src.api.app import create_app


class TestCreateApp:
    """Tests for the create_app factory function."""

    def test_create_app_returns_fastapi(self) -> None:
        """create_app returns a FastAPI instance."""
        app = create_app()
        assert isinstance(app, FastAPI)

    def test_app_title(self) -> None:
        """App title is 'Turing Agent API'."""
        app = create_app()
        assert app.title == "Turing Agent API"

    def test_app_has_health_route(self) -> None:
        """App has /health route registered."""
        app = create_app()
        routes = [route.path for route in app.routes]
        assert "/health" in routes

    def test_app_has_ready_route(self) -> None:
        """App has /ready route registered."""
        app = create_app()
        routes = [route.path for route in app.routes]
        assert "/ready" in routes

    def test_agent_cancel_route_resolves_at_root(self) -> None:
        """``POST /runs/{run_id}/cancel`` resolves at root, not only under the
        ``/api/v1/agent`` prefix — the documented cancel path previously 404'd.
        """
        app = create_app()
        routes = [route.path for route in app.routes]
        assert "/runs/{run_id}/cancel" in routes

    def test_agent_routes_dual_mounted(self) -> None:
        """The agent router is mounted at BOTH root and ``/api/v1/agent`` so the
        prefixed path stays backward-compatible while the bare documented paths
        (``/run``, ``/runs/{run_id}``, ``/runs/{run_id}/cancel``) also resolve.
        """
        app = create_app()
        routes = [route.path for route in app.routes]
        # Prefixed (backward compat) — enqueue_run's status_url points here.
        assert "/api/v1/agent/run" in routes
        assert "/api/v1/agent/runs/{run_id}" in routes
        assert "/api/v1/agent/runs/{run_id}/cancel" in routes
        # Root (documented paths).
        assert "/run" in routes
        assert "/runs/{run_id}" in routes
        assert "/runs/{run_id}/cancel" in routes
