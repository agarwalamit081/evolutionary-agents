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
