"""Regression for Fix 2 — the opt-in ``X-Dashboard-Key`` auth gate.

The dashboard binds ``0.0.0.0`` (host port 8800) and previously had NO auth on
any ``/dashboard*`` route — exposing runs, per-model cost, the HITL review
output, the eval matrix, and the mutation timeline. Fix 2 adds an opt-in gate
applied on the router itself: an empty ``DASHBOARD_API_KEY`` leaves the UI open
(byte-identical to the prior local-dev behavior); a set key requires a
constant-time-matching ``X-Dashboard-Key`` header on every dashboard route.

These tests override the ``_dashboard_api_key`` dependency (so no live
``.env``/``get_settings`` is needed) and exercise the gate through the real HTTP
layer. The 401 cases need no Redis/DB (the gate fires before the handler); the
200 cases patch the data layer so the handler renders without infra.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.routes import dashboard_data as data
from src.api.routes.dashboard import _dashboard_api_key


_EMPTY_SUMMARY = {
    "runs_total": 0,
    "runs_in_flight": 0,
    "runs_completed": 0,
    "total_cost_usd": 0.0,
}


class _FakeSession:
    """Minimal async context manager standing in for ``get_session()``."""

    async def __aenter__(self) -> MagicMock:
        return MagicMock()

    async def __aexit__(self, *_exc: object) -> bool:
        return False


def _client_with_key(key: str) -> TestClient:
    """An app+client whose ``_dashboard_api_key`` dependency returns ``key``."""
    app = create_app()
    app.dependency_overrides[_dashboard_api_key] = lambda: key
    return TestClient(app)


@contextmanager
def _render_index() -> Generator[None, None, None]:
    """Patch the index handler's infra + data fns so it renders 200 without Redis/DB."""
    redis = MagicMock()
    redis.aclose = AsyncMock()
    with patch(
        "src.api.routes.dashboard._open_store",
        new=AsyncMock(return_value=(MagicMock(), redis)),
    ), patch(
        "src.api.routes.dashboard.get_session", return_value=_FakeSession()
    ), patch.object(
        data, "runs_with_cost", new=AsyncMock(return_value=([], _EMPTY_SUMMARY))
    ), patch.object(
        data, "mutation_timeline", new=AsyncMock(return_value=[])
    ):
        yield


class TestDashboardAuthGate:
    def test_open_when_key_unset(self) -> None:
        client = _client_with_key("")
        with _render_index():
            resp = client.get("/dashboard")
        assert resp.status_code == 200, resp.text

    def test_401_when_key_set_and_no_header(self) -> None:
        client = _client_with_key("s3cret!")
        # The gate fires before the handler, so no Redis/DB patch is needed.
        resp = client.get("/dashboard")
        assert resp.status_code == 401

    def test_200_when_key_set_and_header_matches(self) -> None:
        client = _client_with_key("s3cret!")
        with _render_index():
            resp = client.get("/dashboard", headers={"X-Dashboard-Key": "s3cret!"})
        assert resp.status_code == 200, resp.text

    def test_401_when_header_wrong(self) -> None:
        client = _client_with_key("s3cret!")
        resp = client.get("/dashboard", headers={"X-Dashboard-Key": "nope"})
        assert resp.status_code == 401

    def test_gate_applies_to_every_dashboard_route(self) -> None:
        # The gate is on the router itself (not per-route), so a non-index route
        # is locked too. 401 fires before the handler → no infra patch needed.
        client = _client_with_key("s3cret!")
        for path in ("/dashboard/runs", "/dashboard/curve", "/dashboard/mutations"):
            assert client.get(path).status_code == 401, path
