"""HTTP server tests for the optimizer sidecar (Phase 2 C2).

Mirrors ``test_sandbox/test_runner_server.py``: an aiohttp ``TestClient`` backed
by :func:`build_app` with an INJECTED fake optimizer — no real dspy, no DB, no
LLM. Pins the wire contract + the security.md error discipline:

  - ``GET /healthz`` → ``{"status": "ok"}`` (compose healthcheck).
  - ``POST /optimize`` with an empty body → ``200`` + the engine's
    :class:`OptimizeResponse` (the scheduler's real nightly call shape).
  - ``POST /optimize`` with a non-JSON body → ``400`` (``request body must be
    JSON``).
  - ``POST /optimize`` with a bad ``backend`` enum → ``422`` (Pydantic
    field-level validation errors — safe input feedback, not internals).
  - engine :class:`ConfigurationError` (e.g. ``backend="textgrad"``) → ``400``
    (a caller bug, surfaced — NOT a 500).
  - any other engine failure → ``500`` generic (``Something went wrong`` — no
    stack trace / path / DB error leaked).

The server module imports :mod:`src.optimizer.engine` (which imports dspy at
module top), so the whole module is importorskip-guarded on dspy.
"""

from __future__ import annotations

from typing import Optional

import pytest
from aiohttp.test_utils import TestClient, TestServer

# engine.py does ``import dspy`` at module top → server.py (which imports the
# engine) needs dspy present. importorskip keeps this module a clean skip when
# the ML stack isn't installed (mirrors how the engine/integration tests gate).
pytest.importorskip("dspy")

from src.optimizer.models import (  # noqa: E402 — after importorskip
    ConfigurationError,
    OptimizeRequest,
    OptimizeResponse,
)
from src.optimizer.server import build_app  # noqa: E402

# A canned success outcome the fake replays (no LLM involved).
_OK_RESPONSE = OptimizeResponse(
    node="classify",
    promoted=True,
    reason="promoted",
    baseline=0.6,
    candidate_score=0.8,
    suffixes=["optimized instruction"],
)


class _FakeOptimizer:
    """Duck-typed stand-in for :class:`PromptOptimizer`.

    Records each :class:`OptimizeRequest` and either replays ``response`` or
    raises ``raises``. Passing it to :func:`build_app` short-circuits the real
    ``PromptOptimizer()`` construction (``optimizer or PromptOptimizer()``) so
    no gateway / DB / dspy is touched.
    """

    def __init__(
        self,
        *,
        response: Optional[OptimizeResponse] = None,
        raises: Optional[BaseException] = None,
    ) -> None:
        self.response = response
        self.raises = raises
        self.calls: list[OptimizeRequest] = []

    async def optimize(self, req: OptimizeRequest) -> OptimizeResponse:
        self.calls.append(req)
        if self.raises is not None:
            raise self.raises
        assert self.response is not None  # configured for the success path
        return self.response


async def _started_client(fake: _FakeOptimizer) -> TestClient:
    """Build + start a TestClient backed by ``build_app(fake)``.

    Caller MUST ``await client.close()`` when done (the tests use try/finally).
    """
    app = build_app(fake)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


@pytest.mark.asyncio
async def test_healthz_returns_ok() -> None:
    """``GET /healthz`` → ``{"status": "ok"}`` (the compose healthcheck)."""
    client = await _started_client(_FakeOptimizer(response=_OK_RESPONSE))
    try:
        resp = await client.get("/healthz")
        assert resp.status == 200
        assert await resp.json() == {"status": "ok"}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_optimize_empty_body_returns_200_with_response() -> None:
    """The scheduler's real call shape: empty body → resolved defaults → 200."""
    fake = _FakeOptimizer(response=_OK_RESPONSE)
    client = await _started_client(fake)
    try:
        resp = await client.post("/optimize", json={})
        assert resp.status == 200
        data = await resp.json()
        assert data["node"] == "classify"
        assert data["promoted"] is True
        # The fake saw the request, with the empty body resolved to all-None.
        assert len(fake.calls) == 1
        assert fake.calls[0].node is None
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_optimize_non_json_body_rejected_with_400() -> None:
    """A non-JSON body (right content-type) → 400 (validation happens before optimize)."""
    client = await _started_client(_FakeOptimizer(response=_OK_RESPONSE))
    try:
        resp = await client.post(
            "/optimize", data="", headers={"Content-Type": "application/json"}
        )
        assert resp.status == 400
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_optimize_bad_backend_enum_rejected_with_422() -> None:
    """An invalid ``backend`` → 422 + Pydantic field-level errors (input feedback)."""
    client = await _started_client(_FakeOptimizer(response=_OK_RESPONSE))
    try:
        resp = await client.post("/optimize", json={"backend": "torch"})
        assert resp.status == 422
        errors = await resp.json()
        # Pydantic emits a list of field-error objects; locate the backend one.
        assert isinstance(errors, list)
        assert any("backend" in str(e.get("loc", "")) for e in errors)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_optimize_configuration_error_returns_400() -> None:
    """A caller bug (e.g. backend=textgrad) is a surfaced 400, NOT a 500."""
    fake = _FakeOptimizer(raises=ConfigurationError("textgrad deferred (torch); use dspy-gepa"))
    client = await _started_client(fake)
    try:
        resp = await client.post("/optimize", json={"backend": "textgrad"})
        assert resp.status == 400
        assert "textgrad" in await resp.text()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_optimize_generic_engine_failure_returns_500_generic() -> None:
    """Any non-Configuration failure → generic 500; never a stack trace / path / DB error."""
    fake = _FakeOptimizer(raises=RuntimeError("boom: secret DB connection string"))
    client = await _started_client(fake)
    try:
        resp = await client.post("/optimize", json={})
        assert resp.status == 500
        body = await resp.text()
        assert body == "Something went wrong"
        # Security: the internal detail must NOT leak to the client.
        assert "DB connection" not in body
        assert "boom" not in body
    finally:
        await client.close()
