"""Tests for ``src.sandbox.runner_server`` (Phase 3b/c) — the runner container.

These exercise the REAL aiohttp app over a ``TestClient`` and run REAL python
subprocesses (trivial, deterministic ones) — the honest way, mirroring how
``tests/test_sandbox/test_executor.py::test_subprocess_mode_success`` grounds
the executor. No mocks of the system under test; only the worker side of the
wire is mocked elsewhere (``test_runner_client.py``).

Together with the client tests, these pin the wire contract from BOTH ends, so a
shape drift between ``runner_client._parse`` and ``handle_execute`` is caught.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from src.sandbox.runner_server import RunnerServerSettings, build_app


@pytest.fixture
async def client(tmp_path: Path) -> AsyncIterator[TestClient]:
    """A live TestClient backed by the real app, results_root under tmp_path.

    ``runner_max_timeout_s=10`` keeps the timeout test snappy while still real.
    """
    settings = RunnerServerSettings(
        runner_results_root=str(tmp_path / "results"),
        runner_max_timeout_s=10,
    )
    app = build_app(settings)
    server = TestServer(app)
    test_client = TestClient(server)
    await test_client.start_server()
    yield test_client
    await test_client.close()


@pytest.mark.asyncio
async def test_health_returns_ok(client: TestClient) -> None:
    resp = await client.get("/health")
    assert resp.status == 200
    assert await resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_execute_runs_code_and_returns_sandbox_result_shape(client: TestClient) -> None:
    resp = await client.post("/execute", json={"code": "print(1 + 1)", "timeout": 10})

    assert resp.status == 200
    data = await resp.json()
    # Exact SandboxResult shape the client's _parse() expects.
    assert set(data) == {
        "success", "exit_code", "stdout", "stderr",
        "duration_seconds", "timed_out",
    }
    assert data["success"] is True
    assert data["exit_code"] == 0
    assert data["stdout"].strip() == "2"
    assert data["timed_out"] is False


@pytest.mark.asyncio
async def test_execute_script_failure_surfaces_nonzero_exit(client: TestClient) -> None:
    resp = await client.post(
        "/execute", json={"code": "raise ValueError('boom')", "timeout": 10}
    )

    data = await resp.json()
    assert data["success"] is False
    assert data["exit_code"] != 0
    assert "boom" in data["stderr"]


@pytest.mark.asyncio
async def test_execute_writes_results_to_shared_volume_cwd(
    client: TestClient, tmp_path: Path
) -> None:
    """CWD = results_root parent → a relative results/<file> write lands on disk."""
    resp = await client.post(
        "/execute",
        json={"code": "open('results/out.txt', 'w').write('hi')", "timeout": 10},
    )

    data = await resp.json()
    assert data["success"] is True
    assert (tmp_path / "results" / "out.txt").read_text() == "hi"


@pytest.mark.asyncio
async def test_execute_subprocess_timeout_returns_timed_out_result(
    client: TestClient
) -> None:
    """An infinite loop killed at the timeout is a timed_out result, not a 500."""
    resp = await client.post(
        "/execute", json={"code": "while True:\n    pass", "timeout": 1}
    )

    assert resp.status == 200  # a timeout is a RESULT, not an infrastructure error
    data = await resp.json()
    assert data["timed_out"] is True
    assert data["success"] is False
    assert data["exit_code"] is None


@pytest.mark.asyncio
async def test_execute_appends_test_code(client: TestClient) -> None:
    """test_code is concatenated after the code so it can reference its symbols."""
    resp = await client.post(
        "/execute",
        json={"code": "x = 21", "test_code": "print(x * 2)", "timeout": 10},
    )

    data = await resp.json()
    assert data["success"] is True
    assert data["stdout"].strip() == "42"


@pytest.mark.asyncio
async def test_execute_caps_timeout_to_server_max(tmp_path: Path) -> None:
    """A requested timeout above runner_max_timeout_s is clamped server-side."""
    settings = RunnerServerSettings(
        runner_results_root=str(tmp_path / "results"),
        runner_max_timeout_s=2,
    )
    app = build_app(settings)
    server = TestServer(app)
    test_client = TestClient(server)
    await test_client.start_server()
    try:
        # 9999s would hang the test for ages if not clamped to 2s; the infinite
        # loop then hits the 2s cap and returns a timed_out result promptly.
        resp = await test_client.post(
            "/execute", json={"code": "while True:\n    pass", "timeout": 9999}
        )
        data = await resp.json()
        assert data["timed_out"] is True
        assert data["duration_seconds"] < 10  # proved it did NOT wait 9999s
    finally:
        await test_client.close()


@pytest.mark.asyncio
async def test_execute_bad_json_returns_400(client: TestClient) -> None:
    resp = await client.post(
        "/execute", data="<<not json>>", headers={"Content-Type": "application/json"}
    )
    assert resp.status == 400


@pytest.mark.asyncio
async def test_execute_missing_code_returns_400(client: TestClient) -> None:
    resp = await client.post("/execute", json={"timeout": 5})
    assert resp.status == 400


@pytest.mark.asyncio
async def test_execute_non_string_code_returns_400(client: TestClient) -> None:
    resp = await client.post("/execute", json={"code": 123})
    assert resp.status == 400


def test_build_app_registers_both_routes() -> None:
    """The app wires /execute (POST) and /health (GET)."""
    app = build_app(RunnerServerSettings())

    routes = {(r.method, r.resource.canonical) for r in app.router.routes()}
    assert ("POST", "/execute") in routes
    assert ("GET", "/health") in routes
