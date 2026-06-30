"""SI-4 — cross-process wire smoke for the no-DinD runner (Phase 3b/c).

Promoted from ``scripts/smoke_runner_wire.py`` (which was mis-homed as a script).
This is the ONE wire gap the unit suites do not cover: ``RunnerClient`` (httpx)
talking to the REAL ``runner_server`` (aiohttp) over actual HTTP, in SEPARATE
processes — the way the worker talks to the runner in compose.
``test_runner_client`` mocks httpx's transport; ``test_runner_server`` uses an
in-process aiohttp TestClient. Neither exercises the two ends agreeing on the
wire contract with REAL subprocess execution.

No Docker, no LLM API key. A module-scoped fixture starts the server on an
ephemeral port + waits for ``/health``, then three independent tests assert the
contract (success / script-failure / timeout) through a fresh ``RunnerClient``.

Run (opt-in, excluded from the default ``not e2e`` gate)::

    python -m pytest tests/test_e2e/test_runner_wire.py -v -m e2e
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from src.sandbox.runner_client import RunnerClient

# tests/test_e2e/test_runner_wire.py → parents[2] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER = REPO_ROOT / "src" / "sandbox" / "runner_server.py"

# Class-level: this is an E2E test (excluded from the default ``not e2e`` gate).
pytestmark = pytest.mark.e2e


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _wait_for_health(base_url: str, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_exc: Exception | None = None
    async with httpx.AsyncClient(timeout=2.0) as client:
        while time.monotonic() < deadline:
            try:
                if (await client.get(f"{base_url}/health")).status_code == 200:
                    return
            except httpx.HTTPError as exc:
                last_exc = exc
            await asyncio.sleep(0.2)
    raise RuntimeError(f"runner server never became healthy: {last_exc}")


@pytest.fixture(scope="module")
def runner_base_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Start the REAL runner_server subprocess, yield its base URL, tear down.

    Synchronous on purpose: process management belongs outside the event loop,
    and only the health-wait needs a (one-shot ``asyncio.run``) loop. A fresh
    ``RunnerClient`` is constructed per test (it is stateless), so this fixture
    only owns the shared server process + a session-scoped results tmp dir.
    """
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    results_root = tmp_path_factory.mktemp("runner_wire")
    env = {
        **os.environ,
        "RUNNER_HOST": "127.0.0.1",
        "RUNNER_PORT": str(port),
        "RUNNER_RESULTS_ROOT": str(results_root / "results"),
        "RUNNER_MAX_TIMEOUT_S": "5",
    }
    proc = subprocess.Popen(
        [sys.executable, str(SERVER)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT),
    )
    try:
        asyncio.run(_wait_for_health(base_url))
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _client(base_url: str) -> RunnerClient:
    return RunnerClient(base_url, max_timeout_s=5)


async def test_success_returns_exit_zero_and_captured_stdout(
    runner_base_url: str,
) -> None:
    """A clean script returns success + exit 0 + captured stdout."""
    client = _client(runner_base_url)
    result = await client.execute("print(6 * 7)")
    assert result.success
    assert result.exit_code == 0
    assert result.stdout.strip() == "42"


async def test_script_failure_is_a_result_not_a_raise(
    runner_base_url: str,
) -> None:
    """A non-zero exit is a RESULT (success=False), never an exception."""
    client = _client(runner_base_url)
    result = await client.execute("raise ValueError('boom-wire')")
    assert not result.success
    assert result.exit_code != 0
    assert "boom-wire" in result.stderr


async def test_timeout_is_a_timed_out_result_not_a_raise(
    runner_base_url: str,
) -> None:
    """A runaway script yields a timed_out RESULT, never a hanging exception."""
    client = _client(runner_base_url)
    result = await client.execute("while True:\n    pass", timeout=1)
    assert result.timed_out
    assert not result.success
