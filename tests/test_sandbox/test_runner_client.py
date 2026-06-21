"""Tests for ``src.sandbox.runner_client.RunnerClient`` (Phase 3b/c).

The client is the wire protocol between the worker (caller side) and the remote
no-DinD runner container. These tests pin that contract with ``httpx.MockTransport``
(no real socket): a successful run maps to a ``SandboxResult``; a script failure
or timeout is a normal (failed) result, never raised; and any infrastructure
problem (runner down / 5xx / non-JSON / malformed payload) raises
``SandboxUnavailable`` so callers apply their OWN fallback — mirroring the
docker-mode ``_run_docker`` contract exactly.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from src.sandbox.executor import SandboxResult, SandboxUnavailable
from src.sandbox.runner_client import RunnerClient


def _client(handler: Any, **kw: Any) -> RunnerClient:
    """Build a RunnerClient wired to a MockTransport running ``handler``."""
    return RunnerClient(
        "http://runner:8090/", transport=httpx.MockTransport(handler), **kw
    )


def _ok_payload(stdout: str = "hello\n") -> dict[str, Any]:
    return {
        "success": True,
        "exit_code": 0,
        "stdout": stdout,
        "stderr": "",
        "duration_seconds": 0.12,
        "timed_out": False,
    }


@pytest.mark.asyncio
async def test_execute_success_maps_to_sandbox_result() -> None:
    """A 200 + valid payload → SandboxResult with the runner's fields."""
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/execute"
        body = json.loads(request.content)
        assert body["code"] == "print('hi')"
        assert body["timeout"] == 5
        return httpx.Response(200, json=_ok_payload("hi\n"))

    result = await _client(handler).execute("print('hi')", timeout=5)

    assert isinstance(result, SandboxResult)
    assert result.success is True
    assert result.exit_code == 0
    assert result.stdout == "hi\n"
    assert result.timed_out is False
    assert result.memory_mb is None


@pytest.mark.asyncio
async def test_execute_script_failure_returns_failed_result_not_raised() -> None:
    """A non-zero exit is the script's RESULT — surfaced, never raised."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "success": False,
                "exit_code": 1,
                "stdout": "",
                "stderr": "ZeroDivisionError",
                "duration_seconds": 0.05,
                "timed_out": False,
            },
        )

    result = await _client(handler).execute("1/0")

    assert result.success is False
    assert result.exit_code == 1
    assert "ZeroDivisionError" in result.stderr


@pytest.mark.asyncio
async def test_execute_subprocess_timeout_returns_timed_out_result() -> None:
    """A run that the runner killed for time is a timed_out result, not raised."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "success": False,
                "exit_code": None,
                "stdout": "",
                "stderr": "Execution timed out after 2s",
                "duration_seconds": 2.0,
                "timed_out": True,
            },
        )

    result = await _client(handler).execute("while True: pass", timeout=2)

    assert result.timed_out is True
    assert result.success is False


@pytest.mark.asyncio
async def test_execute_connect_error_raises_sandbox_unavailable() -> None:
    """A down runner (connection refused) → SandboxUnavailable for fallback."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(SandboxUnavailable, match="runner unreachable"):
        await _client(handler).execute("print(1)")


@pytest.mark.asyncio
async def test_execute_5xx_raises_sandbox_unavailable() -> None:
    """A runner-side error (5xx) is infrastructure → SandboxUnavailable."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    with pytest.raises(SandboxUnavailable, match="HTTP 500"):
        await _client(handler).execute("print(1)")


@pytest.mark.asyncio
async def test_execute_non_json_body_raises_sandbox_unavailable() -> None:
    """A 200 with a non-JSON body → SandboxUnavailable (malformed transport)."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<<not json>>")

    with pytest.raises(SandboxUnavailable, match="non-JSON"):
        await _client(handler).execute("print(1)")


@pytest.mark.asyncio
async def test_execute_malformed_payload_raises_sandbox_unavailable() -> None:
    """A 200 JSON payload with a non-int exit_code → SandboxUnavailable."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"exit_code": "not-an-int"})

    with pytest.raises(SandboxUnavailable, match="malformed payload"):
        await _client(handler).execute("print(1)")


@pytest.mark.asyncio
async def test_execute_caps_timeout_to_max() -> None:
    """A requested timeout above max_timeout_s is clamped (no hour-long holds)."""
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["timeout"] = json.loads(request.content)["timeout"]
        return httpx.Response(200, json=_ok_payload())

    await _client(handler, max_timeout_s=10).execute("print(1)", timeout=9999)

    assert seen["timeout"] == 10


@pytest.mark.asyncio
async def test_execute_includes_test_code_in_payload_when_given() -> None:
    """``test_code`` flows into the payload so the runner can append it."""
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=_ok_payload())

    await _client(handler).execute("x = 1", test_code="assert x == 1")

    assert seen["test_code"] == "assert x == 1"


def test_from_settings_reads_runner_group() -> None:
    """``from_settings`` reads RunnerSettings off a root Settings or the group."""
    from types import SimpleNamespace

    settings = SimpleNamespace(
        runner=SimpleNamespace(
            runner_url="http://r:1234/",
            runner_connect_timeout_s=2.0,
            runner_max_timeout_s=42,
        )
    )

    client = RunnerClient.from_settings(settings)

    assert client._base_url == "http://r:1234"  # trailing slash stripped
    assert client._connect_timeout_s == 2.0
    assert client._max_timeout_s == 42


def test_from_settings_reads_root_runner_settings_directly() -> None:
    """A bare RunnerSettings-like object (no ``.runner`` attr) also works."""
    from types import SimpleNamespace

    runner_settings = SimpleNamespace(
        runner_url="http://bare:9",
        runner_connect_timeout_s=1.0,
        runner_max_timeout_s=5,
    )

    client = RunnerClient.from_settings(runner_settings)

    assert client._base_url == "http://bare:9"
    assert client._max_timeout_s == 5
