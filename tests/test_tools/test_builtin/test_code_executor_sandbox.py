"""Phase 2c (docker) + Phase 3b/c (runner) — code_executor sandbox routing + fallback.

These tests pin the DISPATCHER contract: which collaborator runs, and when an
isolated-sandbox path falls back to the host. The collaborators themselves —
``_run_host_subprocess`` and ``_run_in_sandbox`` (→
``SandboxExecutor.execute_runtime_code``) — are exercised directly elsewhere
(``test_builtin_tools.py`` for host; ``test_sandbox/test_executor.py`` for the
docker/runner isolation), so here they are stubbed to isolate the routing
decision.

Key invariant under test: an isolated run (docker OR runner) that fails at the
SCRIPT level returns its own result and is NEVER re-run on the host; only
``SandboxUnavailable`` (infrastructure) triggers the host fallback. Re-running
untrusted code on the host would defeat the isolation an operator opted into.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.sandbox.executor import SandboxUnavailable
from src.tools.builtin import code_executor as ce


def _sandbox_settings(mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        code_executor_mode=mode,
        code_executor_sandbox_image="turing-toolbox:latest",
        code_executor_sandbox_memory_mb=512,
        code_executor_sandbox_timeout=99,
        code_executor_results_mount="",
        code_executor_sandbox_workdir_dest="/workspace/results",
    )


class _Spies:
    def __init__(self) -> None:
        self.host_called = False
        self.sandbox_called = False
        self.host_timeout: int | None = None
        self.sandbox_timeout: int | None = None
        self.sandbox_mode: str | None = None


def _install(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    *,
    sandbox_returns: str | None = None,
    sandbox_raises: BaseException | None = None,
    host_returns: str = "HOST-OUT",
) -> _Spies:
    """Wire fake settings + stubbed collaborators onto the code_executor module.

    ``mode`` flows through to the stubbed ``_run_in_sandbox`` so tests can assert
    the dispatcher passed the right mode ("docker" or "runner").
    """
    spies = _Spies()
    monkeypatch.setattr(ce, "_tool_sandbox", lambda: _sandbox_settings(mode))
    monkeypatch.setattr(
        ce, "_tool_limits", lambda: SimpleNamespace(code_executor_timeout=42)
    )

    async def _host(code: str, timeout: int) -> str:
        del code
        spies.host_called = True
        spies.host_timeout = timeout
        return host_returns

    async def _sandbox(code: str, timeout: int, m: str) -> str:
        del code
        spies.sandbox_called = True
        spies.sandbox_timeout = timeout
        spies.sandbox_mode = m
        if sandbox_raises is not None:
            raise sandbox_raises
        return sandbox_returns if sandbox_returns is not None else f"{m.upper()}-OUT"

    monkeypatch.setattr(ce, "_run_host_subprocess", _host)
    monkeypatch.setattr(ce, "_run_in_sandbox", _sandbox)
    return spies


@pytest.mark.asyncio
async def test_subprocess_mode_uses_host_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default mode routes to the host subprocess and never touches the sandbox."""
    spies = _install(monkeypatch, "subprocess")
    out = await ce.code_executor("print('x')")
    assert out == "HOST-OUT"
    assert spies.host_called is True
    assert spies.sandbox_called is False


@pytest.mark.asyncio
async def test_docker_mode_uses_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opt-in docker mode routes to the sandbox with mode='docker' and never runs
    on the host."""
    spies = _install(monkeypatch, "docker", sandbox_returns="DOCKER-OUT")
    out = await ce.code_executor("print('x')")
    assert out == "DOCKER-OUT"
    assert spies.sandbox_called is True
    assert spies.sandbox_mode == "docker"
    assert spies.host_called is False


@pytest.mark.asyncio
async def test_docker_mode_falls_back_on_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SandboxUnavailable (docker missing / daemon down / image absent) falls
    back to the host subprocess so a run never hard-fails on a missing daemon."""
    spies = _install(
        monkeypatch,
        "docker",
        sandbox_raises=SandboxUnavailable("no daemon"),
        host_returns="HOST-OUT",
    )
    out = await ce.code_executor("print('x')")
    assert out == "HOST-OUT"
    assert spies.sandbox_called is True
    assert spies.host_called is True  # fell back


@pytest.mark.asyncio
async def test_docker_mode_does_not_fall_back_on_script_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A docker run that RETURNS (script exited non-zero) is NOT re-run on the
    host — the failed result is passed through. This is the isolation invariant
    that distinguishes a script failure from an infrastructure failure."""
    spies = _install(
        monkeypatch, "docker", sandbox_returns="SCRIPT-FAILED-EXIT-1"
    )
    out = await ce.code_executor("raise RuntimeError('boom')")
    assert out == "SCRIPT-FAILED-EXIT-1"
    assert spies.sandbox_called is True
    assert spies.host_called is False  # no fallback for a script failure


@pytest.mark.asyncio
async def test_runner_mode_uses_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opt-in runner mode routes to the remote runner with mode='runner' and never
    runs on the host (Phase 3b/c)."""
    spies = _install(monkeypatch, "runner", sandbox_returns="RUNNER-OUT")
    out = await ce.code_executor("print('x')")
    assert out == "RUNNER-OUT"
    assert spies.sandbox_called is True
    assert spies.sandbox_mode == "runner"
    assert spies.host_called is False


@pytest.mark.asyncio
async def test_runner_mode_falls_back_on_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A down/unreachable runner (SandboxUnavailable) falls back to the host
    subprocess so a run never hard-fails when the runner service is absent."""
    spies = _install(
        monkeypatch,
        "runner",
        sandbox_raises=SandboxUnavailable("runner unreachable"),
        host_returns="HOST-OUT",
    )
    out = await ce.code_executor("print('x')")
    assert out == "HOST-OUT"
    assert spies.sandbox_called is True
    assert spies.host_called is True  # fell back


@pytest.mark.asyncio
async def test_runner_mode_does_not_fall_back_on_script_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A runner run that RETURNS (script exited non-zero) is NOT re-run on the
    host — the failed result is passed through, same isolation invariant as
    docker mode."""
    spies = _install(
        monkeypatch, "runner", sandbox_returns="SCRIPT-FAILED-EXIT-1"
    )
    out = await ce.code_executor("raise RuntimeError('boom')")
    assert out == "SCRIPT-FAILED-EXIT-1"
    assert spies.sandbox_called is True
    assert spies.host_called is False  # no fallback for a script failure


@pytest.mark.asyncio
async def test_timeout_resolves_per_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """``timeout=None`` resolves to the mode-appropriate default:
    ToolLimits.code_executor_timeout (subprocess) or
    ToolSandbox.code_executor_sandbox_timeout (docker / runner)."""
    sp_sub = _install(monkeypatch, "subprocess")
    await ce.code_executor("print('x')")
    assert sp_sub.host_timeout == 42  # ToolLimits default (mocked)

    sp_docker = _install(monkeypatch, "docker")
    await ce.code_executor("print('x')")
    assert sp_docker.sandbox_timeout == 99  # ToolSandbox default (mocked)

    sp_runner = _install(monkeypatch, "runner")
    await ce.code_executor("print('x')")
    assert sp_runner.sandbox_timeout == 99  # same sandbox-timeout default as docker


@pytest.mark.asyncio
async def test_explicit_timeout_overrides_mode_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit timeout wins over the mode default for both isolated modes."""
    sp = _install(monkeypatch, "docker")
    await ce.code_executor("print('x')", timeout=7)
    assert sp.sandbox_timeout == 7


def test_tool_sandbox_returns_settings_group() -> None:
    """The real accessor returns the registered ToolSandboxSettings group."""
    settings = ce._tool_sandbox()
    assert settings.code_executor_mode == "subprocess"  # default off
    assert settings.code_executor_sandbox_image == "turing-toolbox:latest"
