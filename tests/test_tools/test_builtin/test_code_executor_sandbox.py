"""Phase 2c — code_executor docker-vs-subprocess routing + fallback policy.

These tests pin the DISPATCHER contract: which collaborator runs, and when the
docker path falls back to the host. The collaborators themselves —
``_run_host_subprocess`` and ``_run_in_docker_sandbox`` (→
``SandboxExecutor.execute_runtime_code``) — are exercised directly elsewhere
(``test_builtin_tools.py`` for host; ``test_sandbox/test_executor.py`` for the
docker isolation), so here they are stubbed to isolate the routing decision.

Key invariant under test: a docker run that fails at the SCRIPT level returns its
own result and is NEVER re-run on the host; only ``SandboxUnavailable``
(infrastructure) triggers the host fallback. Re-running untrusted code on the
host would defeat the isolation an operator opted into.
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
        self.docker_called = False
        self.host_timeout: int | None = None
        self.docker_timeout: int | None = None


def _install(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    *,
    docker_returns: str | None = None,
    docker_raises: BaseException | None = None,
    host_returns: str = "HOST-OUT",
) -> _Spies:
    """Wire fake settings + stubbed collaborators onto the code_executor module."""
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

    async def _docker(code: str, timeout: int) -> str:
        del code
        spies.docker_called = True
        spies.docker_timeout = timeout
        if docker_raises is not None:
            raise docker_raises
        return docker_returns if docker_returns is not None else "DOCKER"

    monkeypatch.setattr(ce, "_run_host_subprocess", _host)
    monkeypatch.setattr(ce, "_run_in_docker_sandbox", _docker)
    return spies


@pytest.mark.asyncio
async def test_subprocess_mode_uses_host_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default mode routes to the host subprocess and never touches the sandbox."""
    spies = _install(monkeypatch, "subprocess")
    out = await ce.code_executor("print('x')")
    assert out == "HOST-OUT"
    assert spies.host_called is True
    assert spies.docker_called is False


@pytest.mark.asyncio
async def test_docker_mode_uses_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opt-in docker mode routes to the sandbox and never runs on the host."""
    spies = _install(monkeypatch, "docker", docker_returns="DOCKER-OUT")
    out = await ce.code_executor("print('x')")
    assert out == "DOCKER-OUT"
    assert spies.docker_called is True
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
        docker_raises=SandboxUnavailable("no daemon"),
        host_returns="HOST-OUT",
    )
    out = await ce.code_executor("print('x')")
    assert out == "HOST-OUT"
    assert spies.docker_called is True
    assert spies.host_called is True  # fell back


@pytest.mark.asyncio
async def test_docker_mode_does_not_fall_back_on_script_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A docker run that RETURNS (script exited non-zero) is NOT re-run on the
    host — the failed result is passed through. This is the isolation invariant
    that distinguishes a script failure from an infrastructure failure."""
    spies = _install(
        monkeypatch, "docker", docker_returns="SCRIPT-FAILED-EXIT-1"
    )
    out = await ce.code_executor("raise RuntimeError('boom')")
    assert out == "SCRIPT-FAILED-EXIT-1"
    assert spies.docker_called is True
    assert spies.host_called is False  # no fallback for a script failure


@pytest.mark.asyncio
async def test_timeout_resolves_per_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """``timeout=None`` resolves to the mode-appropriate default:
    ToolLimits.code_executor_timeout (subprocess) or
    ToolSandbox.code_executor_sandbox_timeout (docker)."""
    sp_sub = _install(monkeypatch, "subprocess")
    await ce.code_executor("print('x')")
    assert sp_sub.host_timeout == 42  # ToolLimits default (mocked)

    sp_docker = _install(monkeypatch, "docker")
    await ce.code_executor("print('x')")
    assert sp_docker.docker_timeout == 99  # ToolSandbox default (mocked)


@pytest.mark.asyncio
async def test_explicit_timeout_overrides_mode_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit timeout wins over the mode default for both modes."""
    sp = _install(monkeypatch, "docker")
    await ce.code_executor("print('x')", timeout=7)
    assert sp.docker_timeout == 7


def test_tool_sandbox_returns_settings_group() -> None:
    """The real accessor returns the registered ToolSandboxSettings group."""
    settings = ce._tool_sandbox()
    assert settings.code_executor_mode == "subprocess"  # default off
    assert settings.code_executor_sandbox_image == "turing-toolbox:latest"
