"""Tests for src.sandbox.executor — SandboxExecutor in subprocess mode."""

from __future__ import annotations

import shutil

import pytest

from src.sandbox.executor import SandboxExecutor, SandboxResult


# ── Helpers ────────────────────────────────────────────────────────────


def _subprocess_settings() -> object:
    """Return a settings-like object configured for subprocess mode."""

    class _Settings:
        evolution_sandbox_mode = "subprocess"
        evolution_sandbox_image = "python:3.12-slim"
        evolution_sandbox_memory_mb = 256
        evolution_sandbox_timeout = 10

    return _Settings()


# ── Subprocess mode tests ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_subprocess_mode_success() -> None:
    """A simple print statement should succeed and capture stdout."""
    executor = SandboxExecutor(_subprocess_settings())
    result = await executor.execute_code('print("hello")')

    assert isinstance(result, SandboxResult)
    assert result.success is True
    assert result.exit_code == 0
    assert "hello" in result.stdout
    assert result.timed_out is False
    assert result.duration_seconds > 0


@pytest.mark.asyncio
async def test_subprocess_mode_exception() -> None:
    """Code that raises an exception should report failure with stderr."""
    executor = SandboxExecutor(_subprocess_settings())
    code = 'raise ValueError("boom")'
    result = await executor.execute_code(code)

    assert result.success is False
    assert result.exit_code != 0
    assert "boom" in result.stderr
    assert result.timed_out is False


@pytest.mark.asyncio
async def test_subprocess_mode_timeout() -> None:
    """An infinite loop with a short timeout should produce a timed-out result."""
    executor = SandboxExecutor(_subprocess_settings())
    code = "while True: pass"
    result = await executor.execute_code(code, timeout=1)

    assert result.success is False
    assert result.timed_out is True
    assert result.duration_seconds >= 0.9


@pytest.mark.asyncio
async def test_subprocess_execute_test() -> None:
    """execute_test should combine code and test_code and run them together."""
    executor = SandboxExecutor(_subprocess_settings())
    code = "def add(a, b):\n    return a + b\n"
    test_code = (
        "assert add(2, 3) == 5, 'expected 5'\n"
        "print('all tests passed')\n"
    )
    result = await executor.execute_test(code, test_code)

    assert result.success is True
    assert "all tests passed" in result.stdout


# ── Docker mode tests ─────────────────────────────────────────────────


def _docker_available() -> bool:
    """Check whether Docker is installed and the daemon is running."""
    return shutil.which("docker") is not None


@pytest.mark.asyncio
@pytest.mark.skipif(
    not _docker_available(),
    reason="Docker is not available in this environment",
)
async def test_docker_mode_executes_code() -> None:
    """When Docker is available, a simple print should succeed inside a container."""

    class _DockerSettings:
        evolution_sandbox_mode = "docker"
        evolution_sandbox_image = "python:3.12-slim"
        evolution_sandbox_memory_mb = 256
        evolution_sandbox_timeout = 30

    executor = SandboxExecutor(_DockerSettings())
    await executor.ensure_image()
    result = await executor.execute_code('print("docker hello")', timeout=60)

    assert result.success is True
    assert "docker hello" in result.stdout


@pytest.mark.asyncio
async def test_ensure_image_does_not_raise() -> None:
    """ensure_image should not raise even when Docker is unavailable."""
    executor = SandboxExecutor(_subprocess_settings())
    # Should silently succeed (subprocess mode skips image pull)
    await executor.ensure_image()

    # Also test with Docker mode but no daemon — should not raise
    class _DockerSettings:
        evolution_sandbox_mode = "docker"
        evolution_sandbox_image = "python:3.12-slim"
        evolution_sandbox_memory_mb = 256
        evolution_sandbox_timeout = 30

    executor_docker = SandboxExecutor(_DockerSettings())
    await executor_docker.ensure_image()


# ── execute_code_subprocess (Finding #2: tool-gen self-test env alignment) ──


@pytest.mark.asyncio
async def test_execute_code_subprocess_forces_subprocess_in_docker_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """execute_code_subprocess must bypass docker even when mode='docker'.

    Finding #2 regression: the tool-gen self-test must run in the host subprocess
    env (which has the allowlisted deps), NOT the stripped ``python:3.12-slim``
    container that is the default mode. Assert the subprocess path is taken and
    the docker path is never entered, and the explicit timeout threads through.
    """

    class _DockerSettings:
        evolution_sandbox_mode = "docker"
        evolution_sandbox_image = "python:3.12-slim"
        evolution_sandbox_memory_mb = 256
        evolution_sandbox_timeout = 10

    executor = SandboxExecutor(_DockerSettings())
    assert executor._mode == "docker"  # sanity: the default really is docker

    captured: dict[str, object] = {}

    async def fake_subprocess(
        code: str, test_script: object, timeout: int
    ) -> SandboxResult:
        captured["code"] = code
        captured["test_script"] = test_script
        captured["timeout"] = timeout
        return SandboxResult(
            success=True,
            exit_code=0,
            stdout="forced-subprocess",
            stderr="",
            duration_seconds=0.0,
            memory_mb=None,
            timed_out=False,
        )

    async def fail_docker(code: str, test_script: object, timeout: int) -> SandboxResult:
        del code, test_script, timeout  # mirrors _run_docker sig; must never be called
        raise AssertionError("execute_code_subprocess must not enter _run_docker")

    monkeypatch.setattr(executor, "_run_subprocess", fake_subprocess)
    monkeypatch.setattr(executor, "_run_docker", fail_docker)

    result = await executor.execute_code_subprocess("print('x')", timeout=7)

    assert captured["code"] == "print('x')"
    # execute_code_subprocess passes no test_script (it is a code-only API).
    assert captured["test_script"] is None
    assert captured["timeout"] == 7
    assert result.success is True
    assert "forced-subprocess" in result.stdout


@pytest.mark.asyncio
async def test_subprocess_runs_in_same_interpreter() -> None:
    """Subprocess execution uses sys.executable, not a PATH-resolved python.

    Per project rule (never assume bare ``python`` points at the venv), the
    subprocess must be THIS interpreter — the same one that materializes the
    handler in-process — so the self-test env matches the materialization env.
    """
    import sys

    executor = SandboxExecutor(_subprocess_settings())
    result = await executor.execute_code_subprocess("import sys; print(sys.executable)")

    assert result.success is True, f"stderr={result.stderr}"
    assert result.stdout.strip() == sys.executable


@pytest.mark.asyncio
async def test_subprocess_can_import_host_only_allowlisted_dep() -> None:
    """Finding #2 regression: an allowlisted dep present in the host venv
    (loguru) imports cleanly under execute_code_subprocess. The default
    ``python:3.12-slim`` docker self-test would raise ModuleNotFoundError and
    false-reject the tool — this is exactly the q06 failure mode.
    """
    executor = SandboxExecutor(_subprocess_settings())
    result = await executor.execute_code_subprocess(
        "from loguru import logger\nprint('loguru-ok')"
    )

    assert result.success is True, f"stderr={result.stderr}"
    assert "loguru-ok" in result.stdout


@pytest.mark.asyncio
async def test_subprocess_writes_tempfile_fixture() -> None:
    """Finding #2 regression: a self-test that writes a file fixture under the
    system temp dir succeeds in subprocess mode. The read-only docker container
    FS (only /tmp via tmpfs) would fail the write and false-reject the tool.
    """
    executor = SandboxExecutor(_subprocess_settings())
    code = (
        "import tempfile, pathlib\n"
        "p = pathlib.Path(tempfile.gettempdir()) / 'turing_selftest_fixture.txt'\n"
        "p.write_text('fixture-ok')\n"
        "print(p.read_text())\n"
    )
    result = await executor.execute_code_subprocess(code)

    assert result.success is True, f"stderr={result.stderr}"
    assert "fixture-ok" in result.stdout
