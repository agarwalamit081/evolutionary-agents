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
