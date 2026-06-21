"""Tests for src.sandbox.executor — SandboxExecutor in subprocess mode."""

from __future__ import annotations

import shutil
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from src.sandbox.executor import SandboxExecutor, SandboxResult, SandboxUnavailable


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


# ── execute_runtime_code (Phase 2c — runtime code_executor docker isolation) ──
#
# Module-level fake ``docker`` exception classes so tests can construct instances
# (e.g. _FakeImageNotFound()) and hand them to _install_fake_docker as the
# behavior the fake client.containers.run should exhibit.


class _FakeDockerException(Exception):
    pass


class _FakeImageNotFound(_FakeDockerException):
    pass


class _FakeContainerError(_FakeDockerException):
    def __init__(self, exit_status: int = 1, stderr: bytes = b"") -> None:
        super().__init__("container error")
        self.exit_status = exit_status
        self.stderr = stderr


class _FakeContainer:
    """A container-like object returned by the fake client on the success path."""

    def wait(self) -> dict[str, int]:
        return {"StatusCode": 0}

    def logs(self, stdout: bool = True, stderr: bool = False) -> bytes:
        return b"hello-from-container" if stdout else b""

    def remove(self, force: bool = True) -> None:
        del force


def _docker_settings() -> object:
    """Settings configured for docker mode + the turing-toolbox image."""

    class _Settings:
        evolution_sandbox_mode = "docker"
        evolution_sandbox_image = "turing-toolbox:latest"
        evolution_sandbox_memory_mb = 512
        evolution_sandbox_timeout = 30

    return _Settings()


def _install_fake_docker(
    monkeypatch: pytest.MonkeyPatch, run_behavior: object
) -> dict[str, object]:
    """Install a fake ``docker`` package into ``sys.modules`` for hermetic testing.

    ``run_behavior`` controls ``client.containers.run``:
      - a container-like object (has .wait/.logs/.remove) → success path;
      - an exception INSTANCE → raised as-is (infra or ContainerError).

    Returns the ``capture`` dict whose ``["kwargs"]`` holds the run kwargs.
    """
    errors: Any = types.ModuleType("docker.errors")
    errors.DockerException = _FakeDockerException
    errors.ContainerError = _FakeContainerError
    errors.ImageNotFound = _FakeImageNotFound
    errors.APIError = _FakeDockerException

    capture: dict[str, object] = {"kwargs": None}

    class _FakeContainers:
        def run(self, **kwargs: object) -> _FakeContainer:
            capture["kwargs"] = kwargs
            if isinstance(run_behavior, BaseException):
                raise run_behavior
            assert not isinstance(run_behavior, type), "pass an instance, not a class"
            return run_behavior  # type: ignore[return-value]

    class _FakeClient:
        containers = _FakeContainers()

        def close(self) -> None:
            pass

    docker: Any = types.ModuleType("docker")
    docker.from_env = lambda: _FakeClient()
    docker.errors = errors

    monkeypatch.setitem(sys.modules, "docker", docker)
    monkeypatch.setitem(sys.modules, "docker.errors", errors)
    return capture


@pytest.mark.asyncio
async def test_execute_runtime_code_requires_docker_mode() -> None:
    """execute_runtime_code refuses to run when the executor isn't docker-mode.

    The runtime path gates UNVALIDATED code; silently running it in a host
    subprocess (the evolution path's degrade behavior) would re-open the
    sandbox-bypass gap. It must raise SandboxUnavailable instead.
    """
    executor = SandboxExecutor(_subprocess_settings())  # mode=subprocess
    with pytest.raises(SandboxUnavailable):
        await executor.execute_runtime_code(
            "print('x')", workdir="/tmp", workdir_dest="/workspace/results"
        )


@pytest.mark.asyncio
async def test_execute_runtime_code_mounts_results_rw_and_isolates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The runtime container mounts results/ RW (deliverables persist), the
    script RO, runs network-off + read-only + mem-capped, with working_dir set
    to the mount's parent so a relative ``results/<file>`` resolves.

    Hermetic: a fake ``docker`` package captures the ``containers.run`` kwargs
    instead of touching a real daemon.
    """
    capture = _install_fake_docker(monkeypatch, _FakeContainer())
    executor = SandboxExecutor(_docker_settings())

    workdir = tmp_path / "results"
    workdir.mkdir()
    result = await executor.execute_runtime_code(
        "print('hi')",
        workdir=str(workdir),
        workdir_dest="/workspace/results",
    )

    assert result.success is True
    assert result.exit_code == 0
    assert "hello-from-container" in result.stdout

    kwargs = capture["kwargs"]
    assert kwargs is not None
    volumes = kwargs["volumes"]  # type: ignore[index]
    # results dir mounted RW at /workspace/results so deliverables persist
    assert any(
        str(workdir.resolve()) == src
        and spec["bind"] == "/workspace/results"
        and spec["mode"] == "rw"
        for src, spec in volumes.items()
    ), f"results dir not mounted RW: {volumes}"
    # working_dir is the mount's parent so a relative results/<file> resolves
    assert kwargs["working_dir"] == "/workspace"  # type: ignore[index]
    # isolation invariants
    assert kwargs["network_disabled"] is True  # type: ignore[index]
    assert kwargs["read_only"] is True  # type: ignore[index]
    assert kwargs["mem_limit"] == "512m"  # type: ignore[index]
    assert kwargs["tmpfs"] == {"/tmp": "size=50m"}  # type: ignore[index]
    # the script itself stays read-only
    assert any(spec["mode"] == "ro" for spec in volumes.values()), volumes


@pytest.mark.asyncio
async def test_execute_runtime_code_propagates_image_not_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing turing-toolbox image is INFRASTRUCTURE, not a script failure:
    execute_runtime_code raises SandboxUnavailable so code_executor falls back to
    the host subprocess rather than masking the missing image as a failed run.
    """
    _install_fake_docker(monkeypatch, _FakeImageNotFound())
    executor = SandboxExecutor(_docker_settings())

    with pytest.raises(SandboxUnavailable):
        await executor.execute_runtime_code(
            "print('hi')",
            workdir=str(tmp_path),
            workdir_dest="/workspace/results",
        )


@pytest.mark.asyncio
async def test_execute_runtime_code_returns_result_on_container_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A script that runs but exits non-zero (ContainerError) is a RESULT, not
    infra: execute_runtime_code returns a failed SandboxResult and does NOT
    raise. This is the isolation invariant — a failing script must never be
    re-run on the host (which would defeat the sandbox).
    """
    _install_fake_docker(
        monkeypatch, _FakeContainerError(exit_status=5, stderr=b"script-boom")
    )
    executor = SandboxExecutor(_docker_settings())

    result = await executor.execute_runtime_code(
        "raise RuntimeError('boom')",
        workdir=str(tmp_path),
        workdir_dest="/workspace/results",
    )

    assert result.success is False
    assert result.exit_code == 5


@pytest.mark.asyncio
async def test_execute_code_swallows_image_not_found_without_propagate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for the evolution path: execute_code (propagate_unavailable
    defaults False) with a missing image returns a failed SandboxResult — it
    does NOT raise. Evolution code is already statically vetted, so the swallow-
    and-degrade behavior is correct there (distinct from the runtime path).
    """
    _install_fake_docker(monkeypatch, _FakeImageNotFound())
    executor = SandboxExecutor(_docker_settings())

    result = await executor.execute_code("print('x')")

    assert result.success is False  # swallowed, not raised
    assert not isinstance(result, Exception)
