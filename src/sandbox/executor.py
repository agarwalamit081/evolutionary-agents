"""Sandbox executor for safe code execution in Docker or subprocess mode.

Provides isolated execution of code generated during the evolution phase,
with resource limits, timeouts, and network isolation when running in
Docker mode. Falls back to subprocess execution when Docker is unavailable.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiofiles
from loguru import logger


@dataclass
class SandboxResult:
    """Result of a sandboxed code execution."""

    success: bool
    exit_code: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    memory_mb: float | None
    timed_out: bool


class SandboxExecutor:
    """Execute code in an isolated sandbox environment.

    Supports two modes:
    - Docker: Full isolation with resource limits, network disabled, read-only filesystem.
    - Subprocess: Lightweight fallback without resource limits.

    The mode is determined by ``settings.evolution_sandbox_mode``.
    """

    def __init__(self, settings: Any) -> None:
        self._settings = settings
        self._mode: str = getattr(settings, "evolution_sandbox_mode", "subprocess")
        self._image: str = getattr(settings, "evolution_sandbox_image", "python:3.12-slim")
        self._memory_mb: int = getattr(settings, "evolution_sandbox_memory_mb", 256)
        self._default_timeout: int = getattr(settings, "evolution_sandbox_timeout", 30)

    # ── Public API ────────────────────────────────────────────────────

    async def execute_code(
        self,
        code: str,
        timeout: int | None = None,
    ) -> SandboxResult:
        """Execute a Python code snippet inside the sandbox.

        Args:
            code: Python source code to execute.
            timeout: Maximum execution time in seconds. Uses the default
                from settings if not provided.

        Returns:
            SandboxResult with stdout, stderr, exit code, and timing info.
        """
        effective_timeout = timeout if timeout is not None else self._default_timeout
        if self._mode == "docker":
            return await self._run_docker(code, None, effective_timeout)
        return await self._run_subprocess(code, None, effective_timeout)

    async def execute_test(
        self,
        code: str,
        test_code: str,
        timeout: int | None = None,
    ) -> SandboxResult:
        """Execute code alongside a test script inside the sandbox.

        The code and test are concatenated into a single script so that
        the test can import/reference symbols defined in ``code``.

        Args:
            code: Python source code under test.
            test_code: Test assertions / pytest-style test code.
            timeout: Maximum execution time in seconds.

        Returns:
            SandboxResult with test execution results.
        """
        combined = f"{code}\n\n# --- test ---\n{test_code}"
        effective_timeout = timeout if timeout is not None else self._default_timeout
        if self._mode == "docker":
            return await self._run_docker(code, test_code, effective_timeout)
        return await self._run_subprocess(combined, None, effective_timeout)

    async def ensure_image(self) -> None:
        """Pull the configured Docker image if it is not available locally.

        Silently succeeds when Docker is not available or when running in
        subprocess mode.
        """
        if self._mode != "docker":
            logger.debug("Sandbox mode is subprocess — skipping image pull")
            return

        try:
            import docker

            client = await asyncio.to_thread(docker.from_env)
            try:
                await asyncio.to_thread(client.images.get, self._image)
                logger.debug("Docker image already available: {}", self._image)
            except docker.errors.ImageNotFound:
                logger.info("Pulling Docker image: {} ...", self._image)
                await asyncio.to_thread(client.images.pull, self._image)
                logger.info("Docker image pulled: {}", self._image)
            finally:
                client.close()
        except Exception as exc:
            logger.warning(
                "Could not verify/pull Docker image {}: {}", self._image, exc
            )

    async def cleanup(self) -> None:
        """Clean up resources held by the executor.

        Currently a no-op; retained for forward compatibility.
        """
        logger.debug("SandboxExecutor cleanup — nothing to clean")

    # ── Docker mode ───────────────────────────────────────────────────

    async def _run_docker(
        self,
        code: str,
        test_script: str | None,
        timeout: int,
    ) -> SandboxResult:
        """Run code inside a Docker container with resource limits."""
        start = time.monotonic()

        try:
            import docker
            import docker.errors
        except ImportError:
            logger.warning(
                "docker package not installed — falling back to subprocess mode"
            )
            return await self._run_subprocess(code, test_script, timeout)

        tmp_dir = tempfile.mkdtemp(prefix="turing_sandbox_")
        script_path = Path(tmp_dir) / "script.py"

        try:
            # Write the script to disk
            if test_script is not None:
                full_script = f"{code}\n\n# --- test ---\n{test_script}"
            else:
                full_script = code

            async with aiofiles.open(str(script_path), mode="w") as f:
                await f.write(full_script)

            container_script_path = "/sandbox/script.py"

            def _run_container() -> tuple[int, str, str]:
                client = docker.from_env()
                try:
                    # Use detach=True so we can fetch logs before the container is removed
                    container = client.containers.run(
                        image=self._image,
                        command=["python", container_script_path],
                        volumes={
                            str(script_path): {
                                "bind": container_script_path,
                                "mode": "ro",
                            }
                        },
                        mem_limit=f"{self._memory_mb}m",
                        network_disabled=True,
                        read_only=True,
                        tmpfs={"/tmp": "size=50m"},
                        stdout=True,
                        stderr=True,
                        detach=True,
                    )
                    # Wait for container to finish
                    result = container.wait()
                    exit_code = result.get("StatusCode", 1)
                    # Fetch logs before removing
                    out_bytes = container.logs(stdout=True, stderr=False) or b""
                    err_bytes = container.logs(stdout=False, stderr=True) or b""
                    # Clean up
                    try:
                        container.remove(force=True)
                    except Exception:
                        pass
                    return (
                        exit_code,
                        out_bytes.decode("utf-8", errors="replace") if isinstance(out_bytes, bytes) else str(out_bytes),
                        err_bytes.decode("utf-8", errors="replace") if isinstance(err_bytes, bytes) else str(err_bytes),
                    )
                except Exception as exc:
                    # Try to extract ContainerError details
                    exit_status = getattr(exc, "exit_status", 1)
                    raw = getattr(exc, "stderr", None) or b""
                    out = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
                    return (exit_status, "", out)
                finally:
                    client.close()

            try:
                exit_code, stdout, stderr = await asyncio.wait_for(
                    asyncio.to_thread(_run_container),
                    timeout=timeout,
                )
                duration = time.monotonic() - start
                return SandboxResult(
                    success=exit_code == 0,
                    exit_code=exit_code,
                    stdout=stdout,
                    stderr=stderr,
                    duration_seconds=round(duration, 4),
                    memory_mb=None,
                    timed_out=False,
                )
            except asyncio.TimeoutError:
                duration = time.monotonic() - start
                logger.warning("Docker sandbox timed out after {:.1f}s", duration)
                return SandboxResult(
                    success=False,
                    exit_code=None,
                    stdout="",
                    stderr=f"Execution timed out after {timeout}s",
                    duration_seconds=round(duration, 4),
                    memory_mb=None,
                    timed_out=True,
                )

        except docker.errors.DockerException as exc:
            duration = time.monotonic() - start
            logger.error("Docker error in sandbox: {}", exc)
            return SandboxResult(
                success=False,
                exit_code=None,
                stdout="",
                stderr=str(exc),
                duration_seconds=round(duration, 4),
                memory_mb=None,
                timed_out=False,
            )
        except Exception as exc:
            duration = time.monotonic() - start
            logger.error("Unexpected error in Docker sandbox: {}", exc)
            return SandboxResult(
                success=False,
                exit_code=None,
                stdout="",
                stderr=str(exc),
                duration_seconds=round(duration, 4),
                memory_mb=None,
                timed_out=False,
            )
        finally:
            # Best-effort cleanup of temp directory
            try:
                script_path.unlink(missing_ok=True)
                Path(tmp_dir).rmdir()
            except OSError:
                pass

    # ── Subprocess mode ───────────────────────────────────────────────

    async def _run_subprocess(
        self,
        code: str,
        test_script: str | None,
        timeout: int,
    ) -> SandboxResult:
        """Run code in a subprocess without resource limits."""
        start = time.monotonic()
        tmp_dir = tempfile.mkdtemp(prefix="turing_sandbox_")
        script_path = Path(tmp_dir) / "script.py"

        try:
            if test_script is not None:
                full_script = f"{code}\n\n# --- test ---\n{test_script}"
            else:
                full_script = code

            async with aiofiles.open(str(script_path), mode="w") as f:
                await f.write(full_script)

            proc = await asyncio.create_subprocess_exec(
                "python",
                str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
                duration = time.monotonic() - start
                stdout_str = stdout_bytes.decode("utf-8", errors="replace")
                stderr_str = stderr_bytes.decode("utf-8", errors="replace")

                return SandboxResult(
                    success=proc.returncode == 0,
                    exit_code=proc.returncode,
                    stdout=stdout_str,
                    stderr=stderr_str,
                    duration_seconds=round(duration, 4),
                    memory_mb=None,
                    timed_out=False,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                duration = time.monotonic() - start
                logger.warning("Subprocess sandbox timed out after {:.1f}s", duration)
                return SandboxResult(
                    success=False,
                    exit_code=None,
                    stdout="",
                    stderr=f"Execution timed out after {timeout}s",
                    duration_seconds=round(duration, 4),
                    memory_mb=None,
                    timed_out=True,
                )

        except Exception as exc:
            duration = time.monotonic() - start
            logger.error("Unexpected error in subprocess sandbox: {}", exc)
            return SandboxResult(
                success=False,
                exit_code=None,
                stdout="",
                stderr=str(exc),
                duration_seconds=round(duration, 4),
                memory_mb=None,
                timed_out=False,
            )
        finally:
            try:
                script_path.unlink(missing_ok=True)
                Path(tmp_dir).rmdir()
            except OSError:
                pass
