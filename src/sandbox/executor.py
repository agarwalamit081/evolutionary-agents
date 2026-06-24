"""Sandbox executor for safe code execution in Docker or subprocess mode.

Provides isolated execution of code generated during the evolution phase,
with resource limits, timeouts, and network isolation when running in
Docker mode. Falls back to subprocess execution when Docker is unavailable.
"""

from __future__ import annotations

import asyncio
import sys
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


class SandboxUnavailable(Exception):
    """Docker/runner infrastructure is unavailable for isolated execution.

    Raised ONLY for infrastructure problems (docker package missing, daemon
    down, image absent) — NOT for a script that runs and exits non-zero or
    raises. Callers that gate UNVALIDATED runtime code (the ``code_executor``
    builtin) use this to decide their own host-subprocess fallback policy,
    unlike the evolution path which silently degrades to subprocess because its
    code is already statically vetted. See ``execute_runtime_code``.
    """


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
        # Lazily-built RunnerClient for runner mode (Phase 3b/c). Typed Any to
        # avoid a circular top-level import (runner_client imports this module).
        self._runner: Any = None

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
        if self._mode == "runner":
            return await self._run_runner(code, None, effective_timeout)
        return await self._run_subprocess(code, None, effective_timeout)

    async def execute_code_subprocess(
        self,
        code: str,
        timeout: int | None = None,
    ) -> SandboxResult:
        """Execute Python in a host subprocess, ignoring the configured mode.

        Unlike ``execute_code`` — which honors ``evolution_sandbox_mode`` and so
        defaults to a stripped ``python:3.12-slim`` container — this ALWAYS runs
        in a host subprocess via ``sys.executable``. That guarantees the run uses
        the SAME interpreter + site-packages the code is later materialized in
        (see ``ToolGenerator._materialize_handler`` → ``get_materializer_namespace``),
        so a smoke test cannot false-reject code whose only "offense" is importing
        an allowlisted third-party dep (httpx/loguru/pandas/…) present in the host
        env but absent from ``python:3.12-slim``. Observed live on q06: every
        spawned tool (sha256_file / pytest_output_parser / spawn_subagent) failed
        its self-test with ModuleNotFoundError / write-to-read-only-FS, so the
        agent fell back to slower inline code.

        Why this is safe: callers only reach this AFTER the static
        ``SafetyPipeline`` has vetted the code (allowlisted imports, no
        os/sys/subprocess/socket/eval/exec, complexity ≤ 20). This method is a
        FUNCTIONAL smoke test, not a security barrier — and
        ``_materialize_handler`` runs the identical code in-process in the host
        namespace immediately afterward anyway. So the security boundary is the
        static pipeline, which still runs first; this only matches the env that
        materialization already trusts.
        """
        effective_timeout = timeout if timeout is not None else self._default_timeout
        return await self._run_subprocess(code, None, effective_timeout)

    async def execute_runtime_code(
        self,
        code: str,
        timeout: int | None = None,
        *,
        workdir: str,
        workdir_dest: str,
    ) -> SandboxResult:
        """Run UNVALIDATED runtime one-off code in docker isolation (Phase 2c).

        This is the entry point the ``code_executor`` builtin uses to close the
        T2-high sandbox-bypass gap: untrusted LLM-generated one-off scripts run
        with network disabled, a read-only rootfs, a memory cap, and a writable
        ``workdir`` mount so ``results/<file>`` deliverables persist to disk.

        Distinct from ``execute_code`` in TWO ways:
        1. It mounts the host ``workdir`` read-write at ``workdir_dest`` and runs
           with ``working_dir`` set to that dest's PARENT, so a script's relative
           ``results/<file>`` path resolves to the mounted host results dir — the
           same contract the host ``code_executor`` subprocess honors.
        2. It does NOT silently degrade to a subprocess on a missing daemon
           (``execute_code`` does, which is fine for already-vetted evolution
           code). Instead it RAISES ``SandboxUnavailable`` on any infrastructure
           problem so ``code_executor`` can apply its own fallback policy. A
           script that runs but exits non-zero / raises returns a normal
           (failed) ``SandboxResult`` — it is never re-run on the host.

        Args:
            code: Unvalidated Python source (already wrapped in any bootstrap).
            timeout: Max seconds. Defaults to the executor's configured timeout.
            workdir: Host directory to mount read-write (the agent results dir).
            workdir_dest: Container path where ``workdir`` is mounted. The
                container's ``working_dir`` is set to this path's parent so a
                relative ``results/…`` path resolves into the mount.

        Raises:
            SandboxUnavailable: docker package missing / daemon down / image
                absent — infrastructure only, never a script failure.

        Returns:
            SandboxResult of the isolated run.
        """
        effective_timeout = timeout if timeout is not None else self._default_timeout
        if self._mode == "docker":
            return await self._run_docker(
                code,
                None,
                effective_timeout,
                workdir=workdir,
                workdir_dest=workdir_dest,
                propagate_unavailable=True,
            )
        if self._mode == "runner":
            # The runner uses its OWN configured results dir (the shared
            # turing-workspace volume); the docker workdir/workdir_dest bind-mount
            # concepts do not apply, so they are ignored here.
            return await self._run_runner(
                code, None, effective_timeout, propagate_unavailable=True
            )
        raise SandboxUnavailable(
            f"sandbox mode is {self._mode!r}, not 'docker' or 'runner' — "
            "cannot isolate runtime code"
        )

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
        effective_timeout = timeout if timeout is not None else self._default_timeout
        if self._mode == "docker":
            return await self._run_docker(code, test_code, effective_timeout)
        if self._mode == "runner":
            return await self._run_runner(code, test_code, effective_timeout)
        combined = f"{code}\n\n# --- test ---\n{test_code}"
        return await self._run_subprocess(combined, None, effective_timeout)

    async def execute_with_packages(
        self,
        code: str,
        packages: list[str],
        timeout: int | None = None,
    ) -> SandboxResult:
        """Execute code with pre-approved packages installed in the sandbox.

        Validates all requested packages against SAFE_PIP_PACKAGES before
        execution. In Docker mode, packages are installed inside the container.
        In subprocess mode, a temporary venv is created with the packages.

        Args:
            code: Python source code to execute.
            packages: Package names to install (must be in SAFE_PIP_PACKAGES).
            timeout: Maximum execution time in seconds.

        Returns:
            SandboxResult with execution output.
        """
        from src.tools.dynamic.allowlist import SAFE_PIP_PACKAGES

        # Validate all packages against the allowlist
        invalid = [p for p in packages if p not in SAFE_PIP_PACKAGES]
        if invalid:
            return SandboxResult(
                success=False,
                exit_code=1,
                stdout="",
                stderr=f"Blocked packages not in allowlist: {invalid}",
                duration_seconds=0,
                memory_mb=None,
                timed_out=False,
            )

        effective_timeout = timeout or self._default_timeout

        if self._mode == "docker":
            return await self._run_docker_with_packages(code, packages, effective_timeout)
        if self._mode == "runner":
            # The runner image mirrors the allowlist (incl. these validated
            # packages) 1:1, so they are already importable there — no pip step
            # is needed (unlike docker mode's pip-prepend or subprocess venv).
            # The allowlist was validated above; just run the code remotely.
            return await self._run_runner(code, None, effective_timeout)
        return await self._run_subprocess_with_packages(code, packages, effective_timeout)

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
            import docker.errors as docker_errors

            client = await asyncio.to_thread(docker.from_env)
            try:
                await asyncio.to_thread(client.images.get, self._image)
                logger.debug("Docker image already available: {}", self._image)
            except docker_errors.ImageNotFound:
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

    # ── Runner mode (remote no-DinD container; Phase 3b/c) ────────────

    def _get_runner(self) -> Any:
        """Lazily build the RunnerClient used by runner mode.

        The runner endpoint is process-global config, so when the injected
        settings carry no ``.runner`` (e.g. code_executor's SimpleNamespace, or
        a bare EvolutionSettings) we read the canonical ``Settings.runner``.
        Tests that need to avoid the network inject a ``.runner`` (or patch this
        method).
        """
        if self._runner is None:
            from src.sandbox.runner_client import RunnerClient

            injected = getattr(self._settings, "runner", None)
            if injected is not None and hasattr(injected, "runner_url"):
                self._runner = RunnerClient.from_settings(injected)
            else:
                from src.config import get_settings

                self._runner = RunnerClient.from_settings(get_settings().runner)
        return self._runner

    async def _run_runner(
        self,
        code: str,
        test_script: str | None,
        timeout: int,
        *,
        propagate_unavailable: bool = False,
    ) -> SandboxResult:
        """Run code via the remote no-DinD runner (Phase 3b/c).

        Mirrors ``_run_docker``'s isolation intent (a separate no-creds
        container) but over HTTP — no ``docker.from_env()``, so the worker needs
        NO Docker socket. The runner executes the script as a subprocess in its
        OWN disposable container (network off, no DB creds).

        The ``propagate_unavailable`` policy mirrors ``_run_docker`` EXACTLY:

        - True (the runtime / ``code_executor`` path): RAISE
          :class:`SandboxUnavailable` on any connection problem so
          ``code_executor`` applies its OWN host-subprocess fallback (with the
          results-dir CWD + bootstrap) rather than this executor's stripped
          ``sys.executable`` subprocess.
        - False (the evolution path — already statically SafetyPipeline-vetted
          code): log + degrade to ``_run_subprocess`` so evolution never
          hard-fails on a briefly-down runner.

        A script that runs and exits non-zero / raises is a normal (failed)
        :class:`SandboxResult` in BOTH paths — it is NEVER re-run on the host
        (re-running untrusted code would defeat the isolation).
        """
        client = self._get_runner()
        try:
            return await client.execute(code, timeout=timeout, test_code=test_script)
        except SandboxUnavailable:
            if propagate_unavailable:
                raise
            logger.warning(
                "runner unavailable; degrading to subprocess for vetted code"
            )
            return await self._run_subprocess(code, test_script, timeout)

    # ── Docker mode ───────────────────────────────────────────────────

    async def _run_docker(
        self,
        code: str,
        test_script: str | None,
        timeout: int,
        *,
        workdir: str | None = None,
        workdir_dest: str | None = None,
        propagate_unavailable: bool = False,
    ) -> SandboxResult:
        """Run code inside a Docker container with resource limits.

        Args:
            code/test_script/timeout: as above.
            workdir: optional host dir mounted read-write inside the container
                (the runtime ``code_executor`` path mounts the agent results dir
                so ``results/<file>`` writes persist). ``None`` = no mount (the
                evolution materialization path — handler code under test).
            workdir_dest: container path where ``workdir`` is mounted. The
                container ``working_dir`` is this path's parent so a relative
                ``results/…`` path resolves into the mount.
            propagate_unavailable: when True (runtime path only), raise
                ``SandboxUnavailable`` on infra problems (missing docker package,
                daemon down, image absent) instead of silently degrading to a
                subprocess. Script failures (container ran, non-zero exit) always
                return a normal ``SandboxResult``.
        """
        start = time.monotonic()

        try:
            import docker
            import docker.errors as docker_errors
        except ImportError:
            if propagate_unavailable:
                # Runtime path: let code_executor apply its OWN host fallback
                # (with the results-dir CWD + bootstrap) rather than this
                # executor's stripped sys.executable subprocess.
                raise SandboxUnavailable("docker package is not installed")
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

            # Runtime path: mount the host results dir RW at workdir_dest and run
            # with working_dir = workdir_dest's parent, so a script's relative
            # ``results/<file>`` write lands in the mounted host results dir.
            workdir_parent: str | None = None
            if workdir and workdir_dest:
                workdir_parent = str(Path(workdir_dest).parent)

            def _run_container() -> tuple[int, str, str]:
                client = docker.from_env()
                try:
                    # Use detach=True so we can fetch logs before the container is removed
                    run_kwargs: dict[str, Any] = dict(
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
                    if workdir and workdir_dest:
                        run_kwargs["volumes"][str(Path(workdir).resolve())] = {
                            "bind": workdir_dest,
                            "mode": "rw",
                        }
                        run_kwargs["working_dir"] = workdir_parent
                    container = client.containers.run(**run_kwargs)
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
                except docker_errors.ContainerError as exc:
                    # The container RAN and exited non-zero — that's a script
                    # RESULT, not infrastructure. Surface it as a normal failed
                    # result so the caller never re-runs the code on the host.
                    raw = getattr(exc, "stderr", None) or b""
                    out = (
                        raw.decode("utf-8", errors="replace")
                        if isinstance(raw, bytes)
                        else str(raw)
                    )
                    return (int(getattr(exc, "exit_status", 1)), "", out)
                except Exception as exc:
                    # APIError / ImageNotFound / ConnectionError / daemon-down =
                    # INFRASTRUCTURE. For the runtime path, re-raise so the outer
                    # handler turns it into SandboxUnavailable (→ host fallback,
                    # e.g. self-evolving-agent-toolbox not yet built). For the evolution path
                    # swallow into a failed result (code already vetted).
                    if propagate_unavailable:
                        raise
                    exit_status = getattr(exc, "exit_status", 1)
                    raw = getattr(exc, "stderr", None) or b""
                    out = (
                        raw.decode("utf-8", errors="replace")
                        if isinstance(raw, bytes)
                        else str(raw)
                    )
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

        except docker_errors.DockerException as exc:
            if propagate_unavailable:
                raise SandboxUnavailable(f"docker infrastructure error: {exc}") from exc
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
        except SandboxUnavailable:
            raise
        except Exception as exc:
            if propagate_unavailable:
                raise SandboxUnavailable(f"docker infrastructure error: {exc}") from exc
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

    # ── Package installation modes ────────────────────────────────────

    async def _run_docker_with_packages(
        self,
        code: str,
        packages: list[str],
        timeout: int,
    ) -> SandboxResult:
        """Run code in Docker with pip install of approved packages."""
        # Prepend pip install to the script
        install_line = (
            f"import subprocess; "
            f"subprocess.run(['pip', 'install', '-q'] + {packages}, check=True)\n"
        )
        full_script = install_line + code
        return await self._run_docker(full_script, None, timeout)

    async def _run_subprocess_with_packages(
        self,
        code: str,
        packages: list[str],
        timeout: int,
    ) -> SandboxResult:
        """Run code in a temp venv with pip install of approved packages."""
        start = time.monotonic()
        tmp_dir = tempfile.mkdtemp(prefix="turing_pkg_sandbox_")
        venv_dir = Path(tmp_dir) / "venv"
        script_path = Path(tmp_dir) / "script.py"
        # Operator-configurable via EvolutionSettings (SANDBOX_VENV_CREATE_TIMEOUT
        # / SANDBOX_PACKAGE_INSTALL_TIMEOUT). getattr-with-default mirrors the
        # __init__ pattern so root Settings, EvolutionSettings, and test mocks
        # all resolve.
        venv_create_timeout = getattr(self._settings, "sandbox_venv_create_timeout", 60)
        package_install_timeout = getattr(
            self._settings, "sandbox_package_install_timeout", 120
        )

        try:
            # Create venv from THIS interpreter (never a PATH-resolved "python"),
            # so subprocess-mode execution is deterministic w.r.t. the running env.
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "venv", str(venv_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=venv_create_timeout)

            # Install packages
            pip_path = str(venv_dir / "bin" / "pip")
            proc = await asyncio.create_subprocess_exec(
                pip_path, "install", "-q", *packages,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=package_install_timeout)

            # Write script
            async with aiofiles.open(str(script_path), mode="w") as f:
                await f.write(code)

            # Execute in venv python
            venv_python = str(venv_dir / "bin" / "python")
            proc = await asyncio.create_subprocess_exec(
                venv_python, str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout,
                )
                duration = time.monotonic() - start
                return SandboxResult(
                    success=proc.returncode == 0,
                    exit_code=proc.returncode,
                    stdout=stdout_bytes.decode("utf-8", errors="replace"),
                    stderr=stderr_bytes.decode("utf-8", errors="replace"),
                    duration_seconds=round(duration, 4),
                    memory_mb=None,
                    timed_out=False,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                duration = time.monotonic() - start
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
            logger.error("Error in package sandbox: {}", exc)
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
                # Remove venv and tmp dir
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)
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
                sys.executable,
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
