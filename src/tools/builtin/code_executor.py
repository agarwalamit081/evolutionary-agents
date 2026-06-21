"""Code executor tool — runs Python code in a subprocess."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from loguru import logger

from src.config.settings import ToolSandboxSettings, get_settings
from src.sandbox.executor import SandboxUnavailable
from src.tools._paths import project_root, results_root

if TYPE_CHECKING:
    from src.sandbox.executor import SandboxResult

# Execution timeout is operator-configurable via ToolLimitsSettings
# (CODE_EXECUTOR_TIMEOUT). The schema display default below mirrors the settings
# default; enforcement reads settings at call-time via _tool_limits().
_SCHEMA_DEFAULT_TIMEOUT = 30  # mirrors ToolLimitsSettings.code_executor_timeout


def _tool_limits():
    """Call-time accessor — never capture get_settings() at module import."""
    return get_settings().tools


def _tool_sandbox() -> ToolSandboxSettings:
    """Call-time accessor for the runtime code-exec sandbox settings (Phase 2c)."""
    return get_settings().tool_sandbox


# Prepended to every executed script. The subprocess CWD is already the results
# directory, so relative-path writes persist — but a generator script that does
# ``open("design_patterns/x.md", "w")`` without first ``os.makedirs``-ing the
# subdir fails silently, leaving deliverables missing (the F8 gap). This shim
# auto-creates parent directories for relative write/append/exclusive paths so
# such scripts succeed. Absolute paths and read modes are left untouched.
_WRITE_BOOTSTRAP = (
    "import builtins as _turing_b, os as _turing_os\n"
    "_turing_open_orig = _turing_b.open\n"
    "def _turing_open(p, m='r', *a, **k):\n"
    "    if any(c in str(m) for c in 'wax'):\n"
    "        _d = _turing_os.path.dirname(str(p))\n"
    "        if _d and not _turing_os.path.isabs(str(p)):\n"
    "            _turing_os.makedirs(_d, exist_ok=True)\n"
    "    return _turing_open_orig(p, m, *a, **k)\n"
    "_turing_b.open = _turing_open\n"
)


async def code_executor(code: str, timeout: Optional[int] = None) -> str:
    """Execute Python code and return the output.

    Three execution modes (Phase 2c docker; Phase 3b/c runner):

    - **subprocess** (default): runs in a host subprocess with CWD = project
      root (parent of ``results_root``) — the same root ``file_writer``/
      ``terminal_command`` use — so a path like ``results/foo.md`` resolves
      identically whether written here, read via ``file_reader``, or globbed in
      this script. Write deliverables explicitly to ``results/<file>`` (the
      bootstrap auto-creates parent dirs); read existing deliverables as
      ``results/<file>`` (e.g. ``glob('results/*.md')``).
    - **docker** (opt-in via ``CODE_EXECUTOR_MODE=docker``): runs the SAME code
      in an isolated container — network disabled, read-only rootfs, a memory
      cap — with the agent results dir mounted read-write so ``results/<file>``
      deliverables still persist. Closes the T2-high sandbox-bypass gap: the
      host subprocess ran untrusted one-off LLM code with full host access.
      Docker mode keeps the ``results/<file>`` contract (only ``results/`` is
      writable inside the container). If Docker is unavailable it logs a WARNING
      and falls back to the host subprocess so a run never hard-fails.
    - **runner** (opt-in via ``CODE_EXECUTOR_MODE=runner``, Phase 3b/c): POSTs
      the code to the remote no-DinD runner container over HTTP. The worker
      needs NO Docker socket (no Docker-in-Docker); the runner executes the
      script in its OWN isolated container (network off, no DB/Redis creds),
      writing ``results/<file>`` deliverables to the shared turing-workspace
      volume. Like docker mode, it falls back to the host subprocess if the
      runner is unreachable, so a run never hard-fails.

    Args:
        code: Python source code to execute.
        timeout: Maximum execution time in seconds. ``None`` resolves to
            ``CODE_EXECUTOR_TIMEOUT`` (subprocess) or
            ``CODE_EXECUTOR_SANDBOX_TIMEOUT`` (docker / runner).

    Returns:
        Stdout + stderr from the execution.
    """
    ts = _tool_sandbox()
    mode = ts.code_executor_mode
    if timeout is None:
        timeout = (
            ts.code_executor_sandbox_timeout
            if mode in ("docker", "runner")
            else _tool_limits().code_executor_timeout
        )
    logger.info(
        "Executing code ({} chars, mode={}, timeout={}s)",
        len(code), mode, timeout,
    )

    if mode in ("docker", "runner"):
        try:
            return await _run_in_sandbox(code, timeout, mode)
        except SandboxUnavailable as exc:
            # Infrastructure-only (docker missing / daemon down / image absent /
            # runner down). A script that ran + failed does NOT take this branch
            # — it returns its own result and is never re-run on the host.
            logger.warning(
                "{} code_executor sandbox unavailable ({}); "
                "falling back to host subprocess",
                mode, exc,
            )
    return await _run_host_subprocess(code, timeout)


async def _run_host_subprocess(code: str, timeout: int) -> str:
    """Execute Python in a host subprocess with CWD = project root.

    The default, non-isolated path — also the fallback when the docker sandbox
    is unavailable. Relative ``results/<file>`` writes persist (the bootstrap
    auto-creates parent dirs).
    """
    # Subprocess CWD = project root (parent of results_root), shared with
    # file_writer/terminal_command so ``results/<file>`` resolves uniformly.
    cwd_dir = project_root()
    cwd_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="turing_exec_", delete=False
    ) as tmp:
        tmp.write(_WRITE_BOOTSTRAP + code)
        tmp_path = tmp.name

    try:
        proc = await asyncio.create_subprocess_exec(
            "python", tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd_dir),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)

        output = ""
        if stdout:
            output += stdout.decode("utf-8", errors="replace")
        if stderr:
            output += f"\nSTDERR:\n{stderr.decode('utf-8', errors='replace')}"

        if proc.returncode != 0:
            output += f"\nExit code: {proc.returncode}"

        return output or "(no output)"

    except asyncio.TimeoutError:
        logger.warning(f"Code execution timed out after {timeout}s")
        return f"ERROR: Execution timed out after {timeout} seconds"
    except Exception as exc:
        return f"ERROR: {exc}"
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _format_sandbox_output(result: SandboxResult) -> str:
    """Render a ``SandboxResult`` in the same shape as the host-subprocess output."""
    output = ""
    if result.stdout:
        output += result.stdout
    if result.stderr:
        output += f"\nSTDERR:\n{result.stderr}"
    if result.timed_out:
        output += f"\nERROR: Execution timed out after {result.duration_seconds}s"
    elif result.exit_code is not None and result.exit_code != 0:
        output += f"\nExit code: {result.exit_code}"
    return output or "(no output)"


async def _run_in_sandbox(code: str, timeout: int, mode: str) -> str:
    """Execute Python in the isolated sandbox — a docker container OR the remote
    runner (Phase 3b/c).

    Both modes isolate untrusted one-off LLM code:

    - **docker** (``mode="docker"``): network-off, read-only rootfs, memory cap,
      with the agent results dir bind-mounted read-write so ``results/<file>``
      writes persist.
    - **runner** (``mode="runner"``): POSTs to the remote no-DinD runner
      container (network off, no DB/Redis creds) over HTTP. The runner writes to
      its OWN results dir under the shared turing-workspace volume, so the
      docker workdir/workdir_dest bind-mount concepts are ignored by
      ``execute_runtime_code`` in this mode.

    The write-bootstrap is prepended in both so relative ``results/<file>``
    parent dirs are auto-created (docker via the bind mount; runner via the
    shared volume).

    Raises ``SandboxUnavailable`` on infrastructure problems (docker missing /
    daemon down / image absent / runner down) so ``code_executor`` can fall back
    to the host subprocess. A script that runs but exits non-zero / raises
    returns a normal formatted result — it is NEVER re-run on the host (that
    would defeat the isolation an operator opted into).
    """
    from types import SimpleNamespace

    from src.sandbox.executor import SandboxExecutor

    ts = _tool_sandbox()
    mount_src = ts.code_executor_results_mount or str(results_root())
    # Ensure the host mount target exists so Docker can bind it and a script
    # writing results/<file> has somewhere to land. (For runner mode the runner
    # writes to its OWN results dir under the shared volume; this host dir is
    # the bind source docker mode uses — harmless to ensure-exist for runner.)
    Path(mount_src).mkdir(parents=True, exist_ok=True)

    sandbox = SandboxExecutor(
        SimpleNamespace(
            evolution_sandbox_mode=mode,
            evolution_sandbox_image=ts.code_executor_sandbox_image,
            evolution_sandbox_memory_mb=ts.code_executor_sandbox_memory_mb,
            evolution_sandbox_timeout=timeout,
        )
    )
    result = await sandbox.execute_runtime_code(
        _WRITE_BOOTSTRAP + code,
        timeout=timeout,
        workdir=mount_src,
        workdir_dest=ts.code_executor_sandbox_workdir_dest,
    )
    return _format_sandbox_output(result)


# Tool definition for registry
TOOL_DEFINITION = {
    "name": "code_executor",
    "handler": code_executor,
    "description": (
        "Execute Python code in a subprocess for one-off calculations, quick "
        "data transformations, and testing code snippets. The working directory "
        "is the PROJECT ROOT, so reference deliverables by their results/ path "
        "(e.g. glob('results/*.md'), open('results/out.md')). Write any scratch "
        "or generated files explicitly under results/ (parent directories are "
        "created automatically); for final deliverables prefer file_writer. A "
        "configurable timeout (default 30s) is enforced: avoid infinite loops, "
        "and use http_request/web_scraper for network access instead of raw "
        "sockets (which can hang until the timeout fires)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python source code to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": "Maximum execution time in seconds (default: 30, configurable via CODE_EXECUTOR_TIMEOUT).",
                "default": _SCHEMA_DEFAULT_TIMEOUT,
            },
        },
        "required": ["code"],
    },
}
