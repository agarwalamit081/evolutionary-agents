"""Code executor tool — runs Python code in a subprocess."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from loguru import logger

from src.config.settings import ToolSandboxSettings, get_settings
from src.sandbox.executor import SandboxUnavailable
from src.tools._paths import _subdir_active, get_active_run_id, project_root, results_root

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


def _active_run_subdir() -> str | None:
    """The run_id to isolate code_executor deliverables under, or ``None``.

    Returns the bound run_id only when per-run subfoldering is on
    (``RESULTS_PER_RUN_SUBDIR`` + a bound run_id); otherwise ``None`` so
    ``_write_bootstrap`` emits the legacy (non-isolating) shim and behavior is
    byte-identical to today for non-run-id paths.
    """
    return get_active_run_id() if _subdir_active() else None


def _write_bootstrap(results_root_abs: str, run_subdir: str | None = None) -> str:
    """Build the shim prepended to every executed script.

    For write/append/exclusive opens (``w``/``a``/``x``) the shim has two
    responsibilities:

    1. Auto-create parent dirs for relative writes (the F8 fix) so a script
       doing ``open("results/sub/x.md", "w")`` without ``os.makedirs`` still
       succeeds instead of leaving the deliverable missing.
    2. When ``results_root_abs`` (the absolute results dir for THIS mode) is
       given, RELOCATE a *bare* relative write — one NOT already under
       ``results/`` — into that dir. The host subprocess runs with
       ``cwd = project_root()`` (the results dir's PARENT, not the results dir
       itself), so without this a script doing
       ``open("vector_db_comparator.py", "w")`` writes straight into the
       project root — polluting the repo, tripping ``ruff check .``, and
       shadowing real modules on ``sys.path`` (#314). Reads, absolute paths,
       and writes already under ``results/`` are left untouched; a relocation
       that would escape ``results/`` (path traversal like ``../x``) falls back
       to the original path, so the shim is never worse than today.

    ``results_root_abs=""`` yields the legacy parent-mkdir-only shim (no
    relocation) — reached only when no results root is known (degenerate; both
    real modes pass a root today).

    ``run_subdir`` (a validated single-component run_id), when set, additionally
    isolates THIS run's deliverables under ``<results_root_abs>/<run_subdir>/``:
    relative writes are stripped of a leading ``results/``/root-name/
    ``run_subdir`` component, namespaced under the subdir, and traversal-guarded
    to stay inside it; relative reads resolve subdir-first with a flat-root
    fallback (a write→read round-trip finds the file; legacy flat data still
    recalls). ``run_subdir=None`` is byte-identical to the legacy relocating
    shim. This closes the flat-write contamination vector: code_executor
    deliverables used to land FLAT under ``results/`` (file_writer subfolders),
    so a prior run's flat file was recalled by a later run via the flat fallback.
    """
    # Absolute-path + read-mode opens are never touched; only relative writes
    # are mkdir'd and (when a results root is given) relocated.
    if not results_root_abs:
        return (
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
    # Escape injected literals so an odd config (backslash/quote) cannot break
    # out of the generated string. POSIX paths/run_ids have neither, but stay safe.
    root_literal = results_root_abs.replace("\\", "\\\\").replace('"', '\\"')
    if run_subdir is None:
        # Legacy relocating shim: a bare relative write (NOT already under
        # results/) is relocated into results_root_abs. Reads/abs-paths untouched.
        return (
            "import builtins as _turing_b, os as _turing_os\n"
            "_turing_open_orig = _turing_b.open\n"
            f'_TURING_RESULTS = "{root_literal}"\n'
            "def _turing_open(p, m='r', *a, **k):\n"
            "    if any(c in str(m) for c in 'wax'):\n"
            "        _s = str(p)\n"
            "        if not _turing_os.path.isabs(_s):\n"
            "            _here = _turing_os.path.abspath(_s)\n"
            "            if not (_here == _TURING_RESULTS or _here.startswith(_TURING_RESULTS + _turing_os.sep)):\n"
            "                _re = _turing_os.path.abspath(_turing_os.path.join(_TURING_RESULTS, _s))\n"
            "                if _re == _TURING_RESULTS or _re.startswith(_TURING_RESULTS + _turing_os.sep):\n"
            "                    _s = _re\n"
            "            _d = _turing_os.path.dirname(_s)\n"
            "            if _d:\n"
            "                _turing_os.makedirs(_d, exist_ok=True)\n"
            "            p = _s\n"
            "    return _turing_open_orig(p, m, *a, **k)\n"
            "_turing_b.open = _turing_open\n"
        )
    # Run-subdir-aware shim: writes AND reads relocate under
    # <results_root_abs>/<run_subdir>/ (traversal-guarded to that cell). Mirrors
    # _paths.normalize (writes) + resolve_existing (reads: subdir-first, flat
    # fallback) so code_executor deliverables isolate per-run and round-trip with
    # file_writer's subfoldered writes. chr(92) keeps the shim backslash-free, so
    # the generated source needs no escaping beyond the root/sub literals above.
    sub_literal = run_subdir.replace("\\", "\\\\").replace('"', '\\"')
    root_name = results_root_abs.rstrip("/").rsplit("/", 1)[-1].lower() or "results"
    strip_literal = repr(tuple(sorted({"results", root_name, run_subdir.lower()})))
    return (
        "import builtins as _turing_b, os as _turing_os\n"
        "_turing_open_orig = _turing_b.open\n"
        f'_TURING_ROOT = "{root_literal}"\n'
        f'_TURING_SUB = "{sub_literal}"\n'
        f"_TURING_STRIP = {strip_literal}\n"
        "def _turing_parts(_p):\n"
        "    _n = str(_p).replace(chr(92), '/').strip('/')\n"
        "    return [_x for _x in _n.split('/') if _x not in ('', '.')]\n"
        "def _turing_open(_p, _m='r', *_a, **_k):\n"
        "    _s = str(_p)\n"
        "    if _turing_os.path.isabs(_s):\n"
        "        if any(_c in str(_m) for _c in 'wax'):\n"
        "            _d = _turing_os.path.dirname(_s)\n"
        "            if _d:\n"
        "                _turing_os.makedirs(_d, exist_ok=True)\n"
        "        return _turing_open_orig(_p, _m, *_a, **_k)\n"
        "    _pp = _turing_parts(_s)\n"
        "    if not _pp:\n"
        "        return _turing_open_orig(_p, _m, *_a, **_k)\n"
        "    while len(_pp) > 1 and _pp[0].lower() in _TURING_STRIP:\n"
        "        _pp = _pp[1:]\n"
        "    if any(_c in str(_m) for _c in 'wax'):\n"
        "        if _pp[0] != _TURING_SUB:\n"
        "            _pp = [_TURING_SUB] + _pp\n"
        "        _tgt = _turing_os.path.abspath(_turing_os.path.join(_TURING_ROOT, *_pp))\n"
        "        _cell = _turing_os.path.join(_TURING_ROOT, _TURING_SUB)\n"
        "        if _tgt == _cell or _tgt.startswith(_cell + _turing_os.sep):\n"
        "            _d = _turing_os.path.dirname(_tgt)\n"
        "            if _d:\n"
        "                _turing_os.makedirs(_d, exist_ok=True)\n"
        "            _p = _tgt\n"
        "    else:\n"
        "        _sub = _turing_os.path.abspath(_turing_os.path.join(_TURING_ROOT, _TURING_SUB, *_pp))\n"
        "        if _turing_os.path.exists(_sub):\n"
        "            _p = _sub\n"
        "        else:\n"
        "            _flat = _turing_os.path.abspath(_turing_os.path.join(_TURING_ROOT, *_pp))\n"
        "            if _flat.startswith(_TURING_ROOT + _turing_os.sep) and _turing_os.path.exists(_flat):\n"
        "                _p = _flat\n"
        "    return _turing_open_orig(_p, _m, *_a, **_k)\n"
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

    # Relocating bootstrap keyed on the host results dir: a bare relative write
    # (``open("x.py", "w")``) lands under results/ instead of the project root.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="turing_exec_", delete=False
    ) as tmp:
        tmp.write(_write_bootstrap(str(results_root()), _active_run_subdir()) + code)
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

    # The relocating bootstrap needs the results dir AS THE SCRIPT SEES IT.
    # Docker mounts results_root at the container path ``workdir_dest`` (CWD is
    # its parent), so relocate bare writes there. The runner executes in its OWN
    # container, but worker and runner share the turing-workspace volume at the
    # SAME path (RESULTS_ROOT), so the worker's resolved results_root() is valid
    # inside the runner — pass it (not "") so a per-run write isolates under
    # results/<run_id>/ and a bare write no longer lands in the volume root.
    bootstrap_root = ts.code_executor_sandbox_workdir_dest if mode == "docker" else str(results_root())
    run_subdir = _active_run_subdir()

    sandbox = SandboxExecutor(
        SimpleNamespace(
            evolution_sandbox_mode=mode,
            evolution_sandbox_image=ts.code_executor_sandbox_image,
            evolution_sandbox_memory_mb=ts.code_executor_sandbox_memory_mb,
            evolution_sandbox_timeout=timeout,
        )
    )
    result = await sandbox.execute_runtime_code(
        _write_bootstrap(bootstrap_root, run_subdir) + code,
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
        "created automatically). A bare relative write like "
        "open('helper.py', 'w') is auto-routed under results/ too, so generated "
        "modules never pollute the project root — prefer an explicit "
        "'results/<file>' path for clarity; for final deliverables prefer "
        "file_writer. A configurable timeout (default 30s) is enforced: avoid "
        "infinite loops, and use http_request/web_scraper for network access "
        "instead of raw sockets (which can hang until the timeout fires)."
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
