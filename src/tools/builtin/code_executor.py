"""Code executor tool — runs Python code in a subprocess."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Optional

from loguru import logger

from src.config.settings import get_settings
from src.tools._paths import project_root

# Execution timeout is operator-configurable via ToolLimitsSettings
# (CODE_EXECUTOR_TIMEOUT). The schema display default below mirrors the settings
# default; enforcement reads settings at call-time via _tool_limits().
_SCHEMA_DEFAULT_TIMEOUT = 30  # mirrors ToolLimitsSettings.code_executor_timeout


def _tool_limits():
    """Call-time accessor — never capture get_settings() at module import."""
    return get_settings().tools


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
    """Execute Python code in a subprocess and return the output.

    The subprocess working directory is the **project root** (parent of
    ``results_root``) — the same root ``file_writer``/``terminal_command`` use —
    so a path like ``results/foo.md`` resolves identically whether written here,
    read via ``file_reader``, or globbed in this script. Write deliverables
    explicitly to ``results/<file>`` (the bootstrap auto-creates parent dirs);
    read existing deliverables as ``results/<file>`` (e.g.
    ``glob('results/*.md')``). Aligning cwd here fixes the double-nest
    (``results/results/*.md``) that previously left scripts finding nothing.

    Args:
        code: Python source code to execute.
        timeout: Maximum execution time in seconds. ``None`` resolves to
            ``CODE_EXECUTOR_TIMEOUT`` (ToolLimitsSettings, default 30).

    Returns:
        Stdout + stderr from the execution.
    """
    if timeout is None:
        timeout = _tool_limits().code_executor_timeout
    logger.info(f"Executing code ({len(code)} chars, timeout={timeout}s)")

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
