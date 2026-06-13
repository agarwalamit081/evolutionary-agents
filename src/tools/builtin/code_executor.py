"""Code executor tool — runs Python code in a subprocess."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from loguru import logger

from src.config.settings import get_settings


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


async def code_executor(code: str, timeout: int = 30) -> str:
    """Execute Python code in a subprocess and return the output.

    The subprocess working directory is set to ``settings.agent.results_root``
    so that files created via relative paths (e.g. ``plt.savefig("chart.png")``)
    land in the run's workspace directory (the current working directory) instead
    of the project root.

    Args:
        code: Python source code to execute.
        timeout: Maximum execution time in seconds.

    Returns:
        Stdout + stderr from the execution.
    """
    logger.info(f"Executing code ({len(code)} chars, timeout={timeout}s)")

    # Resolve results directory as subprocess CWD
    results_dir = Path(get_settings().agent.results_root).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

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
            cwd=str(results_dir),
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
        "is the results/ folder, so files written via relative paths persist "
        "there (parent directories are created automatically); for final "
        "deliverables prefer file_writer. A configurable timeout (default 30s) "
        "is enforced: avoid infinite loops, and use http_request/web_scraper "
        "for network access instead of raw sockets (which can hang until the "
        "timeout fires)."
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
                "description": "Maximum execution time in seconds (default: 30).",
                "default": 30,
            },
        },
        "required": ["code"],
    },
}
