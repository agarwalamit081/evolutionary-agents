"""Code executor tool — runs Python code in a subprocess."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from loguru import logger


async def code_executor(code: str, timeout: int = 30) -> str:
    """Execute Python code in a subprocess and return the output.

    Args:
        code: Python source code to execute.
        timeout: Maximum execution time in seconds.

    Returns:
        Stdout + stderr from the execution.
    """
    logger.info(f"Executing code ({len(code)} chars, timeout={timeout}s)")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="turing_exec_", delete=False
    ) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        proc = await asyncio.create_subprocess_exec(
            "python", tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
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
        "Execute Python code in a subprocess and return stdout/stderr. "
        "Use for running calculations, data processing, or testing code snippets. "
        "Code runs in an isolated process with a configurable timeout."
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
