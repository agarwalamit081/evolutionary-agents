"""Lean 4 formal-verification builtin (opt-in, default-OFF; Phase 2 #17).

Lean 4 is a theorem prover / functional language; type-checking Lean code is a
machine-checked verification substrate for goals that demand formal proofs. A
builtin lets the agent verify Lean 4 snippets directly — instead of spending one
of its 3-per-run generated-tool slots or reasoning about proof correctness
unaided.

Default-off (``LEAN4_ENABLED``): when disabled, OR when the ``lean`` binary is
absent, the handler is a no-op that returns a clear ``DISABLED:`` message —
mirroring ``git_clone``. When enabled AND the ``lean`` binary is on PATH, the
handler writes the supplied code to a confined ``TemporaryDirectory`` and runs
``lean`` under a hard ``LEAN4_TIMEOUT_S`` ceiling, so a runaway elaboration can
never hang the worker. The subprocess invocation lives in :func:`_check_with_lean`
so tests can stub the external binary without monkeypatching the asyncio module.

Trust note: Lean tactics/meta-programs EXECUTE during elaboration, so running
``lean`` on agent-supplied code is a code-execution trust equivalent to
``code_executor``. Enable only on a host/runner you trust to run agent code.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger

# Bound how much of lean's output we surface — a verbose proof error can be huge.
_MAX_OUTPUT_CHARS = 4000


def _lean4_settings() -> Any:
    """Resolve the Lean4 settings group lazily.

    Lazy import (mirrors ``git_clone._git_clone_settings``) so a test can patch
    ``src.config.settings.get_settings`` and have the change take effect here.
    """
    from src.config.settings import get_settings

    return get_settings().lean4


async def _check_with_lean(lean_bin: str, code: str, timeout_s: int) -> tuple[int, str]:
    """Run ``lean`` on ``code`` in a confined temp dir.

    Returns ``(returncode, combined_output)`` where the output is stderr (Lean
    prints diagnostics there) falling back to stdout. Raises ``asyncio.TimeoutError``
    if the check exceeds ``timeout_s`` (the temp dir is still cleaned up via the
    context manager; the killed process is awaited).
    """
    with tempfile.TemporaryDirectory(prefix="lean4_runner_") as tmp:
        check_file = Path(tmp) / "check.lean"
        check_file.write_text(code, encoding="utf-8")
        proc = await asyncio.create_subprocess_exec(
            lean_bin,
            str(check_file),
            cwd=tmp,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise

    output = (stderr_b or b"").decode("utf-8", errors="replace").strip()
    if not output:
        output = (stdout_b or b"").decode("utf-8", errors="replace").strip()
    return int(proc.returncode or 0), output


async def lean4_runner(lean_code: str = "") -> str:
    """Type-check Lean 4 code against the local Lean toolchain (opt-in).

    Args:
        lean_code: The Lean 4 source to type-check — a theorem with its proof, a
            definition, or a whole module. Indentation-sensitive; passed verbatim
            (whitespace is NOT collapsed). Empty input is rejected.

    Returns:
        A JSON object ``{"status": "ok"|"error", "returncode": int, "output": str}``
        with the compiler's verdict and (truncated) diagnosis, or a ``DISABLED:``/
        ``ERROR:`` string when the feature is off, the binary is missing, the input
        is empty, or the check times out.
    """
    settings = _lean4_settings()
    if not settings.enabled:
        return (
            "DISABLED: lean4_runner is off (LEAN4_ENABLED=false). Ask the operator "
            "to enable formal verification before type-checking Lean 4 code."
        )

    lean_bin = shutil.which("lean")
    if not lean_bin:
        logger.warning("lean4_runner: 'lean' binary not found on PATH")
        return (
            "DISABLED: Lean 4 toolchain not found on PATH (no 'lean' binary). "
            "Install Lean 4 + elan (https://lean-lang.org) and retry."
        )

    code = (lean_code or "").strip()
    if not code:
        return "ERROR: empty lean code"

    timeout_s = int(settings.timeout_s)
    logger.info(f"lean4_runner: type-checking {len(code)} chars (timeout {timeout_s}s)")
    try:
        returncode, output = await _check_with_lean(lean_bin, code, timeout_s)
    except asyncio.TimeoutError:
        return f"ERROR: lean type-check timed out after {timeout_s}s"
    except Exception as exc:  # noqa: BLE001 — a tool failure must never abort a run
        logger.warning(f"lean4_runner failed: {exc}")
        return f"ERROR: lean type-check failed: {exc}"

    status = "ok" if returncode == 0 else "error"
    return json.dumps(
        {"status": status, "returncode": returncode, "output": output[:_MAX_OUTPUT_CHARS]},
        ensure_ascii=False,
    )


TOOL_DEFINITION = {
    "name": "lean4_runner",
    "handler": lean4_runner,
    "description": (
        "Type-check Lean 4 code (the Lean theorem prover / functional language) "
        "against the local Lean toolchain for FORMAL, machine-checked "
        "verification — distinct from code_executor (run arbitrary code) and "
        "code_validator (lint patterns): Lean checks proofs compile and theorems "
        "hold. Use this when a goal demands a verified proof or a machine-checked "
        "property rather than a heuristic answer. Returns a JSON verdict "
        "{status, returncode, output} with the compiler's diagnosis. Opt-in "
        "(LEAN4_ENABLED) and requires the 'lean' binary; a disabled or "
        "toolchain-absent call returns a DISABLED message instead of running."
    ),
    # Runs an external binary on agent code — verdict may change, so NOT cacheable.
    "cacheable": False,
    "parameters": {
        "type": "object",
        "properties": {
            "lean_code": {
                "type": "string",
                "description": (
                    "The Lean 4 source to type-check — a theorem with its proof, a "
                    "definition, or a whole module. Indentation-sensitive; pass "
                    "verbatim (do not collapse whitespace)."
                ),
            },
        },
        "required": ["lean_code"],
    },
}
