"""Shared code-validation gate for generated/edited tools (D9).

A generated tool (runtime ``ToolGenerator``) or an operator-edited tool (the D10
PATCH route) must clear the SAME bar before it enters the registry. This module
bundles that bar so the two entry points cannot drift:

1. **Assertion presence** — ``test_code`` must contain at least one ``assert``
   (an ``ast.Assert`` node). A tool with no assertion gives no evidence its
   handler works; previously an empty/``pass``-only test was *skipped* (the
   sandbox step only ran when ``test_code`` was truthy), letting untested code
   register silently.
2. **Lint** — ``ruff check --select F,E9`` (pyflakes undefined-names + syntax
   errors) on the handler+test together. These are genuine bugs, not style, so
   LLM style noise never blocks a valid tool. ``ruff`` is pinned in
   ``requirements.txt`` and is therefore importable at runtime in the worker.
3. **Safety** — the existing 7-layer ``SafetyPipeline.validate`` (syntax,
   dangerous imports, forbidden patterns, …) on the handler.
4. **Sandbox smoke** — an optional functional run of handler+test in the supplied
   ``SandboxExecutor``, preferring the forced host-subprocess runner so the smoke
   matches the in-process materialization env (Finding #2: an allowlisted
   host-installed dep must not be false-rejected by a stripped docker image).

Each gate fails fast; the result carries the first failing reason plus the layer
dicts (for telemetry / the API 422 body).
"""

from __future__ import annotations

import ast
import re
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from src.safety.pipeline import SafetyPipeline
    from src.sandbox.executor import SandboxExecutor, SandboxResult


@dataclass
class ToolCodeValidation:
    """Result of the shared generated-tool code gate."""

    passed: bool
    reason: str = ""
    safety_result: dict[str, Any] = field(default_factory=dict)
    lint_result: dict[str, Any] = field(default_factory=dict)
    sandbox_result: dict[str, Any] = field(default_factory=dict)


# A ruff issue line begins with a rule code (``F821``, ``E999``) or the literal
# ``invalid-syntax`` marker ruff emits for an unparseable file. The ``Found N
# errors.`` summary and the ``-->``/``|`` decoration lines are filtered out.
_RULE_LINE = re.compile(r"^(?:[A-Z]+\d{2,4}\b|invalid-syntax)")


def _has_assert(test_code: str) -> bool:
    """True if ``test_code`` parses and contains at least one ``assert``."""
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.Assert) for node in ast.walk(tree))


def _lint_code(combined: str) -> dict[str, Any]:
    """Run ``ruff check --select F,E9`` on the combined handler+test source.

    Returns a layer-result dict (``passed``/``issues``/optional ``note``). ``F``
    = pyflakes (undefined names, unused imports — real bugs); ``E9`` = syntax
    errors. The combined source is piped via stdin with ``--stdin-filename`` so
    ruff treats it as Python. If ``ruff`` is absent from PATH (a stripped test
    env), the gate is *skipped* (logged) rather than blocking — the assertion,
    safety, and sandbox gates still apply, so this never weakens security below
    the pre-D9 baseline.
    """
    try:
        proc = subprocess.run(  # noqa: S603 — fixed arg list; source via stdin, no shell
            ["ruff", "check", "--select", "F,E9", "--no-cache",
             "--stdin-filename", "generated_tool.py", "-"],
            input=combined,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except FileNotFoundError:
        logger.warning("ruff not found on PATH; skipping lint gate for tool code")
        return {"passed": True, "issues": [], "note": "ruff not found on PATH"}
    except subprocess.TimeoutExpired:
        return {"passed": False, "issues": ["ruff lint timed out (>30s)"]}

    if proc.returncode == 0:
        return {"passed": True, "issues": []}

    detail = (proc.stdout or proc.stderr or "").splitlines()
    issues = [
        line.strip() for line in detail if _RULE_LINE.match(line.strip())
    ][:5]
    return {"passed": False, "issues": issues or ["ruff lint failed"]}


async def _run_sandbox_smoke(
    sandbox: SandboxExecutor, handler_code: str, test_code: str
) -> dict[str, Any]:
    """Run handler+test in the sandbox, preferring the forced host-subprocess
    runner (``execute_code_subprocess``) so the smoke matches the in-process
    materialization env — the same rationale as the generator's former
    ``_run_sandbox_test`` (Finding #2). Falls back to ``execute_code`` for
    back-compat with older/legacy sandbox objects.
    """
    combined = f"{handler_code}\n\n{test_code}"
    runner = getattr(sandbox, "execute_code_subprocess", None) or sandbox.execute_code
    try:
        result: SandboxResult = await runner(combined)
        if result.timed_out:
            return {"passed": False, "issues": ["Tool test timed out in sandbox"]}
        if not result.success:
            stderr_preview = result.stderr[:300] if result.stderr else "unknown"
            return {"passed": False, "issues": [f"Sandbox test failed: {stderr_preview}"]}
        return {"passed": True, "issues": []}
    except Exception as e:
        return {"passed": False, "issues": [f"Sandbox execution error: {e}"]}


async def validate_tool_code(
    *,
    handler_code: str,
    test_code: str,
    tool_name: str,
    safety_pipeline: SafetyPipeline,
    sandbox: SandboxExecutor | None = None,
    allowlisted_modules: set[str] | None = None,
) -> ToolCodeValidation:
    """Run the shared generated-tool code gate (D9).

    Gates run cheapest-first (assertion → lint → safety) so a trivially-bad tool
    fails before the expensive sandbox smoke. Returns a
    :class:`ToolCodeValidation` whose ``reason`` names the first failing gate;
    on success ``passed`` is ``True`` and the layer dicts are populated.
    """
    # Gate 1: test_code present and asserts something.
    if not test_code.strip():
        return ToolCodeValidation(
            passed=False, reason="test_code is required and must not be empty",
        )
    if not _has_assert(test_code):
        return ToolCodeValidation(
            passed=False,
            reason="test_code must contain at least one assert statement",
        )

    # Gate 2: lint the handler+test together (the test references the handler).
    lint_result = _lint_code(f"{handler_code}\n\n{test_code}")
    if not lint_result["passed"]:
        return ToolCodeValidation(
            passed=False,
            reason=f"Lint failed: {'; '.join(lint_result['issues'][:3])}",
            lint_result=lint_result,
        )

    # Gate 3: safety (7-layer) on the handler.
    from src.tools.dynamic.allowlist import ALLOWED_MODULES

    effective_allowlist = (
        allowlisted_modules if allowlisted_modules is not None else set(ALLOWED_MODULES)
    )
    safety_result = await safety_pipeline.validate(
        code=handler_code,
        context={
            "mutation_type": "tool",
            "description": "runtime generated tool",
            "tool_name": tool_name,
        },
        sandbox_executor=None,
        allowlisted_modules=effective_allowlist,
    )
    if not safety_result.get("passed", False):
        issues = safety_result.get("issues", [])
        return ToolCodeValidation(
            passed=False,
            reason=f"Safety validation failed: {'; '.join(issues[:3])}",
            lint_result=lint_result,
            safety_result=safety_result,
        )

    # Gate 4: functional sandbox smoke (optional).
    sandbox_result: dict[str, Any] = {"passed": True, "note": "no sandbox available"}
    if sandbox is not None:
        sandbox_result = await _run_sandbox_smoke(sandbox, handler_code, test_code)
        if not sandbox_result.get("passed", False):
            return ToolCodeValidation(
                passed=False,
                reason=f"Sandbox test failed: "
                       f"{'; '.join(sandbox_result.get('issues', ['unknown'])[:3])}",
                lint_result=lint_result,
                safety_result=safety_result,
                sandbox_result=sandbox_result,
            )

    return ToolCodeValidation(
        passed=True,
        lint_result=lint_result,
        safety_result=safety_result,
        sandbox_result=sandbox_result,
    )
