"""7-layer safety pipeline for evolution mutations."""

from __future__ import annotations

import ast
import re
from typing import Any

from src.graph.enums import MutationType



# Forbidden patterns that should never appear in generated code
_FORBIDDEN_PATTERNS: list[str] = [
    r"os\.system\s*\(",
    r"subprocess\.call\s*\([^)]*shell\s*=\s*True",
    r"__import__\s*\(\s*['\"]os['\"]\s*\)",
    r"eval\s*\(",
    r"exec\s*\(",
    r"compile\s*\([^)]*['\"]exec['\"]",
    r"open\s*\([^)]*['\"]w['\"].*(?:/etc/passwd|/etc/shadow|\.ssh|\.env)",
    r"rm\s+-rf",
    r"shutil\.rmtree",
    r"pickle\.loads?\s*\(",
    r"marshal\.loads?\s*\(",
    r"netrc",
    r"cred",
]


# Mutation types whose ``mutated_content`` is executable Python source. Only
# these run the AST-dependent safety layers (syntax / imports / semantic);
# prompt/config/memory text would hard-fail ``ast.parse`` and so could never
# deploy — the most common evolution opportunity (prompt refinement) was
# permanently rejected at the syntax layer. Callers that pass no
# ``mutation_type`` (dynamic-tool validation) are treated as code and run every
# layer. New mutation types default to non-code (code is the special case).
_CODE_MUTATIONS: frozenset[MutationType] = frozenset({
    MutationType.CODE,
    MutationType.TOOL,
})


def should_gate_destructive(tool_name: str, registry: Any) -> bool:
    """Decide whether ``tool_name``'s invocation routes through the HITL gate (F3).

    A tool gates when its MCP ``destructiveHint`` is True. The registry is
    duck-typed (``is_destructive(name) -> bool``) so this module does NOT import
    ``src.tools`` — avoiding a safety→tools import cycle. This helper only
    answers "is this tool destructive"; the execute node combines it with the
    ``DESTRUCTIVE_TOOL_HITL_ENABLED`` opt-in knob to decide whether to actually
    interrupt. Any error (unknown tool, missing method, ``registry is None``)
    returns False — the safe default is to RUN the tool (HITL is opt-in, and
    mis-annotation must never block a legitimate invocation indefinitely).
    """
    if registry is None:
        return False
    try:
        return bool(registry.is_destructive(tool_name))
    except Exception:
        return False


class SafetyPipeline:
    """7-layer sequential safety gate for evolution mutations.

    Layers:
    1. Syntax validation (AST parse)
    2. Static analysis (complexity, length checks)
    3. Security scan (forbidden patterns)
    4. Import validation (no dangerous imports)
    5. Behavioral constraints (resource limits)
    6. Sandbox execution (isolated test)
    7. Semantic check (behavioral invariants)

    All layers must pass for a mutation to be deployed.
    """

    async def validate(
        self,
        code: str,
        context: dict[str, Any] | None = None,
        sandbox_executor: Any | None = None,
        allowlisted_modules: set[str] | None = None,
    ) -> dict[str, Any]:
        """Run all 7 safety layers on the provided code.

        Args:
            code: The Python source code to validate.
            context: Optional context (target_path, mutation_type, etc.).
            sandbox_executor: Optional SandboxExecutor for Layer 6 sandbox testing.
            allowlisted_modules: Optional set of module names to allow even if
                normally considered dangerous (for generated tool validation).

        Returns:
            Dict with 'passed' bool, 'layers' results, and 'issues' list.
        """
        _ctx = context or {}
        results: dict[str, dict[str, Any]] = {}
        all_issues: list[str] = []

        # AST-dependent layers (syntax/imports/semantic) only apply to
        # Python-source mutations. Non-code mutations (prompt/config/memory
        # text) skip them — ast.parse on natural language hard-fails, so
        # otherwise non-code mutations could never deploy. Absent mutation_type
        # (dynamic-tool validation) runs every layer.
        mutation_type = _ctx.get("mutation_type")
        run_code_layers = mutation_type is None or mutation_type in _CODE_MUTATIONS

        # Layer 1: Syntax validation
        if run_code_layers:
            results["syntax"] = self._check_syntax(code)
        else:
            results["syntax"] = {"passed": True, "issues": [], "note": "skipped: non-code mutation"}
        if not results["syntax"]["passed"]:
            all_issues.extend(results["syntax"]["issues"])

        # Layer 2: Static analysis
        results["static"] = self._check_static(code)
        if not results["static"]["passed"]:
            all_issues.extend(results["static"]["issues"])

        # Layer 3: Security scan
        results["security"] = self._check_security(code)
        if not results["security"]["passed"]:
            all_issues.extend(results["security"]["issues"])

        # Layer 4: Import validation — context-requested modules extend the
        # allowlist (e.g. an evolution CODE mutation that legitimately needs a
        # normally-blocked module), layered on the caller's explicit allowlist.
        effective_allowlist = set(allowlisted_modules or ())
        required = _ctx.get("required_modules")
        if isinstance(required, (set, list, tuple)):
            effective_allowlist |= set(required)
        if run_code_layers:
            results["imports"] = self._check_imports(code, effective_allowlist)
        else:
            results["imports"] = {"passed": True, "issues": [], "note": "skipped: non-code mutation"}
        if not results["imports"]["passed"]:
            all_issues.extend(results["imports"]["issues"])

        # Layer 5: Behavioral constraints (AST file-write scoping)
        results["behavioral"] = self._check_behavioral(code, _ctx)
        if not results["behavioral"]["passed"]:
            all_issues.extend(results["behavioral"]["issues"])

        # Layer 6: Sandbox execution
        if sandbox_executor is not None:
            results["sandbox"] = await self._check_sandbox(code, sandbox_executor)
            if not results["sandbox"]["passed"]:
                all_issues.extend(results["sandbox"]["issues"])
        else:
            results["sandbox"] = {"passed": True, "issues": [], "note": "sandbox skipped (no executor)"}

        # Layer 7: Semantic check
        if run_code_layers:
            results["semantic"] = self._check_semantic(code)
        else:
            results["semantic"] = {"passed": True, "issues": [], "note": "skipped: non-code mutation"}
        if not results["semantic"]["passed"]:
            all_issues.extend(results["semantic"]["issues"])

        passed = all(layer["passed"] for layer in results.values())

        return {
            "passed": passed,
            "layers": results,
            "issues": all_issues,
        }

    def _check_syntax(self, code: str) -> dict[str, Any]:
        """Layer 1: Validate Python syntax via AST parsing."""
        try:
            ast.parse(code)
            return {"passed": True, "issues": []}
        except SyntaxError as exc:
            msg = f"Syntax error at line {exc.lineno}: {exc.msg}"
            return {"passed": False, "issues": [msg]}

    def _check_static(self, code: str) -> dict[str, Any]:
        """Layer 2: Static analysis — size and complexity checks."""
        issues: list[str] = []
        lines = code.splitlines()

        if len(lines) > 500:
            issues.append(f"Code too long: {len(lines)} lines (max 500)")

        if len(code) > 50_000:
            issues.append(f"Code too large: {len(code)} chars (max 50K)")

        # Check function complexity
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    complexity = sum(1 for _ in ast.walk(node) if isinstance(_, (ast.If, ast.For, ast.While, ast.ExceptHandler)))
                    if complexity > 20:
                        issues.append(f"Function '{node.name}' too complex: {complexity} branches (max 20)")
        except SyntaxError:
            pass

        if issues:
            return {"passed": False, "issues": issues}
        return {"passed": True, "issues": []}

    def _check_security(self, code: str) -> dict[str, Any]:
        """Layer 3: Security scan for forbidden patterns."""
        issues: list[str] = []

        for pattern in _FORBIDDEN_PATTERNS:
            matches = re.findall(pattern, code, re.IGNORECASE)
            if matches:
                issues.append(f"Forbidden pattern detected: {pattern}")

        if issues:
            return {"passed": False, "issues": issues}
        return {"passed": True, "issues": []}

    def _check_imports(
        self,
        code: str,
        allowlisted: set[str] | None = None,
    ) -> dict[str, Any]:
        """Layer 4: Validate imports — block dangerous modules.

        Args:
            code: Python source code to check.
            allowlisted: Optional set of module names to exempt from blocking.
        """
        dangerous_modules = {
            "os", "sys", "subprocess", "shutil", "ctypes",
            "multiprocessing", "threading", "socket",
            "http.server", "pickle", "marshal",
            "importlib", "pkgutil", "code", "codeop",
        }

        # Remove allowlisted modules from the dangerous set
        if allowlisted:
            dangerous_modules = dangerous_modules - allowlisted

        issues: list[str] = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root_module = alias.name.split(".")[0]
                        if root_module in dangerous_modules:
                            issues.append(f"Dangerous import: {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        root_module = node.module.split(".")[0]
                        if root_module in dangerous_modules:
                            issues.append(f"Dangerous import: from {node.module}")
        except SyntaxError:
            issues.append("Cannot parse imports due to syntax error")

        if issues:
            return {"passed": False, "issues": issues}
        return {"passed": True, "issues": []}

    def _check_behavioral(
        self, code: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Layer 5: Behavioral constraints.

        Flags (a) unconditional ``while True`` loops with no break/return
        (heuristic) and (b) ``open(...)`` calls in a write mode whose literal
        path resolves outside the sandbox root. This replaces the former brittle
        ``"open(" in code and "write" in code`` substring test, which false-
        positives on any legitimate relative-path write and false-negatives on
        writes that never mention the word "write". Writes with a dynamic
        (non-literal) path can't be statically resolved and are left to Layer 6.
        """
        issues: list[str] = []

        if "while True:" in code and "break" not in code and "return" not in code:
            issues.append("Potential infinite loop: while True without break/return")

        issues.extend(self._detect_unsandboxed_writes(code, context))

        if issues:
            return {"passed": False, "issues": issues}
        return {"passed": True, "issues": []}

    def _detect_unsandboxed_writes(
        self, code: str, context: dict[str, Any] | None
    ) -> list[str]:
        """AST-walk ``open()`` calls; flag write-mode paths outside the sandbox.

        Non-Python or syntactically-invalid input yields no issues here (Layer 1
        reports syntax errors). The sandbox root is ``context["sandbox_root"]``,
        defaulting to ``AgentSettings.workspace_root`` when absent.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        sandbox_root = (context or {}).get("sandbox_root")
        if sandbox_root is None:
            try:
                from src.config import get_settings

                sandbox_root = get_settings().agent.workspace_root
            except Exception:
                sandbox_root = None

        issues: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not (isinstance(node.func, ast.Name) and node.func.id == "open"):
                continue
            path = _open_path_literal(node)
            if path is None or not _is_write_open(node):
                continue
            if _path_outside_sandbox(path, sandbox_root):
                issues.append(f"File write outside sandbox directory: {path}")
        return issues

    async def _check_sandbox(self, code: str, sandbox_executor: Any) -> dict[str, Any]:
        """Layer 6: Execute code in sandbox and check for runtime errors."""
        try:
            result = await sandbox_executor.execute_code(code)
            if result.timed_out:
                return {"passed": False, "issues": ["Code timed out in sandbox execution"]}
            if not result.success:
                stderr_preview = result.stderr[:200] if result.stderr else "unknown error"
                return {"passed": False, "issues": [f"Sandbox execution failed: {stderr_preview}"]}
            return {"passed": True, "issues": []}
        except Exception as e:
            return {"passed": False, "issues": [f"Sandbox execution error: {e}"]}

    def _check_semantic(self, code: str) -> dict[str, Any]:
        """Layer 7: Semantic check — verify behavioral invariants via AST analysis.

        For non-code content (JSON, config), only checks for forbidden patterns
        like sys.exit(). The function/class requirement only applies to Python
        code mutations.
        """
        issues: list[str] = []

        # Detect non-code content (JSON/config) — skip strict Python checks
        stripped = code.strip()
        is_structured_content = stripped.startswith("{") or stripped.startswith("[")
        if is_structured_content:
            try:
                import json as _json  # noqa: DOC001
                _json.loads(stripped)
                return {"passed": True, "issues": []}
            except (ValueError, TypeError):
                pass  # Not valid JSON, proceed with Python checks

        try:
            tree = ast.parse(code)

            # Check for sys.exit() calls
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    if isinstance(func, ast.Attribute) and func.attr == "exit":
                        if isinstance(func.value, ast.Name) and func.value.id == "sys":
                            issues.append("Code contains sys.exit() — not allowed in mutations")

            # Check for empty function bodies (pass-only)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    body = node.body
                    if len(body) == 1 and isinstance(body[0], ast.Pass):
                        issues.append(f"Function '{node.name}' has empty body (pass only)")
                    elif len(body) == 1:
                        # Check for docstring-only bodies
                        if (isinstance(body[0], ast.Expr)
                                and isinstance(body[0].value, ast.Constant)
                                and isinstance(body[0].value.value, str)):
                            issues.append(f"Function '{node.name}' has docstring-only body")

            # Check that code has at least one function or class definition
            has_definition = any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                for node in ast.walk(tree)
            )
            if not has_definition and len(code.splitlines()) > 5:
                issues.append("Mutation code has no function or class definitions")

        except SyntaxError:
            issues.append("Cannot perform semantic check due to syntax error")

        if issues:
            return {"passed": False, "issues": issues}
        return {"passed": True, "issues": []}


# ── Layer 5 helpers ────────────────────────────────────────────────────


def _open_path_literal(call: ast.Call) -> str | None:
    """Return the literal path arg of an ``open()`` call, or None if dynamic."""
    if not call.args:
        return None
    arg = call.args[0]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    return None


def _is_write_open(call: ast.Call) -> bool:
    """True if the ``open()`` mode implies writing (``w``/``a``/``x``/``+``)."""
    mode = "r"
    if len(call.args) >= 2:
        val = call.args[1]
        if isinstance(val, ast.Constant) and isinstance(val.value, str):
            mode = val.value
    else:
        for kw in call.keywords:
            if (
                kw.arg == "mode"
                and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, str)
            ):
                mode = kw.value.value
                break
    return any(ch in mode for ch in ("w", "a", "x", "+"))


def _path_outside_sandbox(path: str, sandbox_root: str | None) -> bool:
    """True if ``path`` is an absolute write target outside ``sandbox_root``.

    Relative paths resolve under the process working directory (the sandbox) and
    are treated as inside — a legitimate ``open("out.json", "w")`` is allowed.
    An absolute path with no determinable sandbox root is flagged, since we
    cannot prove it is safe.
    """
    from pathlib import Path

    target = Path(path)
    if not target.is_absolute():
        return False
    if sandbox_root is None:
        return True
    try:
        root = Path(sandbox_root).resolve()
        target.resolve().relative_to(root)
        return False
    except (ValueError, OSError):
        return True
