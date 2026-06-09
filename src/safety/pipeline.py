"""7-layer safety pipeline for evolution mutations."""

from __future__ import annotations

import ast
import re
from typing import Any



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
    ) -> dict[str, Any]:
        """Run all 7 safety layers on the provided code.

        Args:
            code: The Python source code to validate.
            context: Optional context (target_path, mutation_type, etc.).
            sandbox_executor: Optional SandboxExecutor for Layer 6 sandbox testing.

        Returns:
            Dict with 'passed' bool, 'layers' results, and 'issues' list.
        """
        ctx = context or {}
        results: dict[str, dict[str, Any]] = {}
        all_issues: list[str] = []

        # Layer 1: Syntax validation
        results["syntax"] = self._check_syntax(code)
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

        # Layer 4: Import validation
        results["imports"] = self._check_imports(code)
        if not results["imports"]["passed"]:
            all_issues.extend(results["imports"]["issues"])

        # Layer 5: Behavioral constraints
        results["behavioral"] = self._check_behavioral(code)
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
        results["semantic"] = self._check_semantic(code)
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

    def _check_imports(self, code: str) -> dict[str, Any]:
        """Layer 4: Validate imports — block dangerous modules."""
        dangerous_modules = {
            "os", "sys", "subprocess", "shutil", "ctypes",
            "multiprocessing", "threading", "socket",
            "http.server", "xml.etree", "pickle", "marshal",
            "importlib", "pkgutil", "code", "codeop",
        }

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

    def _check_behavioral(self, code: str) -> dict[str, Any]:
        """Layer 5: Check for behavioral constraints."""
        issues: list[str] = []

        # Check for infinite loop patterns
        if "while True:" in code and "break" not in code and "return" not in code:
            issues.append("Potential infinite loop: while True without break/return")

        # Check for file write without sandbox
        if "open(" in code and "write" in code:
            if "sandbox" not in code.lower() and "tmp" not in code.lower():
                issues.append("File write outside sandbox directory")

        if issues:
            return {"passed": False, "issues": issues}
        return {"passed": True, "issues": []}

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
        """Layer 7: Semantic check — verify behavioral invariants via AST analysis."""
        issues: list[str] = []

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
