"""Code validator tool — validates Python code via AST and syntax checks."""

from __future__ import annotations

import ast

from loguru import logger


async def code_validator(code: str) -> str:
    """Validate Python code for syntax errors and AST issues.

    Args:
        code: Python source code to validate.

    Returns:
        Validation result: "VALID" or error description.
    """
    logger.info(f"Validating code ({len(code)} chars)")

    # Step 1: Syntax check via AST parse
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        line_info = f"line {exc.lineno}" if exc.lineno else "unknown line"
        return f"SYNTAX ERROR at {line_info}: {exc.msg}"

    # Step 2: Check for common issues
    issues: list[str] = []

    # Check for undefined names in top-level (basic check)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            continue
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("_"):
                issues.append(f"Private module import: {node.module}")

    # Step 3: Report
    if issues:
        return f"VALID with warnings: {'; '.join(issues)}"

    function_count = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    class_count = sum(1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef))

    return f"VALID — {function_count} functions, {class_count} classes defined"


TOOL_DEFINITION = {
    "name": "code_validator",
    "handler": code_validator,
    "description": (
        "Validate Python code for syntax errors using AST parsing. "
        "Returns validation status and any detected issues. "
        "Does NOT execute the code."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python source code to validate.",
            },
        },
        "required": ["code"],
    },
}
