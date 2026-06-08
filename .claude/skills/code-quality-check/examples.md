# Code Quality Check — Examples

Runnable snippets demonstrating quality verification patterns.

## Example 1: Using ruff to validate Python file quality

```bash
# Lint and format-check in one pass
ruff check --select ALL src/api/handlers.py
ruff format --check src/api/handlers.py

# Auto-fix safe violations
ruff check --fix src/api/handlers.py
```

## Example 2: Checking for undefined functions with AST analysis

```python
"""Detect functions called but never defined in a module."""
import ast, sys

def find_undefined_calls(filepath: str) -> list[str]:
    with open(filepath) as f:
        tree = ast.parse(f.read())

    defined = {
        node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    called = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    return sorted(called - defined)

if __name__ == "__main__":
    undefs = find_undefined_calls(sys.argv[1])
    if undefs:
        print(f"Undefined functions: {', '.join(undefs)}")
        sys.exit(1)
    print("All called functions are defined.")
```

## Example 3: Validating React useEffect dependency arrays

```bash
# The eslint-plugin-react-hooks exhaustive-deps rule catches missing deps
npx eslint --rule 'react-hooks/exhaustive-deps: error' src/**/*.tsx

# Example fix: missing dependency
# BAD
#   useEffect(() => { fetchData(userId); }, []);
# GOOD
#   useEffect(() => { fetchData(userId); }, [userId]);
```

## Example 4: Running a pre-commit quality gate with ruff + mypy

```bash
#!/bin/bash
set -euo pipefail

echo "=== Ruff Lint ==="
ruff check src/

echo "=== Ruff Format ==="
ruff format --check src/

echo "=== Mypy ==="
mypy --strict src/

echo "All quality checks passed."
```

## Example 5: Checking for missing error handling in async functions

```python
"""Find async functions with I/O calls but no try/except."""
import ast, sys

def async_without_error_handling(filepath: str) -> list[str]:
    with open(filepath) as f:
        tree = ast.parse(f.read())

    issues = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        has_io = any(
            isinstance(child, ast.Await)
            for child in ast.walk(node)
        )
        has_try = any(
            isinstance(child, ast.Try)
            for child in ast.walk(node)
        )
        if has_io and not has_try:
            issues.append(f"Line {node.lineno}: async def '{node.name}' awaits without try/except")
    return issues

if __name__ == "__main__":
    findings = async_without_error_handling(sys.argv[1])
    for f in findings:
        print(f)
    sys.exit(1 if findings else 0)
```

## Example 6: Validating import hygiene across a Python package

```bash
# Check for unused imports across the package
ruff check --select F401 src/my_package/

# Check for missing imports (undefined names)
ruff check --select F821 src/my_package/

# Verify import order: stdlib / third-party / local
ruff check --select I001 src/my_package/
ruff check --fix --select I001 src/my_package/  # auto-fix order
```

## Example 7: TypeScript strict mode verification with tsc --noEmit

```bash
# Run strict type-checking without emitting files
npx tsc --noEmit --strict

# Common issues caught:
#   TS2322: Type 'string | undefined' is not assignable to type 'string'
#   TS7006: Parameter 'x' implicitly has an 'any' type
#   TS2532: Object is possibly 'undefined'

# Combine with eslint for runtime pattern checks
npx eslint . --ext .ts,.tsx --max-warnings 0
```
