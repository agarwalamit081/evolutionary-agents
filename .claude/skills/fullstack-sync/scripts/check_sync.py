#!/usr/bin/env python3
"""Check frontend-backend type and API endpoint synchronization.

Scans Python backend files for Pydantic models and FastAPI routes, and
TypeScript frontend files for interfaces/types and API calls. Reports
mismatches, missing fields, type conflicts, and orphaned endpoints.
"""

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Set


# ---------------------------------------------------------------------------
# Pydantic model parsing
# ---------------------------------------------------------------------------

def parse_pydantic_models(backend_dir: Path) -> Dict[str, Dict[str, str]]:
    """Return {class_name: {field_name: field_type}} for Pydantic BaseModel subclasses."""
    models: Dict[str, Dict[str, str]] = {}
    for py_file in backend_dir.rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            base_names = {
                b.id if isinstance(b, ast.Name) else ""
                for b in node.bases
            }
            if "BaseModel" not in base_names:
                continue
            fields: Dict[str, str] = {}
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    field_name = stmt.target.id
                    field_type = ast.unparse(stmt.annotation) if stmt.annotation else "Any"
                    fields[field_name] = field_type
            if fields:
                models[node.name] = fields
    return models


# ---------------------------------------------------------------------------
# TypeScript interface / type parsing
# ---------------------------------------------------------------------------

_TS_FIELD_RE = re.compile(
    r"^\s+(?P<name>\w+)(\?)?:\s*(?P<type>.+?)(?:;|,)?$", re.MULTILINE
)
_TS_INTERFACE_RE = re.compile(
    r"(?:export\s+)?(?:interface|type)\s+(?P<name>\w+)\s*[={]\s*",
    re.MULTILINE,
)


def parse_ts_interfaces(frontend_dir: Path) -> Dict[str, Dict[str, str]]:
    """Return {interface_name: {field_name: field_type}} for TS interfaces/types."""
    interfaces: Dict[str, Dict[str, str]] = {}
    for ts_file in frontend_dir.rglob("*.ts"):
        if ts_file.name.endswith(".d.ts"):
            continue
        try:
            content = ts_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for iface_match in _TS_INTERFACE_RE.finditer(content):
            iface_name = iface_match.group("name")
            start = iface_match.end()
            # Extract block between { and matching }
            depth = 0
            end = start
            for i, ch in enumerate(content[start:], start=start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            block = content[start:end]
            fields: Dict[str, str] = {}
            for field_match in _TS_FIELD_RE.finditer(block):
                fields[field_match.group("name")] = field_match.group("type").strip()
            if fields:
                interfaces[iface_name] = fields
    return interfaces


# ---------------------------------------------------------------------------
# FastAPI route parsing
# ---------------------------------------------------------------------------

_ROUTE_DECORATOR_RE = re.compile(
    r"@(?:app|router)\.(get|post|put|patch|delete)\(\s*['\"](?P<path>[^'\"]+)['\"]",
)
_FRONTEND_CALL_RE = re.compile(
    r"""(?:fetch\(\s*['"]|axios\.\w+\(\s*['"]|(?:api|apiClient)\.\w+\(\s*['"])(?P<path>/[^'"]+)""",
)


def parse_backend_routes(backend_dir: Path) -> Set[str]:
    """Return set of API route paths defined with FastAPI decorators."""
    routes: Set[str] = set()
    for py_file in backend_dir.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for m in _ROUTE_DECORATOR_RE.finditer(content):
            routes.add(m.group("path"))
    return routes


def parse_frontend_api_calls(frontend_dir: Path) -> Set[str]:
    """Return set of API paths referenced in frontend fetch/axios/api calls."""
    paths: Set[str] = set()
    for ts_file in frontend_dir.rglob("*.ts*"):
        try:
            content = ts_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for m in _FRONTEND_CALL_RE.finditer(content):
            paths.add(m.group("path"))
    return paths


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

_PY_TS_TYPE_MAP = {
    "str": "string",
    "int": "number",
    "float": "number",
    "bool": "boolean",
}


def normalize_py_type(t: str) -> str:
    """Best-effort normalization of a Python type annotation for comparison."""
    t = t.strip()
    # Unwrap Optional[X] -> X | null
    opt_match = re.match(r"Optional\[(.+)]", t)
    if opt_match:
        return normalize_py_type(opt_match.group(1))
    # Union[X, None] -> X
    if "None" in t and "|" in t:
        parts = [p.strip() for p in t.split("|") if p.strip() != "None"]
        if len(parts) == 1:
            return normalize_py_type(parts[0])
    # Strip NoneType, etc.
    for py, ts in _PY_TS_TYPE_MAP.items():
        if t == py:
            return ts
    return t


def normalize_ts_type(t: str) -> str:
    """Normalize a TS type for comparison (strip null/undefined unions)."""
    t = t.strip().rstrip(";").rstrip(",")
    parts = [p.strip() for p in t.split("|")]
    parts = [p for p in parts if p not in ("null", "undefined")]
    if len(parts) == 1:
        return parts[0]
    return t


# ---------------------------------------------------------------------------
# Check routines
# ---------------------------------------------------------------------------

def check_types(backend_dir: Path, frontend_dir: Path) -> List[str]:
    """Compare Pydantic models against TypeScript interfaces."""
    findings: List[str] = []
    py_models = parse_pydantic_models(backend_dir)
    ts_interfaces = parse_ts_interfaces(frontend_dir)

    all_names = set(py_models.keys()) | set(ts_interfaces.keys())

    for name in sorted(all_names):
        py_fields = py_models.get(name)
        ts_fields = ts_interfaces.get(name)

        if py_fields and not ts_fields:
            findings.append(f"[MISSING-TS] '{name}' exists in Python but has no matching TypeScript interface")
            continue
        if ts_fields and not py_fields:
            findings.append(f"[MISSING-PY] '{name}' exists in TypeScript but has no matching Pydantic model")
            continue

        py_set = set(py_fields.keys()) if py_fields else set()
        ts_set = set(ts_fields.keys()) if ts_fields else set()

        missing_in_ts = py_set - ts_set
        missing_in_py = ts_set - py_set

        for field in sorted(missing_in_ts):
            findings.append(f"[FIELD-MISSING-TS] '{name}.{field}' exists in Python but not in TypeScript")
        for field in sorted(missing_in_py):
            findings.append(f"[FIELD-MISSING-PY] '{name}.{field}' exists in TypeScript but not in Python")

        # Type conflict check on common fields
        common = py_set & ts_set
        for field in sorted(common):
            py_t = normalize_py_type(py_fields[field]) if py_fields else ""
            ts_t = normalize_ts_type(ts_fields[field]) if ts_fields else ""
            if py_t != ts_t:
                findings.append(
                    f"[TYPE-CONFLICT] '{name}.{field}': Python='{py_fields[field]}' vs TypeScript='{ts_fields[field]}'"
                )

    return findings


def check_endpoints(backend_dir: Path, frontend_dir: Path) -> List[str]:
    """Cross-reference backend routes with frontend API calls."""
    findings: List[str] = []
    routes = parse_backend_routes(backend_dir)
    calls = parse_frontend_api_calls(frontend_dir)

    # Normalize paths: strip trailing slashes for comparison
    routes_normalized = {r.rstrip("/") for r in routes}
    calls_normalized = {c.rstrip("/") for c in calls}

    orphaned_routes = routes_normalized - calls_normalized
    missing_clients = calls_normalized - routes_normalized

    for route in sorted(orphaned_routes):
        findings.append(f"[ORPHANED-ROUTE] Backend route '{route}' has no corresponding frontend API call")
    for call in sorted(missing_clients):
        findings.append(f"[MISSING-ROUTE] Frontend calls '{call}' but no matching backend route found")

    return findings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check frontend-backend type and API endpoint synchronization."
    )
    parser.add_argument(
        "--backend-dir",
        type=Path,
        required=True,
        help="Path to the backend source directory (Python/FastAPI).",
    )
    parser.add_argument(
        "--frontend-dir",
        type=Path,
        required=True,
        help="Path to the frontend source directory (TypeScript/React).",
    )
    parser.add_argument(
        "--check",
        choices=["types", "endpoints", "all"],
        default="all",
        help="Scope of the check (default: all).",
    )
    args = parser.parse_args()

    backend_dir = args.backend_dir.resolve()
    frontend_dir = args.frontend_dir.resolve()

    if not backend_dir.is_dir():
        print(f"Error: backend directory not found: {backend_dir}", file=sys.stderr)
        sys.exit(1)
    if not frontend_dir.is_dir():
        print(f"Error: frontend directory not found: {frontend_dir}", file=sys.stderr)
        sys.exit(1)

    all_findings: List[str] = []

    if args.check in ("types", "all"):
        all_findings.extend(check_types(backend_dir, frontend_dir))
    if args.check in ("endpoints", "all"):
        all_findings.extend(check_endpoints(backend_dir, frontend_dir))

    if all_findings:
        print(f"Found {len(all_findings)} synchronization issue(s):\n")
        for finding in all_findings:
            print(f"  - {finding}")
        sys.exit(1)
    else:
        print("All checks passed. Frontend and backend are in sync.")


if __name__ == "__main__":
    main()
