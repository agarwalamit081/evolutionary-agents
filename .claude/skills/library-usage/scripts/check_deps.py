#!/usr/bin/env python3
"""Check project dependencies for pinned versions and preferred library usage."""

import argparse
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Preferred library mappings (mirrors reference.md tables)
# ---------------------------------------------------------------------------

PYTHON_PREFERRED: dict[str, tuple[str, ...]] = {
    "logging": ("loguru",),
    "os.path": ("pathlib",),
    "pydantic": ("pydantic",),
    "pydantic-settings": ("pydantic-settings",),
    "httpx": ("httpx",),
    "sqlalchemy": ("sqlalchemy",),
    "alembic": ("alembic",),
    "asyncpg": ("asyncpg",),
    "msgspec": ("msgspec",),
    "json-repair": ("json-repair",),
    "ruff": ("ruff",),
    "mypy": ("mypy",),
    "pytest": ("pytest",),
    "tenacity": ("tenacity",),
    "circuitbreaker": ("circuitbreaker",),
    "aiofiles": ("aiofiles",),
    "python-dotenv": ("python-dotenv",),
    "tiktoken": ("tiktoken",),
    "jinja2": ("jinja2",),
    "fastapi": ("fastapi",),
    "uvicorn": ("uvicorn",),
    "playwright": ("playwright",),
}

PYTHON_DISCOURAGED: dict[str, str] = {
    "urllib3": "Use `httpx` for HTTP requests",
    "urllib": "Use `httpx` for HTTP requests",
    "requests": "Use `httpx` for async-capable HTTP requests",
    "flake8": "Use `ruff` for linting",
    "pylint": "Use `ruff` for linting",
    "pickle": "Use `msgspec` for serialization (security risk)",
    "unittest": "Use `pytest` for testing",
    "flask": "Use `fastapi` for async API frameworks",
}

JS_PREFERRED: dict[str, tuple[str, ...]] = {
    "axios": ("axios",),
    "zustand": ("zustand",),
    "jotai": ("jotai",),
    "zod": ("zod",),
    "vitest": ("vitest",),
    "jest": ("jest",),
    "eslint": ("eslint",),
    "prettier": ("prettier",),
    "typescript": ("typescript",),
}

JS_DISCOURAGED: dict[str, str] = {
    "xmlhttprequest": "Use `fetch` or `axios`",
}

# ---------------------------------------------------------------------------
# Dependency parsing
# ---------------------------------------------------------------------------


def parse_pyproject_toml(path: Path) -> dict[str, str]:
    """Parse [project.dependencies] from pyproject.toml using tomllib (stdlib)."""
    try:
        import tomllib
    except ImportError:
        print("ERROR: tomllib not available (requires Python 3.11+)")
        sys.exit(1)

    with open(path, "rb") as f:
        data = tomllib.load(f)

    deps: dict[str, str] = {}
    for dep_spec in data.get("project", {}).get("dependencies", []):
        dep_spec = dep_spec.strip()
        match = re.match(r"^([A-Za-z0-9_.-]+)\s*(.*)", dep_spec)
        if match:
            name = match.group(1).lower().replace("_", "-")
            version = match.group(2).strip()
            deps[name] = version
    return deps


def parse_requirements_txt(path: Path) -> dict[str, str]:
    """Parse requirements.txt into {package: version_spec}."""
    deps: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        match = re.match(r"^([A-Za-z0-9_.-]+)\s*(.*)", line)
        if match:
            name = match.group(1).lower().replace("_", "-")
            version = match.group(2).strip()
            deps[name] = version
    return deps


def parse_package_json(path: Path) -> dict[str, str]:
    """Parse dependencies + devDependencies from package.json."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    deps: dict[str, str] = {}
    for section in ("dependencies", "devDependencies"):
        for name, version in data.get(section, {}).items():
            deps[name.lower()] = version
    return deps


def detect_and_parse(project_root: Path) -> tuple[dict[str, str], str]:
    """Auto-detect dependency file and parse it. Returns (deps, file_type)."""
    candidates = [
        (project_root / "pyproject.toml", "python", parse_pyproject_toml),
        (project_root / "requirements.txt", "python", parse_requirements_txt),
        (project_root / "package.json", "js", parse_package_json),
    ]

    for file_path, file_type, parser in candidates:
        if file_path.exists():
            return parser(file_path), file_type

    print("ERROR: No dependency file found (pyproject.toml, requirements.txt, or package.json)")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_pinned(deps: dict[str, str]) -> list[str]:
    """Check that all deps have version pins. Returns list of warnings."""
    warnings: list[str] = []
    for name, version_spec in deps.items():
        if not version_spec:
            warnings.append(f"  UNPINNED: {name} (no version constraint)")
        elif not re.match(r"^(==|>=|<=|~=|>|<)", version_spec):
            # package.json often uses ^ or ~ which are acceptable
            if not re.match(r"^[\^~>=<]", version_spec):
                warnings.append(f"  UNPINNED: {name} ({version_spec})")
    return warnings


def check_preferred(deps: dict[str, str], file_type: str) -> list[str]:
    """Check for discouraged libraries. Returns list of warnings."""
    discouraged = PYTHON_DISCOURAGED if file_type == "python" else JS_DISCOURAGED
    warnings: list[str] = []
    for name in deps:
        normalized = name.lower().replace("_", "-")
        if normalized in discouraged:
            reason = discouraged[normalized]
            warnings.append(f"  DISCOURAGED: {name} -- {reason}")
    return warnings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check project dependencies for pinned versions and preferred library usage."
    )
    parser.add_argument(
        "--path",
        default=".",
        help="Path to project root (default: current directory)",
    )
    parser.add_argument(
        "--check",
        choices=["preferred", "pinned", "all"],
        default="all",
        help="Which check to run (default: all)",
    )
    args = parser.parse_args()

    project_root = Path(args.path).resolve()
    if not project_root.is_dir():
        print(f"ERROR: {project_root} is not a directory")
        sys.exit(1)

    deps, file_type = detect_and_parse(project_root)
    print(f"Detected dependency file type: {file_type}")
    print(f"Total dependencies: {len(deps)}")
    print()

    overall_pass = True

    if args.check in ("pinned", "all"):
        print("=" * 60)
        print("CHECK: Pinned Versions")
        print("=" * 60)
        warnings = check_pinned(deps)
        if warnings:
            overall_pass = False
            print(f"FAIL: {len(warnings)} unpinned dependency(ies) found")
            for w in warnings:
                print(w)
        else:
            print("PASS: All dependencies have version constraints")
        print()

    if args.check in ("preferred", "all"):
        print("=" * 60)
        print("CHECK: Preferred Libraries")
        print("=" * 60)
        warnings = check_preferred(deps, file_type)
        if warnings:
            overall_pass = False
            print(f"WARN: {len(warnings)} discouraged dependency(ies) found")
            for w in warnings:
                print(w)
        else:
            print("PASS: No discouraged libraries detected")
        print()

    print("=" * 60)
    if overall_pass:
        print("RESULT: ALL CHECKS PASSED")
    else:
        print("RESULT: SOME CHECKS FAILED -- review warnings above")
    print("=" * 60)

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
