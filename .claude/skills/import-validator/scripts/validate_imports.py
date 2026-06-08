#!/usr/bin/env python3
"""Validate import hygiene in Python and TypeScript files."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple


class Finding(NamedTuple):
    """A single import validation finding."""

    category: str  # unused, missing, grouping, init_stale
    file: str
    line: int
    message: str


def detect_language(path: Path) -> str:
    """Detect language from file extension or directory contents."""
    if path.is_file():
        ext = path.suffix.lower()
        if ext == ".py":
            return "python"
        if ext in (".ts", ".tsx", ".js", ".jsx"):
            return "ts"
    # Directory: check for Python or TypeScript indicators
    if path.is_dir():
        has_py = any(path.rglob("*.py"))
        has_ts = any(path.rglob("*.ts")) or any(path.rglob("*.tsx"))
        if has_py and not has_ts:
            return "python"
        if has_ts and not has_py:
            return "ts"
    return "python"


def run_ruff_check(paths: list[str], fix: bool = False) -> list[Finding]:
    """Run ruff check for unused and redefined imports."""
    cmd = ["ruff", "check", "--select", "F401,F811"]
    if fix:
        cmd.append("--fix")
    cmd.extend(paths)

    findings: list[Finding] = []
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        print("ERROR: ruff is not installed. Install with: pip install ruff", file=sys.stderr)
        sys.exit(1)

    # Parse ruff output: file:line:col: CODE message
    for line in result.stdout.strip().splitlines():
        match = re.match(r"^(.+?):(\d+):\d+: (F\d+)\s+(.*)", line)
        if match:
            filepath, lineno, code, message = match.groups()
            category = "unused" if code == "F401" else "missing"
            if code == "F811":
                category = "unused"
            findings.append(Finding(category, filepath, int(lineno), message))

    return findings


def validate_init_py(init_path: Path, package_dir: Path) -> list[Finding]:
    """Check that modules referenced in __init__.py actually exist."""
    findings: list[Finding] = []
    if not init_path.exists():
        return findings

    content = init_path.read_text(encoding="utf-8")
    # Match: from .module_name import ...
    imports = re.findall(r"from\s+\.(\w+)\s+import", content)

    for idx, module_name in enumerate(imports, start=1):
        module_file = package_dir / f"{module_name}.py"
        module_pkg = package_dir / module_name / "__init__.py"
        if not module_file.exists() and not module_pkg.exists():
            findings.append(
                Finding(
                    "init_stale",
                    str(init_path),
                    idx,
                    f"Module '{module_name}' referenced in __init__.py does not exist",
                )
            )

    return findings


def check_ts_unused_imports(path: Path) -> list[Finding]:
    """Basic check for unused TypeScript imports using regex heuristics.

    For full validation, use eslint with @typescript-eslint/no-unused-vars.
    """
    findings: list[Finding] = []
    ts_files: list[Path] = []
    if path.is_file():
        ts_files = [path]
    else:
        for ext in ("*.ts", "*.tsx", "*.js", "*.jsx"):
            ts_files.extend(path.rglob(ext))

    for ts_file in ts_files:
        content = ts_file.read_text(encoding="utf-8")
        lines = content.splitlines()
        for lineno, line in enumerate(lines, start=1):
            stripped = line.strip()
            # Match: import { A, B } from "..."
            match = re.match(r'import\s+\{([^}]+)\}\s+from\s+["\']', stripped)
            if not match:
                # Match: import Name from "..."
                match2 = re.match(r'import\s+(\w+)\s+from\s+["\']', stripped)
                if match2:
                    name = match2.group(1)
                    # Check if name appears elsewhere in the file
                    rest = content.replace(stripped, "", 1)
                    if not re.search(rf"\b{re.escape(name)}\b", rest):
                        findings.append(
                            Finding("unused", str(ts_file), lineno, f"Import '{name}' may be unused")
                        )
                continue
            names = [n.strip().split(" as ")[-1].strip() for n in match.group(1).split(",")]
            for name in names:
                name = name.strip()
                if not name:
                    continue
                rest = content.replace(stripped, "", 1)
                if not re.search(rf"\b{re.escape(name)}\b", rest):
                    findings.append(
                        Finding("unused", str(ts_file), lineno, f"Import '{name}' may be unused")
                    )

    return findings


def check_import_grouping(path: Path, language: str) -> list[Finding]:
    """Validate import grouping: stdlib, third-party, local separation."""
    findings: list[Finding] = []

    # Python stdlib modules (common subset for detection)
    STDLIB_MODULES = {
        "os", "sys", "json", "pathlib", "re", "collections", "typing",
        "argparse", "subprocess", "datetime", "logging", "functools",
        "itertools", "abc", "io", "hashlib", "copy", "dataclasses",
        "enum", "contextlib", "asyncio", "unittest", "math", "random",
    }

    files: list[Path] = []
    if path.is_file():
        files = [path]
    elif language == "python":
        files = list(path.rglob("*.py"))
    else:
        for ext in ("*.ts", "*.tsx"):
            files.extend(path.rglob(ext))

    for filepath in files:
        if filepath.name == "__init__.py":
            continue
        content = filepath.read_text(encoding="utf-8")
        lines = content.splitlines()

        prev_group: str | None = None
        for lineno, line in enumerate(lines, start=1):
            stripped = line.strip()
            if language == "python":
                match = re.match(r"^(?:from|import)\s+(\w+)", stripped)
                if not match:
                    continue
                root_module = match.group(1)
                if root_module in STDLIB_MODULES:
                    group = "stdlib"
                elif root_module.startswith(".") or root_module == filepath.stem.split("_")[0]:
                    group = "local"
                else:
                    group = "third-party"
            else:
                # TypeScript
                match = re.match(r'^import\s+.*from\s+["\']([^"\']+)["\']', stripped)
                if not match:
                    continue
                source = match.group(1)
                if source.startswith("."):
                    group = "local"
                else:
                    group = "third-party"

            if prev_group is not None and group != prev_group:
                # Check if there was a blank line between groups
                if lineno >= 2 and lines[lineno - 2].strip() != "":
                    findings.append(
                        Finding(
                            "grouping",
                            str(filepath),
                            lineno,
                            f"Import group change ({prev_group} -> {group}) without blank line separator",
                        )
                    )
            prev_group = group

    return findings


def format_findings(findings: list[Finding]) -> str:
    """Format findings into a readable report."""
    if not findings:
        return "No import issues found."

    categories: dict[str, list[Finding]] = {}
    for f in findings:
        categories.setdefault(f.category, []).append(f)

    lines: list[str] = []
    category_labels = {
        "unused": "Unused Imports",
        "missing": "Missing / Redefined Imports",
        "grouping": "Import Grouping",
        "init_stale": "Stale __init__.py References",
    }

    for cat, cat_findings in categories.items():
        label = category_labels.get(cat, cat)
        lines.append(f"\n== {label} ({len(cat_findings)}) ==")
        for f in sorted(cat_findings, key=lambda x: (x.file, x.line)):
            lines.append(f"  {f.file}:{f.line}: {f.message}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate import hygiene in Python and TypeScript files."
    )
    parser.add_argument("--path", required=True, help="File or directory to validate")
    parser.add_argument("--fix", action="store_true", default=False, help="Auto-fix issues where possible")
    parser.add_argument(
        "--language",
        choices=["python", "ts", "auto"],
        default="auto",
        help="Language to validate (default: auto-detect)",
    )
    args = parser.parse_args()

    target = Path(args.path).resolve()
    if not target.exists():
        print(f"ERROR: Path does not exist: {target}", file=sys.stderr)
        sys.exit(1)

    language = args.language
    if language == "auto":
        language = detect_language(target)

    all_findings: list[Finding] = []

    if language == "python":
        paths = [str(target)] if target.is_file() else [str(target)]
        all_findings.extend(run_ruff_check(paths, fix=args.fix))

        # Check __init__.py files for stale references
        if target.is_dir():
            for init_file in target.rglob("__init__.py"):
                all_findings.extend(validate_init_py(init_file, init_file.parent))
        elif target.is_file() and target.name == "__init__.py":
            all_findings.extend(validate_init_py(target, target.parent))

        all_findings.extend(check_import_grouping(target, "python"))

    elif language == "ts":
        all_findings.extend(check_ts_unused_imports(target))
        all_findings.extend(check_import_grouping(target, "ts"))

        if args.fix:
            print("NOTE: Auto-fix for TypeScript is not supported. Use eslint --fix instead.")

    report = format_findings(all_findings)
    print(report)

    if all_findings:
        print(f"\nTotal issues: {len(all_findings)}")
        if args.fix:
            print("Auto-fix applied where possible.")
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
