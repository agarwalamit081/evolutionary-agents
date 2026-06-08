#!/usr/bin/env python3
"""Run code quality checks on Python and TypeScript files."""

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SEVERITY_CRITICAL = "Critical"
SEVERITY_WARNING = "Warning"
SEVERITY_INFO = "Info"

MAX_FILE_LINES = 500

STUB_PATTERNS = [
    re.compile(r"^\s*pass\s*$"),
    re.compile(r"raise\s+NotImplementedError"),
    re.compile(r"#\s*TODO|#\s*FIXME"),
    re.compile(r"^\s*\.\.\.\s*$"),
]


def classify_file(filepath: str) -> str:
    """Return 'python', 'typescript', or 'unknown'."""
    ext = Path(filepath).suffix.lower()
    if ext in (".py", ".pyw"):
        return "python"
    if ext in (".ts", ".tsx", ".js", ".jsx"):
        return "typescript"
    return "unknown"


def check_file_size(filepath: str) -> list[dict]:
    """Warn if file exceeds line limit."""
    findings = []
    with open(filepath, errors="replace") as f:
        line_count = sum(1 for _ in f)
    if line_count > MAX_FILE_LINES:
        findings.append({
            "severity": SEVERITY_WARNING,
            "message": f"{line_count} lines (limit {MAX_FILE_LINES}) — consider splitting",
        })
    return findings


def check_stubs(filepath: str) -> list[dict]:
    """Detect placeholder / stub code."""
    findings = []
    with open(filepath, errors="replace") as f:
        for lineno, line in enumerate(f, start=1):
            for pattern in STUB_PATTERNS:
                if pattern.search(line):
                    findings.append({
                        "severity": SEVERITY_WARNING,
                        "message": f"Line {lineno}: stub pattern → {line.strip()}",
                    })
    return findings


def check_async_error_handling(filepath: str) -> list[dict]:
    """Find async functions with await but no try/except."""
    findings = []
    try:
        with open(filepath) as f:
            source = f.read()
        tree = ast.parse(source, filename=filepath)
    except SyntaxError:
        return [{"severity": SEVERITY_CRITICAL, "message": "File has syntax errors, skipping AST checks."}]

    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        has_await = any(isinstance(child, ast.Await) for child in ast.walk(node))
        has_try = any(isinstance(child, ast.Try) for child in ast.walk(node))
        if has_await and not has_try:
            findings.append({
                "severity": SEVERITY_WARNING,
                "message": f"Line {node.lineno}: async def '{node.name}' uses await without try/except",
            })
    return findings


def run_ruff(filepath: str) -> list[dict]:
    """Run ruff check on a Python file."""
    findings = []
    try:
        result = subprocess.run(
            ["ruff", "check", "--output-format=json", filepath],
            capture_output=True, text=True, timeout=30,
        )
        if result.stdout.strip():
            for item in json.loads(result.stdout):
                findings.append({
                    "severity": SEVERITY_CRITICAL,
                    "message": f"ruff {item.get('code', '?')}: {item.get('message', '')}",
                })
    except FileNotFoundError:
        findings.append({"severity": SEVERITY_INFO, "message": "ruff not installed — skipping lint."})
    except subprocess.TimeoutExpired:
        findings.append({"severity": SEVERITY_WARNING, "message": "ruff check timed out."})
    return findings


def check_typescript(filepath: str) -> list[dict]:
    """Suggest eslint and tsc for TypeScript files."""
    findings = [
        {"severity": SEVERITY_INFO, "message": f"Run: npx eslint {filepath}"},
        {"severity": SEVERITY_INFO, "message": f"Run: npx tsc --noEmit {filepath}"},
    ]
    return findings


def analyze_file(filepath: str) -> list[dict]:
    """Run all checks on a single file."""
    findings = []
    ftype = classify_file(filepath)

    if ftype == "unknown":
        return [{"severity": SEVERITY_INFO, "message": f"Unsupported file type: {filepath}"}]

    findings.extend(check_file_size(filepath))

    if ftype == "python":
        findings.extend(check_stubs(filepath))
        findings.extend(check_async_error_handling(filepath))
        findings.extend(run_ruff(filepath))
    elif ftype == "typescript":
        findings.extend(check_stubs(filepath))
        findings.extend(check_typescript(filepath))

    return findings


def format_text(results: dict[str, list[dict]]) -> str:
    """Pretty-print results grouped by file and severity."""
    lines = []
    for filepath, findings in results.items():
        if not findings:
            lines.append(f"  {filepath}: clean")
            continue
        lines.append(f"  {filepath}:")
        for f in findings:
            lines.append(f"    [{f['severity']}] {f['message']}")
    return "\n".join(lines)


def format_json(results: dict[str, list[dict]]) -> str:
    """Return JSON string of results."""
    return json.dumps(results, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run code quality checks on Python and TypeScript files.")
    parser.add_argument("--path", required=True, help="File or directory to check.")
    parser.add_argument("--report", choices=["text", "json"], default="text", help="Output format (default: text).")
    args = parser.parse_args()

    target = Path(args.path)
    if not target.exists():
        print(f"Error: {args.path} does not exist.", file=sys.stderr)
        sys.exit(1)

    files: list[str] = []
    if target.is_file():
        files.append(str(target))
    else:
        for root, _, filenames in os.walk(target):
            for fn in filenames:
                files.append(os.path.join(root, fn))

    if not files:
        print("No files found to check.", file=sys.stderr)
        sys.exit(0)

    all_results: dict[str, list[dict]] = {}
    has_critical = False

    for fp in files:
        findings = analyze_file(fp)
        all_results[fp] = findings
        if any(f["severity"] == SEVERITY_CRITICAL for f in findings):
            has_critical = True

    if args.report == "json":
        print(format_json(all_results))
    else:
        print("=== Code Quality Report ===")
        print(format_text(all_results))

    sys.exit(1 if has_critical else 0)


if __name__ == "__main__":
    main()
