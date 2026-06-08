#!/usr/bin/env python3
"""Detect potential resource leaks in Python and TypeScript files."""

import argparse
import ast
import os
import re
import sys
from pathlib import Path
from typing import NamedTuple


class LeakFinding(NamedTuple):
    file: str
    line: int
    category: str
    message: str


def detect_python_leaks(source: str, filepath: str) -> list[LeakFinding]:
    """Scan Python source for common resource leak patterns."""
    findings: list[LeakFinding] = []
    lines = source.splitlines()

    # --- AST-based checks ---
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return findings

    for node in ast.walk(tree):
        # open() calls not inside a `with` statement
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "open":
                # Walk up to check if inside a With item's context_expr
                if not _is_inside_with(node, tree):
                    line = node.lineno
                    findings.append(
                        LeakFinding(
                            filepath,
                            line,
                            "file-io",
                            "open() called without a `with` statement; file handle may leak.",
                        )
                    )

    # --- Regex-based checks ---
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Subprocess without context manager
        if re.search(r"subprocess\.(run|Popen|call|check_output|check_call)\(", stripped):
            if "with " not in line and "with(" not in line:
                findings.append(
                    LeakFinding(
                        filepath,
                        i,
                        "subprocess",
                        "subprocess call without context manager; use `with subprocess.Popen(...)`.",
                    )
                )

        # create_connection / new Pool without close/shutdown
        if re.search(r"(create_connection|ConnectionPool|ThreadPoolExecutor|ProcessPoolExecutor)\(", stripped):
            findings.append(
                LeakFinding(
                    filepath,
                    i,
                    "connection",
                    "Resource created without visible close/shutdown; ensure cleanup in finally or context manager.",
                )
            )

        # Missing await on async context manager (heuristic)
        if re.search(r"(async with|async for)", stripped) is None:
            if re.search(r"await\s+(aiofiles|async_session|client\.\w+)\.", stripped):
                pass  # Legitimate await expression
            elif re.search(r"(client\.(get|post|put|delete|aclose))\(", stripped):
                if "await " not in line:
                    findings.append(
                        LeakFinding(
                            filepath,
                            i,
                            "async",
                            "Async client method called without `await`; result may not be awaited.",
                        )
                    )

    return findings


def _is_inside_with(target_node: ast.AST, tree: ast.Module) -> bool:
    """Check if a node is nested inside a `with` statement's context expression."""
    for node in ast.walk(tree):
        if isinstance(node, ast.With):
            for item in node.items:
                if _contains_node(item.context_expr, target_node):
                    return True
    return False


def _contains_node(parent: ast.AST, target: ast.AST) -> bool:
    """Check if parent AST tree contains the target node (by identity)."""
    for child in ast.walk(parent):
        if child is target:
            return True
    return False


def detect_ts_leaks(source: str, filepath: str) -> list[LeakFinding]:
    """Scan TypeScript/React source for common resource leak patterns."""
    findings: list[LeakFinding] = []
    lines = source.splitlines()

    # Track addEventListener and removeEventListener pairs
    added_events: dict[str, list[int]] = {}
    removed_events: set[str] = set()

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()

        # addEventListener tracking
        add_match = re.search(r"addEventListener\(\s*['\"](\w+)['\"]", stripped)
        if add_match:
            event_name = add_match.group(1)
            added_events.setdefault(event_name, []).append(i)

        # removeEventListener tracking
        remove_match = re.search(r"removeEventListener\(\s*['\"](\w+)['\"]", stripped)
        if remove_match:
            removed_events.add(remove_match.group(1))

    # Report unpaired listeners
    for event_name, line_nums in added_events.items():
        if event_name not in removed_events:
            for ln in line_nums:
                findings.append(
                    LeakFinding(
                        filepath,
                        ln,
                        "event-listener",
                        f"addEventListener('{event_name}') without matching removeEventListener.",
                    )
                )

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()

        # useEffect without cleanup return
        if re.search(r"useEffect\s*\(", stripped):
            # Look ahead for a return function in the same useEffect block
            block = _extract_block(lines, i - 1)
            if block and not re.search(r"return\s*\(\s*\(\)\s*=>|return\s+\(\)\s*=>|return\s+function", block):
                findings.append(
                    LeakFinding(
                        filepath,
                        i,
                        "react",
                        "useEffect without cleanup return; subscriptions or listeners may leak.",
                    )
                )

        # AbortController without abort
        if re.search(r"new\s+AbortController\s*\(", stripped):
            remaining = "\n".join(lines[i:])
            if ".abort()" not in remaining:
                findings.append(
                    LeakFinding(
                        filepath,
                        i,
                        "react",
                        "AbortController created without calling .abort(); request may hang on unmount.",
                    )
                )

        # setInterval without clearInterval
        if re.search(r"setInterval\s*\(", stripped):
            remaining = "\n".join(lines[i:])
            if "clearInterval" not in remaining:
                findings.append(
                    LeakFinding(
                        filepath,
                        i,
                        "timer",
                        "setInterval without corresponding clearInterval; timer will run indefinitely.",
                    )
                )

        # createConnection / new Pool without close/shutdown
        if re.search(r"(createConnection|new\s+Pool|\.connect\()\s*", stripped):
            remaining = "\n".join(lines[i:])
            if not re.search(r"\.(close|end|shutdown|destroy)\(", remaining):
                findings.append(
                    LeakFinding(
                        filepath,
                        i,
                        "connection",
                        "Connection or pool created without visible close/end/shutdown call.",
                    )
                )

    return findings


def _extract_block(lines: list[str], start_idx: int) -> str | None:
    """Extract a curly-brace-delimited block starting from the line at start_idx."""
    brace_count = 0
    block_lines: list[str] = []
    started = False
    for line in lines[start_idx:]:
        block_lines.append(line)
        brace_count += line.count("{") - line.count("}")
        if brace_count > 0:
            started = True
        if started and brace_count <= 0:
            return "\n".join(block_lines)
        if start_idx + len(block_lines) > start_idx + 60:
            break
    return None


def scan_file(filepath: str, language: str) -> list[LeakFinding]:
    """Scan a single file for resource leaks."""
    try:
        source = Path(filepath).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return []

    ext = Path(filepath).suffix.lower()
    if language == "auto":
        if ext in (".py",):
            language = "python"
        elif ext in (".ts", ".tsx", ".js", ".jsx"):
            language = "ts"
        else:
            return []

    if language == "python":
        return detect_python_leaks(source, filepath)
    elif language == "ts":
        return detect_ts_leaks(source, filepath)
    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect potential resource leaks in Python and TypeScript files."
    )
    parser.add_argument(
        "--path",
        required=True,
        help="File or directory to scan.",
    )
    parser.add_argument(
        "--language",
        choices=["python", "ts", "auto"],
        default="auto",
        help="Language to scan (default: auto-detect from file extension).",
    )
    args = parser.parse_args()

    target = Path(args.path)
    files: list[str] = []

    if target.is_file():
        files.append(str(target))
    elif target.is_dir():
        for root, _, filenames in os.walk(target):
            for fname in filenames:
                ext = Path(fname).suffix.lower()
                if args.language == "python" and ext == ".py":
                    files.append(os.path.join(root, fname))
                elif args.language == "ts" and ext in (".ts", ".tsx", ".js", ".jsx"):
                    files.append(os.path.join(root, fname))
                elif args.language == "auto" and ext in (".py", ".ts", ".tsx", ".js", ".jsx"):
                    files.append(os.path.join(root, fname))
    else:
        print(f"Path not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    all_findings: list[LeakFinding] = []
    for fp in files:
        all_findings.extend(scan_file(fp, args.language))

    if not all_findings:
        print("No potential resource leaks detected.")
        return

    print(f"Found {len(all_findings)} potential resource leak(s):\n")
    for f in sorted(all_findings, key=lambda x: (x.file, x.line)):
        print(f"  {f.file}:{f.line}  [{f.category}]  {f.message}")

    sys.exit(0)


if __name__ == "__main__":
    main()
