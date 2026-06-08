#!/usr/bin/env python3
"""Find all callers of a function across the codebase.

Usage:
    python find_callers.py --function get_user_by_id --path /project/root --type python
    python find_callers.py --function fetchData --path . --type ts
    python find_callers.py --function process_order --type all
"""

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def run_ripgrep(pattern: str, path: str, file_type: str | None = None) -> list[str]:
    """Run ripgrep with the given pattern and return raw output lines."""
    cmd = ["rg", "--no-heading", "--line-number", "--color=never", pattern, path]
    if file_type == "python":
        cmd.extend(["--type", "py"])
    elif file_type == "ts":
        cmd.extend(["--type", "ts"])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 2:
            print(f"ripgrep error: {result.stderr}", file=sys.stderr)
            return []
        return result.stdout.splitlines()
    except FileNotFoundError:
        print("Error: ripgrep (rg) is not installed. Install it from https://github.com/BurntSushi/ripgrep", file=sys.stderr)
        sys.exit(1)


def parse_line(line: str) -> tuple[str, int, str]:
    """Parse a ripgrep output line into (file, line_number, content)."""
    parts = line.split(":", 2)
    if len(parts) < 3:
        return parts[0], 0, ""
    return parts[0], int(parts[1]), parts[2]


def classify_python_reference(func_name: str, content: str) -> str:
    """Classify a Python reference as definition, call, or import."""
    stripped = content.strip()
    if stripped.startswith("#"):
        return "comment"
    if f"def {func_name}" in content:
        return "definition"
    if f"from " in content and f"import {func_name}" in content:
        return "import"
    if f"import {func_name}" == stripped or f"import {func_name}," in stripped:
        return "import"
    if f"{func_name}(" in content:
        return "call"
    if func_name in content:
        return "reference"
    return "unknown"


def classify_ts_reference(func_name: str, content: str) -> str:
    """Classify a TypeScript reference as definition, call, or import."""
    stripped = content.strip()
    if stripped.startswith("//") or stripped.startswith("*") or stripped.startswith("/*"):
        return "comment"
    if f"function {func_name}" in content or f"function {func_name}(" in content:
        return "definition"
    if f"const {func_name}" in content and "=>" in content:
        return "definition"
    if "import" in content and func_name in content:
        return "import"
    if f"{func_name}(" in content:
        return "call"
    if func_name in content:
        return "reference"
    return "unknown"


def find_callers(func_name: str, path: str, lang: str) -> dict[str, list[dict]]:
    """Find all references to a function and group results by file."""
    file_type = None if lang == "all" else lang
    lines = run_ripgrep(func_name, path, file_type)

    results: dict[str, list[dict]] = defaultdict(list)

    for line in lines:
        filepath, lineno, content = parse_line(line)
        if not content:
            continue

        ref_type = "other"
        if lang in ("python", "all"):
            ref_type = classify_python_reference(func_name, content)
        if lang == "ts":
            ref_type = classify_ts_reference(func_name, content)

        # Re-classify for mixed mode
        if lang == "all":
            if filepath.endswith(".ts") or filepath.endswith(".tsx"):
                ref_type = classify_ts_reference(func_name, content)
            elif filepath.endswith(".py"):
                ref_type = classify_python_reference(func_name, content)
            else:
                ref_type = "reference"

        results[filepath].append({
            "line": lineno,
            "type": ref_type,
            "content": content.strip(),
        })

    return dict(results)


def print_results(func_name: str, results: dict[str, list[dict]]) -> None:
    """Print results grouped by file, with definitions first."""
    definitions = []
    calls = []
    imports = []
    references = []

    for filepath, refs in sorted(results.items()):
        for ref in refs:
            entry = (filepath, ref["line"], ref["content"])
            if ref["type"] == "definition":
                definitions.append(entry)
            elif ref["type"] == "call":
                calls.append(entry)
            elif ref["type"] == "import":
                imports.append(entry)
            else:
                references.append(entry)

    if definitions:
        print(f"\n{'=' * 60}")
        print(f"DEFINITIONS of '{func_name}'")
        print(f"{'=' * 60}")
        for filepath, lineno, content in definitions:
            print(f"  {filepath}:{lineno}")
            print(f"    {content}")

    if imports:
        print(f"\n{'=' * 60}")
        print(f"IMPORTS of '{func_name}'")
        print(f"{'=' * 60}")
        for filepath, lineno, content in imports:
            print(f"  {filepath}:{lineno}")
            print(f"    {content}")

    if calls:
        print(f"\n{'=' * 60}")
        print(f"CALL SITES of '{func_name}'")
        print(f"{'=' * 60}")
        current_file = None
        for filepath, lineno, content in calls:
            if filepath != current_file:
                print(f"\n  {filepath}:")
                current_file = filepath
            print(f"    {lineno}: {content}")

    if references:
        print(f"\n{'=' * 60}")
        print(f"OTHER REFERENCES to '{func_name}'")
        print(f"{'=' * 60}")
        for filepath, lineno, content in references:
            print(f"  {filepath}:{lineno}")
            print(f"    {content}")

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {len(definitions)} definition(s), "
          f"{len(imports)} import(s), "
          f"{len(calls)} call(s), "
          f"{len(references)} other reference(s)")
    print(f"{'=' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find all callers of a function across the codebase."
    )
    parser.add_argument(
        "--function", required=True,
        help="Name of the function to search for."
    )
    parser.add_argument(
        "--path", default=".",
        help="Project root path to search in (default: current directory)."
    )
    parser.add_argument(
        "--type", dest="lang", choices=["python", "ts", "all"], default="all",
        help="Language to search: python, ts, or all (default: all)."
    )
    args = parser.parse_args()

    resolved_path = str(Path(args.path).resolve())

    print(f"Searching for '{args.function}' in {resolved_path} (type: {args.lang})")
    results = find_callers(args.function, resolved_path, args.lang)

    if not results:
        print(f"No references to '{args.function}' found.")
        sys.exit(0)

    print_results(args.function, results)


if __name__ == "__main__":
    main()
