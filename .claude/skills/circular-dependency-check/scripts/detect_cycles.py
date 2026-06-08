#!/usr/bin/env python3
"""Detect circular import dependencies in Python and TypeScript projects."""

import argparse
import ast
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set


def find_files(root: str, extension: str) -> List[str]:
    """Walk directory tree and return all files matching extension."""
    files = []
    for dirpath, _, filenames in os.walk(root):
        # Skip hidden dirs, node_modules, __pycache__, .git
        dirs_to_skip = {
            "__pycache__",
            "node_modules",
            ".git",
            ".venv",
            "venv",
            ".tox",
            ".mypy_cache",
        }
        dirpath_obj = Path(dirpath)
        if any(part in dirs_to_skip for part in dirpath_obj.parts):
            continue
        for fname in filenames:
            if fname.endswith(extension):
                files.append(os.path.join(dirpath, fname))
    return files


def parse_python_imports(filepath: str, root: str) -> List[str]:
    """Parse a Python file with ast and return list of imported module paths."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=filepath)
    except (SyntaxError, UnicodeDecodeError):
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)

    # Convert module names to file paths relative to root
    resolved = set()
    for mod in imports:
        parts = mod.split(".")
        # Check if it maps to a local file
        candidate = os.path.join(root, *parts) + ".py"
        candidate_init = os.path.join(root, *parts, "__init__.py")
        if os.path.isfile(candidate):
            resolved.add(os.path.normpath(candidate))
        elif os.path.isfile(candidate_init):
            resolved.add(os.path.normpath(candidate_init))

    return list(resolved)


def parse_ts_imports(filepath: str, _root: str) -> List[str]:
    """Parse TypeScript file with regex and return list of resolved import paths."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
    except UnicodeDecodeError:
        return []

    # Match: import ... from './...' or import ... from "..." (relative only)
    pattern = re.compile(r'''import\s+.*?\s+from\s+['"](\.\.?/[^'"]+)['"]''')
    matches = pattern.findall(source)

    resolved = set()
    file_dir = os.path.dirname(filepath)
    for rel_path in matches:
        full = os.path.normpath(os.path.join(file_dir, rel_path))
        # Try with extensions
        for ext in ("", ".ts", ".tsx", ".js", ".jsx", "/index.ts", "/index.tsx"):
            candidate = full + ext
            if os.path.isfile(candidate):
                resolved.add(candidate)
                break

    return list(resolved)


def build_graph(root: str, language: str) -> Dict[str, List[str]]:
    """Build directed adjacency list of file dependencies."""
    if language == "python":
        ext = ".py"
        parse_fn = parse_python_imports
    else:
        ext = ".ts"
        parse_fn = parse_ts_imports

    files = find_files(root, ext)
    graph = defaultdict(list)

    for filepath in files:
        norm = os.path.normpath(filepath)
        deps = parse_fn(filepath, root)
        for dep in deps:
            if dep != norm:  # skip self-imports
                graph[norm].append(dep)

    return graph


def find_cycles(graph: Dict[str, List[str]]) -> List[List[str]]:
    """Run DFS to find all cycles in the directed graph."""
    visited: Set[str] = set()
    rec_stack: Set[str] = set()
    cycles: List[List[str]] = []

    def dfs(node: str, path: List[str]):
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor, path)
            elif neighbor in rec_stack:
                # Found a cycle — extract the cycle portion
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                cycles.append(cycle)

        path.pop()
        rec_stack.remove(node)

    for node in list(graph.keys()):
        if node not in visited:
            dfs(node, [])

    # Deduplicate cycles (normalize by rotating to smallest element)
    unique = []
    seen: Set[frozenset] = set()
    for cycle in cycles:
        core = cycle[:-1]  # remove the repeated last element
        key = frozenset(core)
        if key not in seen:
            seen.add(key)
            unique.append(cycle)

    return unique


def format_path(filepath: str, root: str) -> str:
    """Make path relative to root for display."""
    try:
        return os.path.relpath(filepath, root)
    except ValueError:
        return filepath


def detect_ts_via_madge(root: str) -> bool:
    """Attempt detection using npx madge. Returns True if madge ran successfully."""
    try:
        result = subprocess.run(
            ["npx", "madge", "--circular", root],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0 and "✘" not in result.stdout:
            print("madge: No circular dependencies found.")
            return True
        elif result.stdout.strip():
            print("madge detected cycles:")
            print(result.stdout)
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Detect circular import dependencies in Python and TypeScript projects."
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Root directory to scan for source files.",
    )
    parser.add_argument(
        "--language",
        choices=["python", "ts"],
        default="python",
        help="Language to analyze (default: python).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show full cycle paths instead of summary.",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.path)
    if not os.path.isdir(root):
        print(f"Error: {root} is not a directory.", file=sys.stderr)
        sys.exit(1)

    # For TypeScript, try madge first
    if args.language == "ts":
        if detect_ts_via_madge(root):
            if not args.verbose:
                return
            print("\n--- Falling back to built-in parser for detailed paths ---")

    graph = build_graph(root, args.language)
    cycles = find_cycles(graph)

    if not cycles:
        print("No circular dependencies found.")
        sys.exit(0)

    print(f"Found {len(cycles)} circular dependency cycle(s):\n")

    _ext = ".py" if args.language == "python" else ".ts"
    for i, cycle in enumerate(cycles, 1):
        if args.verbose:
            chain = " -> ".join(format_path(n, root) for n in cycle)
            print(f"  {i}) {chain}")
        else:
            # Summary: unique files involved
            files_in_cycle = sorted(
                set(format_path(n, root) for n in cycle[:-1])
            )
            print(f"  {i}) {' , '.join(files_in_cycle)}")

    sys.exit(1)


if __name__ == "__main__":
    main()
