#!/usr/bin/env python3
"""Estimate token usage for files and directories.

Uses a chars/4 heuristic to approximate token counts and reports
usage as a percentage of the model's context window.
"""

import argparse
import os
import sys
from pathlib import Path

# Model context windows (in tokens)
MODEL_WINDOWS = {
    "sonnet": 200_000,
    "opus": 200_000,
    "haiku": 200_000,
}

# Directories and patterns to exclude from estimation
EXCLUDE_DIRS = {
    "__pycache__",
    "node_modules",
    ".git",
    ".hg",
    ".svn",
    "dist",
    "build",
    ".next",
    ".cache",
    "coverage",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "vendor",
    ".venv",
    "venv",
    "env",
}

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".lock",
    ".min.js",
    ".min.css",
}

CHARS_PER_TOKEN = 4

WARNING_FILE_PERCENT = 10.0   # Warn if single file exceeds this % of context
WARNING_TOTAL_PERCENT = 50.0  # Warn if total exceeds this % of context


def estimate_tokens(filepath: Path) -> int:
    """Estimate token count for a single file using chars/4 heuristic."""
    try:
        size = filepath.stat().st_size
        return size // CHARS_PER_TOKEN
    except OSError:
        return 0


def should_exclude(path: Path) -> bool:
    """Check if a path should be excluded from estimation."""
    if path.is_dir() and path.name in EXCLUDE_DIRS:
        return True
    if path.is_file() and path.suffix in EXCLUDE_EXTENSIONS:
        return True
    return False


def estimate_directory(dirpath: Path, _verbose: bool = False) -> list[tuple[Path, int]]:
    """Walk directory tree and estimate tokens per file.

    Returns a list of (filepath, token_count) tuples sorted by token count descending.
    """
    results = []
    for root, dirs, files in os.walk(dirpath):
        root_path = Path(root)

        # Filter excluded directories in-place to prevent os.walk from descending
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        for filename in files:
            filepath = root_path / filename
            if filepath.suffix in EXCLUDE_EXTENSIONS:
                continue
            tokens = estimate_tokens(filepath)
            if tokens > 0:
                results.append((filepath, tokens))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def format_path(path: Path, base: Path) -> str:
    """Format path relative to base, falling back to absolute."""
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def print_results(
    results: list[tuple[Path, int]],
    context_window: int,
    model: str,
    base_path: Path,
    verbose: bool = False,
) -> None:
    """Print estimation results with warnings."""
    total_tokens = sum(t for _, t in results)
    total_percent = (total_tokens / context_window) * 100 if context_window else 0

    print(f"\nContext Window Estimate (model: {model}, window: {context_window:,} tokens)")
    print("=" * 72)

    if verbose and results:
        print(f"\n{'File':<50} {'Tokens':>8} {'% Window':>10}")
        print("-" * 72)
        for filepath, tokens in results:
            percent = (tokens / context_window) * 100
            rel_path = format_path(filepath, base_path)
            # Truncate long paths
            if len(rel_path) > 48:
                rel_path = "..." + rel_path[-45:]
            print(f"{rel_path:<50} {tokens:>8,} {percent:>9.1f}%")
            if percent >= WARNING_FILE_PERCENT:
                print(f"  WARNING: {rel_path} uses {percent:.1f}% of context window")
        print("-" * 72)

    print(f"\nTotal files scanned: {len(results)}")
    print(f"Total estimated tokens: {total_tokens:,}")
    print(f"Context window usage: {total_percent:.1f}%")

    if total_percent >= WARNING_TOTAL_PERCENT:
        print(
            f"\nWARNING: Total estimated tokens ({total_percent:.1f}%) "
            f"exceeds {WARNING_TOTAL_PERCENT:.0f}% of the {model} context window. "
            f"Consider using targeted reads and /compact."
        )

    # Warn about individual large files even in non-verbose mode
    large_files = [
        (fp, t) for fp, t in results
        if (t / context_window) * 100 >= WARNING_FILE_PERCENT
    ]
    if large_files and not verbose:
        print(f"\nLarge files (>={WARNING_FILE_PERCENT:.0f}% of context window):")
        for filepath, tokens in large_files:
            percent = (tokens / context_window) * 100
            print(f"  {format_path(filepath, base_path)}: {tokens:,} tokens ({percent:.1f}%)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Estimate token usage for files and directories against model context windows."
    )
    parser.add_argument(
        "--path",
        required=True,
        help="File or directory to estimate token usage for.",
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_WINDOWS.keys()),
        default="sonnet",
        help="Model to estimate against (default: sonnet).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-file token breakdown.",
    )
    args = parser.parse_args()

    target = Path(args.path)
    if not target.exists():
        print(f"Error: path '{args.path}' does not exist.", file=sys.stderr)
        return 1

    context_window = MODEL_WINDOWS[args.model]

    if target.is_file():
        tokens = estimate_tokens(target)
        percent = (tokens / context_window) * 100
        print(f"\nFile: {target}")
        print(f"Estimated tokens: {tokens:,}")
        print(f"Context window usage ({args.model}): {percent:.1f}%")
        if percent >= WARNING_FILE_PERCENT:
            print(
                f"\nWARNING: This file uses {percent:.1f}% of the {args.model} "
                f"context window. Use Read with offset/limit for targeted access."
            )
    elif target.is_dir():
        results = estimate_directory(target, _verbose=args.verbose)
        if not results:
            print(f"No scannable files found in '{args.path}'.")
            return 0
        print_results(results, context_window, args.model, target, verbose=args.verbose)
    else:
        print(f"Error: '{args.path}' is not a regular file or directory.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
