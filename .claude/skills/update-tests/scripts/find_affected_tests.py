#!/usr/bin/env python3
"""Find test files affected by a function or model change.

Searches common test directories for references to a given function or class
name and categorizes them by relevance: direct test references, fixture
references, and import references.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Common test directory names to search
TEST_DIRS = [
    "tests",
    "test",
    "src/tests",
    "src/test",
    "__tests__",
]

# Filenames that typically contain shared fixtures
FIXTURE_FILES = [
    "conftest.py",
    "fixtures.py",
    "fixture.py",
    "factories.py",
    "factory.py",
]


def _run_ripgrep(pattern: str, search_path: Path, extra_globs: list[str] | None = None) -> str:
    """Run ripgrep and return stdout, or empty string on failure."""
    cmd = ["rg", "--no-heading", "--line-number", "--color=never", "-t", "py"]
    for glob in extra_globs or []:
        cmd.extend(["--glob", glob])
    cmd.extend([pattern, str(search_path)])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return result.stdout
    except FileNotFoundError:
        print("ERROR: ripgrep (rg) is not installed or not on PATH.", file=sys.stderr)
        sys.exit(1)
    return ""


def _resolve_test_dirs(root: Path) -> list[Path]:
    """Return test directories that actually exist under root."""
    dirs: list[Path] = []
    for name in TEST_DIRS:
        candidate = root / name
        if candidate.is_dir():
            dirs.append(candidate)
    return dirs


def _parse_matches(output: str) -> list[dict]:
    """Parse ripgrep output into structured match records.

    Each line is: <filepath>:<line_number>:<matched_line_content>
    """
    matches: list[dict] = []
    for line in output.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        matches.append({
            "file": parts[0],
            "line": parts[1],
            "content": parts[2].strip(),
        })
    return matches


def _classify_match(match: dict, function_name: str) -> str:
    """Classify a match as 'direct', 'fixture', or 'import'."""
    filepath = Path(match["file"])
    filename = filepath.name
    content = match["content"]

    # Fixture files
    if filename in FIXTURE_FILES:
        return "fixture"

    # conftest in any directory
    if filename == "conftest.py":
        return "fixture"

    # Import lines
    if "import" in content and function_name in content:
        return "import"

    return "direct"


def _print_group(label: str, matches: list[dict]) -> None:
    """Print a labelled group of matches."""
    if not matches:
        return
    print(f"\n  [{label}]")
    seen_files: set[str] = set()
    for m in matches:
        prefix = "    " if m["file"] not in seen_files else "    "
        seen_files.add(m["file"])
        print(f"  {prefix}{m['file']}:{m['line']}  {m['content']}")


def find_affected_tests(function_name: str, project_root: str) -> None:
    """Main entry point: search for and report affected test files."""
    root = Path(project_root).resolve()

    if not root.is_dir():
        print(f"ERROR: {root} is not a directory.", file=sys.stderr)
        sys.exit(1)

    test_dirs = _resolve_test_dirs(root)

    if not test_dirs:
        print(f"No test directories found under {root}", file=sys.stderr)
        print(f"Searched for: {', '.join(TEST_DIRS)}", file=sys.stderr)
        sys.exit(1)

    print(f"Searching for '{function_name}' in test directories under {root}")
    print(f"Directories: {', '.join(str(d.relative_to(root)) for d in test_dirs)}")

    all_matches: list[dict] = []

    # Search each test directory
    for test_dir in test_dirs:
        output = _run_ripgrep(function_name, test_dir)
        if output:
            all_matches.extend(_parse_matches(output))

    # Also search fixture files at the project root level
    for fixture_name in FIXTURE_FILES:
        fixture_path = root / fixture_name
        if fixture_path.is_file():
            output = _run_ripgrep(function_name, fixture_path)
            if output:
                all_matches.extend(_parse_matches(output))

    if not all_matches:
        print(f"\nNo references to '{function_name}' found in test files.")
        return

    # Deduplicate by file:line
    seen: set[str] = set()
    unique_matches: list[dict] = []
    for m in all_matches:
        key = f"{m['file']}:{m['line']}"
        if key not in seen:
            seen.add(key)
            unique_matches.append(m)

    # Classify matches
    direct: list[dict] = []
    fixture: list[dict] = []
    import_refs: list[dict] = []

    for m in unique_matches:
        category = _classify_match(m, function_name)
        if category == "direct":
            direct.append(m)
        elif category == "fixture":
            fixture.append(m)
        else:
            import_refs.append(m)

    # Summary
    total = len(unique_matches)
    files_affected = len({m["file"] for m in unique_matches})
    print(f"\nFound {total} reference(s) across {files_affected} file(s):")

    # Print grouped results
    _print_group("DIRECT", direct)
    _print_group("FIXTURE", fixture)
    _print_group("IMPORT", import_refs)

    # File summary
    print(f"\nAffected files:")
    for filepath in sorted({m["file"] for m in unique_matches}):
        count = sum(1 for m in unique_matches if m["file"] == filepath)
        print(f"  {filepath} ({count} reference(s))")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find test files affected by a function or model change.",
    )
    parser.add_argument(
        "--function",
        required=True,
        help="Name of the function or class to search for.",
    )
    parser.add_argument(
        "--path",
        default=".",
        help="Path to the project root directory (default: current directory).",
    )
    args = parser.parse_args()
    find_affected_tests(args.function, args.path)


if __name__ == "__main__":
    main()
