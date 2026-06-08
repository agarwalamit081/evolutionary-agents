#!/usr/bin/env python3
"""Verify that a library API exists and meets version requirements."""

import argparse
import importlib
import importlib.metadata
import sys


def get_installed_version(package: str) -> str | None:
    """Return the installed version of a package, or None if not found."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def parse_version(version_str: str) -> tuple:
    """Parse a version string into a comparable tuple of ints.

    Falls back to simple string comparison logic when the `packaging`
    library is not available.  Each segment is converted to int where
    possible so that (0, 10) > (0, 9) works correctly.
    """
    parts = []
    for segment in version_str.split("."):
        try:
            parts.append(int(segment))
        except ValueError:
            parts.append(segment)
    return tuple(parts)


def compare_versions(installed: str, minimum: str) -> bool:
    """Return True if installed >= minimum.

    Tries `packaging.version` first for robust comparison; falls back
    to tuple-based comparison.
    """
    try:
        from packaging.version import Version

        return Version(installed) >= Version(minimum)
    except ImportError:
        pass

    return parse_version(installed) >= parse_version(minimum)


def check_module_importable(package: str) -> tuple[bool, str | None]:
    """Try to import the package.  Return (success, error_message)."""
    try:
        importlib.import_module(package)
        return True, None
    except ImportError as exc:
        return False, str(exc)


def check_attribute(package: str, method: str) -> tuple[bool, str | None]:
    """Check if an attribute exists on the top-level module.

    Supports dotted paths like ``client.ChatCompletion.create`` by
    walking the attribute chain.

    Returns (exists, error_message).
    """
    try:
        obj = importlib.import_module(package)
    except ImportError as exc:
        return False, f"Cannot import {package}: {exc}"

    parts = method.split(".")
    current = obj
    path = [package]
    for part in parts:
        path.append(part)
        if not hasattr(current, part):
            return False, f"'{'.'.join(path)}' not found"
        current = getattr(current, part)

    return True, None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that a library API exists and meets version requirements."
    )
    parser.add_argument(
        "--package",
        required=True,
        help="Name of the package to check (pip install name).",
    )
    parser.add_argument(
        "--method",
        default=None,
        help="Method or attribute path to verify on the package, e.g. 'client.ChatCompletion.create'.",
    )
    parser.add_argument(
        "--min-version",
        default=None,
        help="Minimum version required (e.g. '0.2.5').",
    )

    args = parser.parse_args()

    exit_code = 0

    # --- 1. Can the module be imported? ---
    importable, import_err = check_module_importable(args.package)
    if not importable:
        print(f"FAIL  module import   {args.package} — {import_err}")
        return 1
    print(f"PASS  module import   {args.package}")

    # --- 2. Installed version ---
    installed_version = get_installed_version(args.package)
    if installed_version:
        print(f"INFO  installed version   {args.package}=={installed_version}")
    else:
        print(
            f"WARN  version unknown   {args.package} "
            "(package imported but metadata version not found)"
        )

    # --- 3. Method / attribute check ---
    if args.method:
        attr_exists, attr_err = check_attribute(args.package, args.method)
        if attr_exists:
            print(f"PASS  attribute exists   {args.package}.{args.method}")
        else:
            print(f"FAIL  attribute missing  {args.package}.{args.method} — {attr_err}")
            exit_code = 1

    # --- 4. Minimum version check ---
    if args.min_version:
        if installed_version is None:
            print(
                f"WARN  version check skipped   "
                f"(cannot determine installed version for {args.package})"
            )
            exit_code = 1
        elif compare_versions(installed_version, args.min_version):
            print(
                f"PASS  version meets minimum   "
                f"{args.package}=={installed_version} >= {args.min_version}"
            )
        else:
            print(
                f"FAIL  version too low   "
                f"{args.package}=={installed_version} < {args.min_version}"
            )
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
