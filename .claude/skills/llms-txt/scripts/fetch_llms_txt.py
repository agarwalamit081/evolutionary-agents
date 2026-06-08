#!/usr/bin/env python3
"""Fetch or generate llms.txt documentation for a Python package.

Attempts to fetch llms.txt from known documentation URLs first,
then falls back to extracting docstrings from the installed package.
Caches results in .claude/llms-cache/<package>.txt

Usage:
    uv run python fetch_llms_txt.py --package <name> [--force] [--output <path>]
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

KNOWN_LLMS_TXT_PROVIDERS: dict[str, list[str]] = {
    "fastapi": [
        "https://fastapi.tiangolo.com/llms-full.txt",
        "https://fastapi.tiangolo.com/llms.txt",
    ],
    "langchain": [
        "https://python.langchain.com/llms-full.txt",
        "https://python.langchain.com/llms.txt",
    ],
    "sqlalchemy": [
        "https://docs.sqlalchemy.org/llms.txt",
    ],
    "pydantic": [
        "https://docs.pydantic.dev/llms-full.txt",
        "https://docs.pydantic.dev/llms.txt",
    ],
    "httpx": [
        "https://www.python-httpx.org/llms.txt",
    ],
    "duckdb": [
        "https://duckdb.org/llms.txt",
    ],
    "litellm": [
        "https://docs.litellm.ai/llms.txt",
    ],
    "django": [
        "https://docs.djangoproject.com/llms.txt",
    ],
}

# Generic URL patterns to try for unknown packages
GENERIC_URL_PATTERNS: list[str] = [
    "https://{package}.readthedocs.io/llms.txt",
    "https://docs.{package}.io/llms.txt",
    "https://{package}.readthedocs.io/en/latest/llms.txt",
    "https://www.{package}.org/llms.txt",
]

DEFAULT_CACHE_DIR = Path(".claude/llms-cache")
TIMEOUT_SECONDS = 15
MAX_FETCH_SIZE = 2 * 1024 * 1024  # 2MB max for llms.txt


def get_cache_dir() -> Path:
    """Return the cache directory, creating it if needed."""
    cache_dir = DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_installed_version(package: str) -> Optional[str]:
    """Get installed version of a package using importlib.metadata."""
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:
        return None


def fetch_from_url(url: str) -> Optional[str]:
    """Fetch content from a URL. Returns None on any failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "llms-txt-fetcher/1.0"})
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            if resp.status != 200:
                return None
            content = resp.read(MAX_FETCH_SIZE)
            charset = resp.headers.get_content_charset() or "utf-8"
            return content.decode(charset, errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
        return None


def fetch_from_known_providers(package: str) -> Optional[str]:
    """Try fetching from known llms.txt providers."""
    urls = KNOWN_LLMS_TXT_PROVIDERS.get(package, [])
    for url in urls:
        content = fetch_from_url(url)
        if content and len(content.strip()) > 100:
            return content
    return None


def fetch_from_generic_patterns(package: str) -> Optional[str]:
    """Try generic URL patterns for unknown packages."""
    for pattern in GENERIC_URL_PATTERNS:
        url = pattern.format(package=package)
        content = fetch_from_url(url)
        if content and len(content.strip()) > 100:
            return content
    return None


def generate_from_docstrings(package: str) -> Optional[str]:
    """Generate llms.txt content from installed package docstrings."""
    try:
        mod = importlib.import_module(package)
    except ImportError:
        return None

    lines: list[str] = []
    lines.append(f"# {package}")
    version = get_installed_version(package)
    if version:
        lines.append(f"Version: {version}")
    lines.append("Source: generated from installed package docstrings")
    lines.append("")

    # Module docstring
    if mod.__doc__:
        lines.append("## Overview")
        lines.append("")
        lines.append(mod.__doc__.strip())
        lines.append("")

    # Public API surface
    lines.append("## Public API")
    lines.append("")

    public_names = getattr(mod, "__all__", None)
    if public_names is None:
        public_names = [
            name
            for name in sorted(dir(mod))
            if not name.startswith("_")
            and callable(getattr(mod, name, None))
        ]

    for name in public_names:
        obj = getattr(mod, name, None)
        if obj is None:
            continue

        if inspect.isclass(obj):
            lines.append(f"### class {name}")
            if obj.__doc__:
                lines.append("")
                lines.append(obj.__doc__.strip())
            # List public methods
            for method_name in sorted(dir(obj)):
                if method_name.startswith("_"):
                    continue
                method = getattr(obj, method_name, None)
                if callable(method):
                    try:
                        sig = inspect.signature(method)
                        lines.append(f"- `{method_name}{sig}`")
                    except (ValueError, TypeError):
                        lines.append(f"- `{method_name}()`")
            lines.append("")

        elif inspect.isfunction(obj) or callable(obj):
            lines.append(f"### {name}")
            try:
                sig = inspect.signature(obj)
                lines.append(f"`{name}{sig}`")
            except (ValueError, TypeError):
                lines.append(f"`{name}()`")
            if obj.__doc__:
                lines.append("")
                lines.append(obj.__doc__.strip())
            lines.append("")

    content = "\n".join(lines)
    return content if len(content.strip()) > 50 else None


def read_cached(package: str) -> Optional[str]:
    """Read cached llms.txt if it exists and version matches."""
    cache_path = get_cache_dir() / f"{package}.txt"
    if not cache_path.exists():
        return None

    content = cache_path.read_text(encoding="utf-8")

    # Check version match
    installed_version = get_installed_version(package)
    if installed_version:
        version_match = re.search(r"^Version:\s*(.+)$", content, re.MULTILINE)
        if version_match:
            cached_version = version_match.group(1).strip()
            if cached_version != installed_version:
                return None  # Version mismatch, need to regenerate

    return content


def write_cache(package: str, content: str) -> Path:
    """Write content to cache and return the path."""
    cache_dir = get_cache_dir()
    cache_path = cache_dir / f"{package}.txt"
    cache_path.write_text(content, encoding="utf-8")
    return cache_path


def fetch_or_generate(package: str, force: bool = False) -> str:
    """Main entry point: fetch or generate llms.txt for a package."""
    # Check cache first
    if not force:
        cached = read_cached(package)
        if cached:
            return cached

    # Try known providers
    content = fetch_from_known_providers(package)

    # Try generic patterns
    if not content:
        content = fetch_from_generic_patterns(package)

    # Fallback: generate from docstrings
    if not content:
        content = generate_from_docstrings(package)

    if not content:
        raise RuntimeError(
            f"Could not fetch or generate llms.txt for '{package}'. "
            f"Package may not be installed or may not have public documentation."
        )

    # Cache the result
    cache_path = write_cache(package, content)
    print(f"Cached: {cache_path}", file=sys.stderr)
    return content


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch or generate llms.txt for a Python package"
    )
    parser.add_argument("--package", required=True, help="Package name")
    parser.add_argument("--force", action="store_true", help="Force refresh cache")
    parser.add_argument("--output", help="Output file path (default: stdout)")
    args = parser.parse_args()

    content = fetch_or_generate(args.package, force=args.force)

    if args.output:
        Path(args.output).write_text(content, encoding="utf-8")
        print(f"Written to: {args.output}", file=sys.stderr)
    else:
        print(content)


if __name__ == "__main__":
    main()
