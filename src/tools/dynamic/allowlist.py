"""Safe module allowlist for dynamically generated tools.

Defines which modules generated tool code may import and provides a
pre-imported namespace for handler materialization (double-barrier security).
"""

from __future__ import annotations

from typing import Any


# Modules that generated tool handlers are allowed to import.
# The safety pipeline will allowlist these when validating generated tool code.
ALLOWED_MODULES: frozenset[str] = frozenset({
    # ── Core stdlib ────────────────────────────────────────────────
    "httpx",
    "json",
    "re",
    "math",
    "datetime",
    "pathlib",
    "collections",
    "itertools",
    "textwrap",
    "typing",
    "dataclasses",
    "copy",
    "decimal",
    "statistics",
    "hashlib",
    "base64",
    "urllib.parse",
    "html.parser",
    "csv",
    "io",
    "xml.etree.ElementTree",
    "loguru",
    # ── Data / AI packages ────────────────────────────────────────
    "aiohttp",
    "bs4",
    "pandas",
    "numpy",
    "yaml",
    "lxml",
    "feedparser",
    "xmltodict",
    "jinja2",
    "PIL",
})

# Maximum number of tools that can be created per agent run.
MAX_TOOLS_PER_RUN: int = 3

# Packages that can be pip-installed in the sandbox.
# Only packages in this set may be requested for installation.
SAFE_PIP_PACKAGES: frozenset[str] = frozenset({
    "aiohttp",
    "beautifulsoup4",
    "pandas",
    "numpy",
    "pyyaml",
    "lxml",
    "feedparser",
    "xmltodict",
    "jinja2",
    "Pillow",
    "httpx",
})


def get_materializer_namespace() -> dict[str, Any]:
    """Build a pre-imported namespace for handler materialization.

    Returns a dict mapping module names to already-imported module objects,
    so generated code can use them without executing import statements.
    This is a security hardening measure: the namespace physically lacks
    dangerous modules like ``os`` and ``subprocess``.

    Returns:
        Dict mapping safe module names to their imported module objects.
    """
    import base64
    import collections
    import copy
    import csv
    import dataclasses
    import datetime
    import decimal
    import hashlib
    import html.parser
    import io
    import itertools
    import json
    import math
    import re
    import statistics
    import textwrap
    import typing
    import urllib.parse
    import xml.etree.ElementTree  # noqa: S405 — safe subset, no entity expansion

    namespace: dict[str, Any] = {
        "json": json,
        "re": re,
        "math": math,
        "datetime": datetime,
        "collections": collections,
        "itertools": itertools,
        "textwrap": textwrap,
        "typing": typing,
        "dataclasses": dataclasses,
        "copy": copy,
        "decimal": decimal,
        "statistics": statistics,
        "hashlib": hashlib,
        "base64": base64,
        "urllib": urllib,
        "urllib.parse": urllib.parse,
        "html": html,
        "html.parser": html.parser,
        "csv": csv,
        "io": io,
        "xml.etree.ElementTree": xml.etree.ElementTree,
    }

    # pathlib.Path as a convenience
    import pathlib

    namespace["pathlib"] = pathlib

    # Optional packages — only include if installed
    for mod_name in ("httpx", "loguru", "aiohttp", "bs4", "pandas", "numpy",
                     "yaml", "lxml", "feedparser", "xmltodict", "jinja2", "PIL"):
        try:
            mod = __import__(mod_name)
            namespace[mod_name] = mod
        except ImportError:
            pass

    return namespace
