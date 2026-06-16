"""Safe module allowlist for dynamically generated tools.

Defines which modules generated tool code may import and provides a
pre-imported namespace for handler materialization (double-barrier security).

Scope note: the dynamic-tool executor is Python-only, so this allowlist covers
Python import names only. Node/npm packages are N/A — there is no JS runtime to
install them into. Browser-automation packages (playwright/selenium/puppeteer/
playwright_stealth) are a deliberately DEFERRED opt-in: they require a managed
browser binary and network-egress/detection-evasion policy that the sandbox
does not yet provide, so they stay off the allowlist until that work lands.
Safe read-only packages added in this pass: requests (HTTP client),
dateutil (python-dateutil, date parsing), jsonschema (JSON schema validation),
and tenacity (retry primitives) — none introduce a managed binary or
detection-evasion surface.
"""

from __future__ import annotations

from typing import Any


# Modules that generated tool handlers are allowed to import.
# The safety pipeline will allowlist these when validating generated tool code.
# Import names — the dist (pip) name may differ (e.g. ``bs4`` ← beautifulsoup4,
# ``pydantic_settings`` ← pydantic-settings, ``json_repair`` ← json-repair,
# ``markdown_it`` ← markdown-it-py); see SAFE_PIP_PACKAGES for the dist names.
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
    # ── Scientific / symbolic math ─────────────────────────────────
    "sympy",
    # ── Validation & serialization ────────────────────────────────
    "pydantic",
    "pydantic_settings",
    "orjson",
    "json_repair",
    # ── Document parsing ───────────────────────────────────────────
    "unstructured",
    "markdown_it",
    # ── Infrastructure (network egress) ────────────────────────────
    # ``redis`` lets a generated tool reach a cache/queue server. It is
    # network-egress-capable, so it is allowed deliberately — only when a tool
    # genuinely needs it — and the sandbox must confine the target host to the
    # allowlisted infra endpoints. ``httpx``/``aiohttp`` above carry the same
    # egress capability and the same responsibility.
    "redis",
    # ── Safe read-only utilities ───────────────────────────────────
    # ``requests``: synchronous HTTP client (network-egress, same contract as
    #   httpx/aiohttp above). ``dateutil``: robust date parsing (pip:
    #   python-dateutil). ``jsonschema``: validate generated/captured JSON.
    #   ``tenacity``: retry-with-backoff primitives for transient failures.
    "requests",
    "dateutil",
    "jsonschema",
    "tenacity",
})

# Maximum number of tools that can be created per agent run.
MAX_TOOLS_PER_RUN: int = 3

# Packages that can be pip-installed in the sandbox.
# Only packages in this set may be requested for installation.
# Dist (pip) names — the matching import name may differ (see ALLOWED_MODULES).
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
    # ── Scientific / symbolic math ─────────────────────────────────
    "sympy",
    # ── Validation & serialization ─────────────────────────────────
    "pydantic",
    "pydantic-settings",
    "orjson",
    "json-repair",
    # ── Document parsing ───────────────────────────────────────────
    "unstructured",
    "markdown-it-py",
    # ── Infrastructure (network egress — see ALLOWED_MODULES note) ─
    "redis",
    # ── Safe read-only utilities (see ALLOWED_MODULES note) ────────
    # requests ← requests; dateutil ← python-dateutil; jsonschema ← jsonschema;
    # tenacity ← tenacity.
    "requests",
    "python-dateutil",
    "jsonschema",
    "tenacity",
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
    for mod_name in (
        "httpx", "loguru", "aiohttp", "bs4", "pandas", "numpy",
        "yaml", "lxml", "feedparser", "xmltodict", "jinja2", "PIL",
        "sympy", "pydantic", "pydantic_settings", "orjson", "json_repair",
        "unstructured", "markdown_it", "redis",
    ):
        try:
            mod = __import__(mod_name)
            namespace[mod_name] = mod
        except ImportError:
            pass

    return namespace
