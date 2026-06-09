"""Safe module allowlist for dynamically generated tools.

Defines which modules generated tool code may import and provides a
pre-imported namespace for handler materialization (double-barrier security).
"""

from __future__ import annotations

from typing import Any


# Modules that generated tool handlers are allowed to import.
# The safety pipeline will allowlist these when validating generated tool code.
ALLOWED_MODULES: frozenset[str] = frozenset({
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
    "loguru",
})

# Maximum number of tools that can be created per agent run.
MAX_TOOLS_PER_RUN: int = 3


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
    import dataclasses
    import datetime
    import decimal
    import hashlib
    import html.parser
    import itertools
    import json
    import math
    import re
    import statistics
    import textwrap
    import typing
    import urllib.parse

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
    }

    # httpx is an optional dependency — only include if installed
    try:
        import httpx

        namespace["httpx"] = httpx
    except ImportError:
        pass

    # loguru is optional — only include if installed
    try:
        import loguru

        namespace["loguru"] = loguru
    except ImportError:
        pass

    # pathlib.Path as a convenience
    import pathlib

    namespace["pathlib"] = pathlib

    return namespace
