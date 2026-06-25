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

Phase 2d (findings-04): installed-but-not-allowed libs admitted — scipy/sklearn
(scientific/numeric+ML; sklearn ← scikit-learn dist), openpyxl (Excel),
tabulate (table formatting), aiofiles (async file I/O), trafilatura (web
extraction, same egress contract as httpx/bs4), and libcst (Python AST
transform — the CODE self-evolution primitive). All pure-compute or
network-egress-under-contract; none introduce a managed binary or
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
    # ── Phase 2d — installed-but-not-allowed libs (findings-04) ─────
    # ``scipy``/``sklearn``: scientific/numeric + ML (pure compute, no egress);
    # ``sklearn`` is the import name for the scikit-learn dist. ``openpyxl``:
    # Excel read/write (file I/O under results/, same contract as ``io``/``csv``).
    # ``tabulate``: table pretty-printing (pure). ``aiofiles``: async file I/O
    # (same FS contract as the stdlib ``io`` already allowed). ``trafilatura``:
    # web-content extraction (network-egress — same contract/responsibility as
    # httpx/aiohttp/requests/bs4/unstructured above; powers web_scraper).
    # ``libcst``: Python syntax-tree parse/transform — the CODE self-evolution
    # primitive (pure, no egress). None introduce a managed binary or
    # detection-evasion surface (browser-automation stays deferred).
    "scipy",
    "sklearn",
    "openpyxl",
    "tabulate",
    "aiofiles",
    "trafilatura",
    "libcst",
    # ── Phase S7 — data/code libs for generated tools (pure-python or bundled solver) ─
    # ``fitz``/``pymupdf`` ← PyMuPDF dist (PDF render/extract). ``pulp``: LP, bundles
    # CBC. ``cvxpy``: convex optimization, bundles OSQP/ECOS/SCS. ``tree_sitter`` ←
    # tree-sitter dist; ``tree_sitter_python`` ← tree-sitter-python dist (incremental
    # parser + Python grammar). ``arxiv``: arXiv API client. Pure-compute or
    # network-egress-under-contract (arxiv); no managed binary beyond the bundled
    # solvers, no detection-evasion surface.
    "fitz",
    "pymupdf",
    "pulp",
    "cvxpy",
    "tree_sitter",
    "tree_sitter_python",
    "arxiv",
    # ── Phase S8 — algebraic modeling + offline solver ──
    # ``pyomo``: LP/MILP/MINLP modeling. Offline solving via ``highspy`` (the HiGHS
    # solver, bundled in its pip wheel — appsi_highs), verified end-to-end: solves a
    # trivial LP to the known optimum with NO glpk/cbc system binary and NO apt dep.
    # ``highspy`` is also allowlisted for direct HiGHS use. Both respect the no-
    # managed-binary invariant (the compiled HiGHS ships inside the highspy wheel).
    "pyomo",
    "highspy",
    # ── Phase 3 D5 — fast HTML/markdown libs (lightweight, no ML runtime) ──
    # ``selectolax``: fast HTML parser (lexbor C, ships wheels, zero deps).
    # ``mdformat``: markdown formatter (pure-py; builds on markdown-it-py).
    # ``mistune``: markdown renderer (pure-py, zero deps). Chosen over markitdown,
    # whose core dep ``magika`` pulls ``onnxruntime`` into every image — at odds
    # with the slim-image principle. HTML main-content extraction stays
    # trafilatura (web_scraper); these give generated tools fast parse/render.
    "selectolax",
    "mdformat",
    "mistune",
    # ── Phase 3 D6 — plotting (matplotlib) ──
    # ``matplotlib`` + its ``matplotlib.pyplot`` submodule so a generated tool can
    # render a chart and ``savefig`` it under results/. Pure compute (no egress);
    # the code_executor bootstrap defaults ``MPLBACKEND=Agg`` so headless savefig
    # works (no display server in host subprocess / runner mode). The dist shares
    # its import name.
    "matplotlib",
    "matplotlib.pyplot",
})

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
    # ── Phase 2d — installed-but-not-allowed libs (findings-04) ─────
    # sklearn ← scikit-learn; the other six share their import name. See the
    # ALLOWED_MODULES Phase-2d note for the egress/policy rationale.
    "scipy",
    "scikit-learn",
    "openpyxl",
    "tabulate",
    "aiofiles",
    "trafilatura",
    "libcst",
    # ── Phase S7 — dist names for the data/code libs above ──
    # pymupdf (import fitz/pymupdf); tree-sitter (import tree_sitter);
    # tree-sitter-python (import tree_sitter_python). pulp/cvxpy/arxiv share their
    # import name. See the ALLOWED_MODULES Phase-S7 note for the rationale.
    "pymupdf",
    "pulp",
    "cvxpy",
    "tree-sitter",
    "tree-sitter-python",
    "arxiv",
    # ── Phase S8 — algebraic modeling + offline HiGHS solver ──
    # pyomo drives highspy (appsi_highs) for offline LP/MILP; highspy bundles the
    # compiled HiGHS solver in its wheel — no glpk/cbc/apt dep. See the ALLOWED_MODULES
    # Phase-S8 note; both share their import name.
    "pyomo",
    "highspy",
    # ── Phase 3 D5 — dist names for the fast HTML/markdown libs above ──
    # selectolax/mdformat/mistune share their import name. See the
    # ALLOWED_MODULES Phase-3-D5 note for the markitdown/onnxruntime rationale.
    "selectolax",
    "mdformat",
    "mistune",
    # ── Phase 3 D6 — dist name for matplotlib (shares its import name) ──
    # See the ALLOWED_MODULES Phase-3-D6 note for the Agg-backend rationale.
    "matplotlib",
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

    # O3 path-anchoring: a Path pre-bound to results_root() so a generated tool
    # can write its deliverable deterministically under results/ (e.g.
    # ``pathlib.Path(results_dir, "out.csv")``) instead of landing in the
    # process CWD where verify cannot find it. Lazy import keeps this a leaf
    # module (no cycle into graph/). May be absent in stripped test namespaces.
    try:
        from src.tools._paths import results_root

        namespace["results_dir"] = results_root()
    except Exception:  # noqa: BLE001 — never let path wiring break materialization
        pass

    # Optional packages — only include if installed
    for mod_name in (
        "httpx", "loguru", "aiohttp", "bs4", "pandas", "numpy",
        "yaml", "lxml", "feedparser", "xmltodict", "jinja2", "PIL",
        "sympy", "pydantic", "pydantic_settings", "orjson", "json_repair",
        "unstructured", "markdown_it", "redis",
        # Phase 2d additions (findings-04): sklearn ← scikit-learn dist.
        "scipy", "sklearn", "openpyxl", "tabulate",
        "aiofiles", "trafilatura", "libcst",
        # Phase S7 additions: fitz/pymupdf (PyMuPDF), pulp, cvxpy,
        # tree_sitter/tree_sitter_python (tree-sitter + Python grammar), arxiv.
        "fitz", "pymupdf", "pulp", "cvxpy",
        "tree_sitter", "tree_sitter_python", "arxiv",
        # Phase S8: pyomo (modeling) + highspy (offline HiGHS solver backend).
        "pyomo", "highspy",
        # Phase 3 D5: fast HTML/markdown libs (selectolax/mdformat/mistune).
        "selectolax", "mdformat", "mistune",
        # Phase 3 D6: matplotlib (plotting); code_executor bootstrap sets Agg.
        "matplotlib",
    ):
        try:
            mod = __import__(mod_name)
            namespace[mod_name] = mod
        except ImportError:
            pass

    return namespace
