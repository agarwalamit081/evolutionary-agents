"""Invariant-guard: the allowlist and the Dockerfile pip sets must agree.

The 1:1 invariant (documented in ``src/tools/dynamic/allowlist.py`` and both
Dockerfiles): a module allowlisted for generated tools must be pip-installable
in BOTH sandbox images (``Dockerfile.runner`` + ``Dockerfile.toolbox``), so an
executed script importing an allowlisted dep resolves in deployed mode too.
``unstructured`` violated this — it sat in ``ALLOWED_MODULES`` /
``SAFE_PIP_PACKAGES`` / ``get_materializer_namespace()`` / ``requirements.txt``
but was missing from both Dockerfiles → ``ImportError`` in deployed runner
mode (S6). This test parses the three package sets and asserts they agree, so
the drift class cannot recur without a failing test.

The Dockerfiles use dist (pip) names. ``SAFE_PIP_PACKAGES`` is already dist
names, so those compare directly. ``ALLOWED_MODULES`` uses import names that
may differ from the dist name (``bs4`` ← ``beautifulsoup4`` etc.) and includes
stdlib modules that are never pip-installed; the import→dist map + stdlib
exclusion below normalizes it before comparing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.tools.dynamic.allowlist import ALLOWED_MODULES, SAFE_PIP_PACKAGES

# Repo root (this file lives at tests/test_tools/).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNNER_DOCKERFILE = _REPO_ROOT / "Dockerfile.runner"
_TOOLBOX_DOCKERFILE = _REPO_ROOT / "Dockerfile.toolbox"

# Import names that are part of the Python stdlib — never pip-installed, so they
# are excluded from the ALLOWED_MODULES→dist comparison. Includes the dotted
# stdlib subpaths present in ALLOWED_MODULES (html.parser, urllib.parse, …).
_STDLIB_MODULES: frozenset[str] = frozenset({
    "json", "re", "math", "datetime", "pathlib", "collections", "itertools",
    "textwrap", "typing", "dataclasses", "copy", "decimal", "statistics",
    "hashlib", "base64", "urllib.parse", "html.parser", "csv", "io",
    "xml.etree.ElementTree",
})

# Import name → dist name for the cases where they differ (mirrors the comments
# in allowlist.py). Every other non-stdlib import name equals its dist name.
_IMPORT_TO_DIST: dict[str, str] = {
    "bs4": "beautifulsoup4",
    "PIL": "Pillow",
    "yaml": "pyyaml",
    "pydantic_settings": "pydantic-settings",
    "json_repair": "json-repair",
    "markdown_it": "markdown-it-py",
    "sklearn": "scikit-learn",
    "dateutil": "python-dateutil",
    # Phase S7 additions.
    "fitz": "pymupdf",
    "tree_sitter": "tree-sitter",
    "tree_sitter_python": "tree-sitter-python",
    # Phase 5 H1 — formal-verification SMT solver (import name z3 ← z3-solver dist).
    "z3": "z3-solver",
}


def _parse_dockerfile_pip_packages(path: Path) -> set[str]:
    """Parse the dist names from a Dockerfile's ``pip install`` RUN block.

    Handles both the current ``RUN pip install --no-cache-dir \\`` form and the
    BuildKit ``RUN --mount=type=cache,… pip install \\`` form. Captures the
    install line plus its backslash-continuation lines, splits on whitespace,
    and drops pip flags (anything beginning with ``-``).
    """
    lines = path.read_text().splitlines()
    pkgs: list[str] = []
    in_block = False
    for line in lines:
        if not in_block:
            if "pip install" in line:
                in_block = True
                rest = line.split("pip install", 1)[1]
            else:
                continue
        else:
            rest = line
        ended = rest.rstrip().endswith("\\")
        for tok in rest.replace("\\", " ").split():
            if tok.startswith("-"):  # pip flag (--no-cache-dir, etc.)
                continue
            pkgs.append(tok)
        if not ended:
            break
    assert pkgs, f"no pip install block parsed from {path.name}"
    return set(pkgs)


def _allowed_dist_names() -> set[str]:
    """ALLOWED_MODULES import names → the dist names they resolve to (stdlib
    excluded). These are exactly the third-party deps a generated tool can
    import — so each must be pip-installed in the sandbox images."""
    dists: set[str] = set()
    for imp in ALLOWED_MODULES:
        if imp in _STDLIB_MODULES:
            continue
        # A submodule import (e.g. ``matplotlib.pyplot``) resolves to its
        # top-level package's dist name — pip installs the root package, so
        # ``matplotlib.pyplot`` → ``matplotlib`` (already in the image). The
        # special-case map keys are all root names, so splitting first is safe.
        root = imp.split(".", 1)[0]
        dists.add(_IMPORT_TO_DIST.get(root, root))
    return dists


@pytest.fixture(scope="module")
def runner_packages() -> set[str]:
    return _parse_dockerfile_pip_packages(_RUNNER_DOCKERFILE)


@pytest.fixture(scope="module")
def toolbox_packages() -> set[str]:
    return _parse_dockerfile_pip_packages(_TOOLBOX_DOCKERFILE)


class TestAllowlistDockerfileSync:
    def test_runner_and_toolbox_install_identical_dep_set(
        self, runner_packages: set[str], toolbox_packages: set[str]
    ) -> None:
        """The two sandbox images must mirror 1:1 — an allowlisted import must
        resolve identically whether code runs via the runner service or the
        ad-hoc toolbox. A divergence here is a class of latent ImportError."""
        assert runner_packages == toolbox_packages, (
            "Dockerfile.runner and Dockerfile.toolbox pip sets diverge:\n"
            f"  only in runner: {sorted(runner_packages - toolbox_packages)}\n"
            f"  only in toolbox: {sorted(toolbox_packages - runner_packages)}"
        )

    def test_every_allowed_module_is_installed_in_runner(self) -> None:
        """Each non-stdlib allowlisted import must resolve to an installed dist
        in Dockerfile.runner — the drift bug ``unstructured`` hit (allowlisted
        but absent from the image → ImportError in deployed runner mode)."""
        installed = _parse_dockerfile_pip_packages(_RUNNER_DOCKERFILE)
        missing = sorted(_allowed_dist_names() - installed)
        assert not missing, (
            "allowlisted modules missing from Dockerfile.runner (ImportError in "
            f"deployed runner mode): {missing}"
        )

    def test_every_allowed_module_is_installed_in_toolbox(self) -> None:
        """Same invariant for Dockerfile.toolbox (the ad-hoc code_executor
        sandbox image)."""
        installed = _parse_dockerfile_pip_packages(_TOOLBOX_DOCKERFILE)
        missing = sorted(_allowed_dist_names() - installed)
        assert not missing, (
            "allowlisted modules missing from Dockerfile.toolbox (ImportError in "
            f"the toolbox sandbox): {missing}"
        )

    def test_every_safe_pip_package_is_installed_in_both(
        self, runner_packages: set[str], toolbox_packages: set[str]
    ) -> None:
        """SAFE_PIP_PACKAGES (the dist names a generated tool may request to
        pip-install) must already be present in both images — otherwise the
        materializer's pre-imported namespace and the on-demand install path
        disagree."""
        missing_runner = sorted(set(SAFE_PIP_PACKAGES) - runner_packages)
        missing_toolbox = sorted(set(SAFE_PIP_PACKAGES) - toolbox_packages)
        assert not missing_runner and not missing_toolbox, (
            "SAFE_PIP_PACKAGES not fully installed in the sandbox images:\n"
            f"  missing from runner: {missing_runner}\n"
            f"  missing from toolbox: {missing_toolbox}"
        )

    def test_unstructured_present_after_drift_fix(
        self, runner_packages: set[str], toolbox_packages: set[str]
    ) -> None:
        """Direct regression for the S6 drift: ``unstructured`` was allowlisted
        but absent from both Dockerfiles. Pin it explicitly so a future edit
        that drops it fails here with a pointed message."""
        assert "unstructured" in runner_packages
        assert "unstructured" in toolbox_packages

    def test_import_to_dist_map_covers_all_non_stdlib_allowed_modules(self) -> None:
        """Sanity: every non-stdlib ALLOWED_MODULES entry resolves to SOME dist
        (via the map or identity). Catches a new allowlisted import whose dist
        name differs and was forgotten in the map (it would silently compare
        the wrong dist name in the install assertions above)."""
        unknown = [
            imp for imp in ALLOWED_MODULES
            if imp not in _STDLIB_MODULES and imp not in _IMPORT_TO_DIST
            # identity case is fine (import name == dist name): only flag names
            # that contain characters invalid as a dist name AND aren't mapped.
            and not re.match(r"^[A-Za-z0-9_.-]+$", imp)
        ]
        assert not unknown, (
            "non-stdlib ALLOWED_MODULES entries with unmapped import→dist names "
            f"(add them to _IMPORT_TO_DIST): {unknown}"
        )
