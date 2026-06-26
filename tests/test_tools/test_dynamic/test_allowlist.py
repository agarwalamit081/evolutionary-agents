"""Tests for the dynamic tool allowlist."""

from __future__ import annotations

import importlib

import pytest

from src.tools.dynamic.allowlist import (
    ALLOWED_MODULES,
    SAFE_PIP_PACKAGES,
    get_materializer_namespace,
)


class TestAllowedModules:
    """Tests for the ALLOWED_MODULES set."""

    def test_includes_httpx(self) -> None:
        assert "httpx" in ALLOWED_MODULES

    def test_includes_json(self) -> None:
        assert "json" in ALLOWED_MODULES

    def test_includes_pathlib(self) -> None:
        assert "pathlib" in ALLOWED_MODULES

    def test_includes_datetime(self) -> None:
        assert "datetime" in ALLOWED_MODULES

    def test_excludes_os(self) -> None:
        assert "os" not in ALLOWED_MODULES

    def test_excludes_subprocess(self) -> None:
        assert "subprocess" not in ALLOWED_MODULES

    def test_excludes_sys(self) -> None:
        assert "sys" not in ALLOWED_MODULES

    def test_excludes_socket(self) -> None:
        assert "socket" not in ALLOWED_MODULES

    def test_excludes_eval_exec(self) -> None:
        assert "eval" not in ALLOWED_MODULES
        assert "exec" not in ALLOWED_MODULES


class TestMaterializerNamespace:
    """Tests for the get_materializer_namespace function."""

    def test_namespace_has_json(self) -> None:
        ns = get_materializer_namespace()
        assert "json" in ns
        import json
        assert ns["json"] is json

    def test_namespace_has_re(self) -> None:
        ns = get_materializer_namespace()
        assert "re" in ns
        import re
        assert ns["re"] is re

    def test_namespace_has_math(self) -> None:
        ns = get_materializer_namespace()
        assert "math" in ns
        import math
        assert ns["math"] is math

    def test_namespace_has_pathlib(self) -> None:
        ns = get_materializer_namespace()
        assert "pathlib" in ns

    def test_namespace_lacks_os(self) -> None:
        ns = get_materializer_namespace()
        assert "os" not in ns

    def test_namespace_lacks_subprocess(self) -> None:
        ns = get_materializer_namespace()
        assert "subprocess" not in ns

    def test_namespace_lacks_sys(self) -> None:
        ns = get_materializer_namespace()
        assert "sys" not in ns

    def test_namespace_modules_are_actual_imports(self) -> None:
        ns = get_materializer_namespace()
        # Each value should be an actual module object
        for name in ("json", "re", "math", "datetime"):
            assert hasattr(ns[name], "__name__"), f"{name} is not a module"

    def test_httpx_included_if_installed(self) -> None:
        ns = get_materializer_namespace()
        try:
            import httpx
            assert "httpx" in ns
            assert ns["httpx"] is httpx
        except ImportError:
            assert "httpx" not in ns


class TestExpandedAllowlist:
    """§8: the expanded safe-package set — each new import name is allowed."""

    def test_includes_sympy(self) -> None:
        assert "sympy" in ALLOWED_MODULES

    def test_includes_pydantic(self) -> None:
        assert "pydantic" in ALLOWED_MODULES

    def test_includes_pydantic_settings(self) -> None:
        assert "pydantic_settings" in ALLOWED_MODULES

    def test_includes_orjson(self) -> None:
        assert "orjson" in ALLOWED_MODULES

    def test_includes_json_repair(self) -> None:
        assert "json_repair" in ALLOWED_MODULES

    def test_includes_unstructured(self) -> None:
        assert "unstructured" in ALLOWED_MODULES

    def test_includes_markdown_it(self) -> None:
        assert "markdown_it" in ALLOWED_MODULES

    def test_includes_redis(self) -> None:
        assert "redis" in ALLOWED_MODULES

    def test_still_excludes_dangerous_modules(self) -> None:
        """Expansion must not admit dangerous modules."""
        for name in ("os", "sys", "subprocess", "socket", "shutil", "ctypes"):
            assert name not in ALLOWED_MODULES


class TestSafePipPackages:
    """§8: the dist (pip) names that may be installed into the sandbox."""

    def test_includes_dist_names_for_new_packages(self) -> None:
        for dist in (
            "sympy",
            "pydantic",
            "pydantic-settings",
            "orjson",
            "json-repair",
            "unstructured",
            "markdown-it-py",
            "redis",
        ):
            assert dist in SAFE_PIP_PACKAGES, f"{dist} missing from SAFE_PIP_PACKAGES"

    def test_beautifulsoup4_dist_name_present(self) -> None:
        """beautifulsoup4 is the dist name for the bs4 import."""
        assert "beautifulsoup4" in SAFE_PIP_PACKAGES

    def test_excludes_arbitrary_packages(self) -> None:
        assert "subprocess" not in SAFE_PIP_PACKAGES
        assert "playwright" not in SAFE_PIP_PACKAGES
        assert "selenium" not in SAFE_PIP_PACKAGES


class TestExpandedMaterializerNamespace:
    """§8: each newly-allowlisted module is pre-imported when installed."""

    def test_expanded_packages_in_namespace_when_installed(self) -> None:
        ns = get_materializer_namespace()
        for mod_name in (
            "sympy",
            "pydantic",
            "pydantic_settings",
            "orjson",
            "json_repair",
            "unstructured",
            "markdown_it",
            "redis",
        ):
            try:
                importlib.import_module(mod_name)
            except ImportError:
                # Not installed in this environment — namespace must omit it.
                assert mod_name not in ns, (
                    f"{mod_name} present in namespace but not importable"
                )
            else:
                assert mod_name in ns, (
                    f"{mod_name} installed but missing from materializer namespace"
                )
                assert hasattr(ns[mod_name], "__name__"), (
                    f"{mod_name} namespace entry is not a module"
                )


class TestReadOnlyUtilityPackages:
    """Safe read-only utilities added in the B2 follow-up pass."""

    # (import name, pip dist name) — dist is None when it equals the import name.
    NEW_PACKAGES: list[tuple[str, str | None]] = [
        ("requests", "requests"),
        ("dateutil", "python-dateutil"),
        ("jsonschema", "jsonschema"),
        ("tenacity", "tenacity"),
    ]

    def test_each_new_import_name_is_allowed(self) -> None:
        for import_name, _ in self.NEW_PACKAGES:
            assert import_name in ALLOWED_MODULES, (
                f"{import_name} missing from ALLOWED_MODULES"
            )

    def test_each_new_pip_dist_name_is_installable(self) -> None:
        for _, dist_name in self.NEW_PACKAGES:
            assert dist_name is not None  # every new pkg has a real dist name
            assert dist_name in SAFE_PIP_PACKAGES, (
                f"{dist_name} missing from SAFE_PIP_PACKAGES"
            )

    def test_dateutil_dist_name_is_python_dateutil(self) -> None:
        """The import name ``dateutil`` ships under the python-dateutil dist."""
        assert "python-dateutil" in SAFE_PIP_PACKAGES

    def test_no_duplicates_introduced(self) -> None:
        """Each newly-added import/pip name is unique within its frozenset."""
        # Reconstruct the same frozensets from the new names only and confirm
        # set semantics (dedup) match — guards against accidental double-adds.
        new_imports = {name for name, _ in self.NEW_PACKAGES}
        assert len(new_imports) == len(self.NEW_PACKAGES)
        new_dists = {dist for _, dist in self.NEW_PACKAGES if dist is not None}
        assert len(new_dists) == sum(1 for _, d in self.NEW_PACKAGES if d is not None)


class TestNewPackagesPassSafetyLayer4:
    """A dynamically-generated tool importing each new package passes Layer-4."""

    @pytest.mark.parametrize("import_name", [
        "requests", "dateutil", "jsonschema", "tenacity",
    ])
    def test_import_statement_passes_layer4(
        self, import_name: str,
    ) -> None:
        from src.safety.pipeline import SafetyPipeline

        pipeline = SafetyPipeline()
        code = f"import {import_name}\n"
        result = pipeline._check_imports(
            code, allowlisted=set(ALLOWED_MODULES)
        )
        assert result["passed"] is True, result["issues"]
        assert result["issues"] == []

    @pytest.mark.parametrize("import_name", [
        "requests", "dateutil", "jsonschema", "tenacity",
    ])
    def test_from_import_statement_passes_layer4(
        self, import_name: str,
    ) -> None:
        from src.safety.pipeline import SafetyPipeline

        pipeline = SafetyPipeline()
        code = f"from {import_name} import something\n"
        result = pipeline._check_imports(
            code, allowlisted=set(ALLOWED_MODULES)
        )
        assert result["passed"] is True, result["issues"]
        assert result["issues"] == []


class TestPhase2dAllowlist:
    """Phase 2d (findings-04): installed-but-not-allowed libs now admitted.

    Each new library is bound by BOTH controls — the import name in
    ALLOWED_MODULES and the pip dist name in SAFE_PIP_PACKAGES — and the
    importable ones are pre-imported by the materializer namespace. sklearn's
    import name differs from its dist name (scikit-learn).
    """

    # (import name, pip dist name) — each dist differs only for sklearn.
    NEW_PACKAGES: list[tuple[str, str]] = [
        ("scipy", "scipy"),
        ("sklearn", "scikit-learn"),
        ("openpyxl", "openpyxl"),
        ("tabulate", "tabulate"),
        ("aiofiles", "aiofiles"),
        ("trafilatura", "trafilatura"),
        ("libcst", "libcst"),
    ]

    def test_each_new_import_name_is_allowed(self) -> None:
        for import_name, _ in self.NEW_PACKAGES:
            assert import_name in ALLOWED_MODULES, (
                f"{import_name} missing from ALLOWED_MODULES"
            )

    def test_each_new_pip_dist_name_is_installable(self) -> None:
        for _, dist_name in self.NEW_PACKAGES:
            assert dist_name in SAFE_PIP_PACKAGES, (
                f"{dist_name} missing from SAFE_PIP_PACKAGES"
            )

    def test_sklearn_dist_name_is_scikit_learn(self) -> None:
        """The import name ``sklearn`` ships under the scikit-learn dist."""
        assert "scikit-learn" in SAFE_PIP_PACKAGES
        # and the plain "sklearn" dist name is NOT also present (no shadow dup)
        assert "sklearn" not in SAFE_PIP_PACKAGES

    def test_no_duplicates_introduced(self) -> None:
        """Each newly-added import/pip name is unique within its own list."""
        new_imports = {name for name, _ in self.NEW_PACKAGES}
        assert len(new_imports) == len(self.NEW_PACKAGES)
        new_dists = {dist for _, dist in self.NEW_PACKAGES}
        assert len(new_dists) == len(self.NEW_PACKAGES)

    def test_new_packages_in_materializer_namespace_when_installed(self) -> None:
        ns = get_materializer_namespace()
        for import_name, _ in self.NEW_PACKAGES:
            try:
                importlib.import_module(import_name)
            except ImportError:
                # Not installed in this environment — namespace must omit it.
                assert import_name not in ns, (
                    f"{import_name} present in namespace but not importable"
                )
            else:
                assert import_name in ns, (
                    f"{import_name} installed but missing from materializer namespace"
                )
                assert hasattr(ns[import_name], "__name__"), (
                    f"{import_name} namespace entry is not a module"
                )

    def test_still_excludes_dangerous_modules(self) -> None:
        """The Phase 2d expansion must not admit dangerous modules."""
        for name in ("os", "sys", "subprocess", "socket", "shutil", "ctypes"):
            assert name not in ALLOWED_MODULES


class TestPhase2dPackagesPassSafetyLayer4:
    """Phase 2d: a generated tool importing each new package clears Layer-4."""

    @pytest.mark.parametrize("import_name", [
        "scipy", "sklearn", "openpyxl", "tabulate",
        "aiofiles", "trafilatura", "libcst",
    ])
    def test_import_statement_passes_layer4(self, import_name: str) -> None:
        from src.safety.pipeline import SafetyPipeline

        pipeline = SafetyPipeline()
        code = f"import {import_name}\n"
        result = pipeline._check_imports(
            code, allowlisted=set(ALLOWED_MODULES)
        )
        assert result["passed"] is True, result["issues"]
        assert result["issues"] == []

    @pytest.mark.parametrize("import_name", [
        "scipy", "sklearn", "openpyxl", "tabulate",
        "aiofiles", "trafilatura", "libcst",
    ])
    def test_from_import_statement_passes_layer4(self, import_name: str) -> None:
        from src.safety.pipeline import SafetyPipeline

        pipeline = SafetyPipeline()
        code = f"from {import_name} import something\n"
        result = pipeline._check_imports(
            code, allowlisted=set(ALLOWED_MODULES)
        )
        assert result["passed"] is True, result["issues"]
        assert result["issues"] == []


class TestPhase3D5Allowlist:
    """Phase 3 D5 (findings.md P2): fast HTML/markdown libs admitted.

    The lightweight set (selectolax/mdformat/mistune) replaces markitdown, whose
    core dep ``magika`` pulls ``onnxruntime`` into every image. Each new lib is
    bound by BOTH controls (import name in ALLOWED_MODULES, dist name in
    SAFE_PIP_PACKAGES) and pre-imported by the materializer when installed.
    These share their import name with their dist name.
    """

    NEW_PACKAGES: list[tuple[str, str]] = [
        ("selectolax", "selectolax"),
        ("mdformat", "mdformat"),
        ("mistune", "mistune"),
    ]

    def test_each_new_import_name_is_allowed(self) -> None:
        for import_name, _ in self.NEW_PACKAGES:
            assert import_name in ALLOWED_MODULES, (
                f"{import_name} missing from ALLOWED_MODULES"
            )

    def test_each_new_pip_dist_name_is_installable(self) -> None:
        for _, dist_name in self.NEW_PACKAGES:
            assert dist_name in SAFE_PIP_PACKAGES, (
                f"{dist_name} missing from SAFE_PIP_PACKAGES"
            )

    def test_markitdown_deliberately_excluded(self) -> None:
        """markitdown was rejected — its core dep magika drags onnxruntime in."""
        assert "markitdown" not in ALLOWED_MODULES
        assert "markitdown" not in SAFE_PIP_PACKAGES

    def test_no_duplicates_introduced(self) -> None:
        new_imports = {name for name, _ in self.NEW_PACKAGES}
        assert len(new_imports) == len(self.NEW_PACKAGES)
        new_dists = {dist for _, dist in self.NEW_PACKAGES}
        assert len(new_dists) == len(self.NEW_PACKAGES)

    def test_new_packages_in_materializer_namespace_when_installed(self) -> None:
        ns = get_materializer_namespace()
        for import_name, _ in self.NEW_PACKAGES:
            try:
                importlib.import_module(import_name)
            except ImportError:
                assert import_name not in ns, (
                    f"{import_name} present in namespace but not importable"
                )
            else:
                assert import_name in ns, (
                    f"{import_name} installed but missing from materializer namespace"
                )
                assert hasattr(ns[import_name], "__name__"), (
                    f"{import_name} namespace entry is not a module"
                )

    def test_still_excludes_dangerous_modules(self) -> None:
        for name in ("os", "sys", "subprocess", "socket", "shutil", "ctypes"):
            assert name not in ALLOWED_MODULES


class TestPhase3D5PackagesPassSafetyLayer4:
    """Phase 3 D5: a generated tool importing each new lib clears Layer-4."""

    @pytest.mark.parametrize("import_name", ["selectolax", "mdformat", "mistune"])
    def test_import_statement_passes_layer4(self, import_name: str) -> None:
        from src.safety.pipeline import SafetyPipeline

        pipeline = SafetyPipeline()
        code = f"import {import_name}\n"
        result = pipeline._check_imports(
            code, allowlisted=set(ALLOWED_MODULES)
        )
        assert result["passed"] is True, result["issues"]
        assert result["issues"] == []

    @pytest.mark.parametrize("import_name", ["selectolax", "mdformat", "mistune"])
    def test_from_import_statement_passes_layer4(self, import_name: str) -> None:
        from src.safety.pipeline import SafetyPipeline

        pipeline = SafetyPipeline()
        code = f"from {import_name} import something\n"
        result = pipeline._check_imports(
            code, allowlisted=set(ALLOWED_MODULES)
        )
        assert result["passed"] is True, result["issues"]
        assert result["issues"] == []


class TestPhase3D6Allowlist:
    """Phase 3 D6 (findings.md P2): matplotlib admitted for generated-tool plotting.

    ``matplotlib`` + ``matplotlib.pyplot`` are allowed imports; ``matplotlib`` is
    an installable dist; installed → pre-imported in the materializer namespace.
    The code_executor bootstrap defaults ``MPLBACKEND=Agg`` (covered in
    test_code_executor) so headless ``savefig`` works.
    """

    @pytest.mark.parametrize("import_name", ["matplotlib", "matplotlib.pyplot"])
    def test_matplotlib_import_names_allowed(self, import_name: str) -> None:
        assert import_name in ALLOWED_MODULES

    def test_matplotlib_dist_name_installable(self) -> None:
        assert "matplotlib" in SAFE_PIP_PACKAGES

    def test_matplotlib_in_materializer_namespace_when_installed(self) -> None:
        ns = get_materializer_namespace()
        try:
            importlib.import_module("matplotlib")
        except ImportError:
            assert "matplotlib" not in ns
        else:
            assert "matplotlib" in ns
            assert hasattr(ns["matplotlib"], "__name__")

    @pytest.mark.parametrize("import_name", ["matplotlib", "matplotlib.pyplot"])
    def test_matplotlib_import_passes_layer4(self, import_name: str) -> None:
        from src.safety.pipeline import SafetyPipeline

        pipeline = SafetyPipeline()
        code = f"import {import_name}\n"
        result = pipeline._check_imports(
            code, allowlisted=set(ALLOWED_MODULES)
        )
        assert result["passed"] is True, result["issues"]
        assert result["issues"] == []

    def test_still_excludes_dangerous_modules(self) -> None:
        for name in ("os", "sys", "subprocess", "socket", "shutil", "ctypes"):
            assert name not in ALLOWED_MODULES


class TestBrowserPackagesStayBlocked:
    """Browser-automation packages are deliberately deferred — must stay blocked."""

    @pytest.mark.parametrize("import_name", ["playwright", "selenium"])
    def test_browser_import_not_in_allowlist(self, import_name: str) -> None:
        assert import_name not in ALLOWED_MODULES

    @pytest.mark.parametrize("import_name", ["playwright", "selenium"])
    def test_browser_import_rejected_by_layer4(self, import_name: str) -> None:
        """A generated tool importing a browser pkg is not allowlisted.

        playwright/selenium are not in the hardcoded ``dangerous_modules`` set,
        so Layer-4's import scan alone does not flag them — they are blocked by
        the allowlist contract: absent from ALLOWED_MODULES, not pip-installable
        (SAFE_PIP_PACKAGES), and absent from the materializer namespace. This
        test asserts all three binding controls hold.
        """
        assert import_name not in ALLOWED_MODULES
        assert import_name not in SAFE_PIP_PACKAGES
        assert import_name not in get_materializer_namespace()

    @pytest.mark.parametrize("import_name", ["playwright", "selenium"])
    def test_browser_pkg_not_pip_installable(self, import_name: str) -> None:
        assert import_name not in SAFE_PIP_PACKAGES


class TestPhase5H1Allowlist:
    """Phase 5 H1 (findings.md H1): formal-verification libs admitted.

    ``hypothesis`` (property-based testing) and ``z3`` (SMT solver; import name
    z3 ← z3-solver dist) let sandboxed generated code *verify* invariants
    rather than only assert them — anti-fabrication / invariant-proving. Both
    are pure-compute (no egress); z3-solver ships the compiled solver in its
    wheel, so it needs no separate container (unlike a proof assistant such as
    Lean 4, deliberately deferred in findings.md H1). Each is bound by BOTH
    controls (import name in ALLOWED_MODULES, dist name in SAFE_PIP_PACKAGES)
    and pre-imported by the materializer when installed.
    """

    # (import name, pip dist name) — z3's dist name differs; hypothesis shares.
    NEW_PACKAGES: list[tuple[str, str]] = [
        ("hypothesis", "hypothesis"),
        ("z3", "z3-solver"),
    ]

    def test_each_new_import_name_is_allowed(self) -> None:
        for import_name, _ in self.NEW_PACKAGES:
            assert import_name in ALLOWED_MODULES, (
                f"{import_name} missing from ALLOWED_MODULES"
            )

    def test_each_new_pip_dist_name_is_installable(self) -> None:
        for _, dist_name in self.NEW_PACKAGES:
            assert dist_name in SAFE_PIP_PACKAGES, (
                f"{dist_name} missing from SAFE_PIP_PACKAGES"
            )

    def test_z3_dist_name_is_z3_solver(self) -> None:
        """The import name ``z3`` ships under the z3-solver dist (not ``z3``)."""
        assert "z3-solver" in SAFE_PIP_PACKAGES
        # the bare ``z3`` dist name is NOT present (no shadow dup)
        assert "z3" not in SAFE_PIP_PACKAGES

    def test_no_duplicates_introduced(self) -> None:
        new_imports = {name for name, _ in self.NEW_PACKAGES}
        assert len(new_imports) == len(self.NEW_PACKAGES)
        new_dists = {dist for _, dist in self.NEW_PACKAGES}
        assert len(new_dists) == len(self.NEW_PACKAGES)

    def test_new_packages_in_materializer_namespace_when_installed(self) -> None:
        ns = get_materializer_namespace()
        for import_name, _ in self.NEW_PACKAGES:
            try:
                importlib.import_module(import_name)
            except ImportError:
                # Not installed in this environment — namespace must omit it.
                assert import_name not in ns, (
                    f"{import_name} present in namespace but not importable"
                )
            else:
                assert import_name in ns, (
                    f"{import_name} installed but missing from materializer namespace"
                )
                assert hasattr(ns[import_name], "__name__"), (
                    f"{import_name} namespace entry is not a module"
                )

    def test_still_excludes_dangerous_modules(self) -> None:
        """The H1 expansion must not admit dangerous modules."""
        for name in ("os", "sys", "subprocess", "socket", "shutil", "ctypes"):
            assert name not in ALLOWED_MODULES


class TestPhase5H1PackagesPassSafetyLayer4:
    """Phase 5 H1: a generated tool importing each lib clears Layer-4."""

    @pytest.mark.parametrize("import_name", ["hypothesis", "z3"])
    def test_import_statement_passes_layer4(self, import_name: str) -> None:
        from src.safety.pipeline import SafetyPipeline

        pipeline = SafetyPipeline()
        code = f"import {import_name}\n"
        result = pipeline._check_imports(
            code, allowlisted=set(ALLOWED_MODULES)
        )
        assert result["passed"] is True, result["issues"]
        assert result["issues"] == []

    @pytest.mark.parametrize("import_name", ["hypothesis", "z3"])
    def test_from_import_statement_passes_layer4(self, import_name: str) -> None:
        from src.safety.pipeline import SafetyPipeline

        pipeline = SafetyPipeline()
        code = f"from {import_name} import something\n"
        result = pipeline._check_imports(
            code, allowlisted=set(ALLOWED_MODULES)
        )
        assert result["passed"] is True, result["issues"]
        assert result["issues"] == []
