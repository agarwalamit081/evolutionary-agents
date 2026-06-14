"""Tests for the dynamic tool allowlist."""

from __future__ import annotations

import importlib

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
        assert "requests" not in SAFE_PIP_PACKAGES


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
