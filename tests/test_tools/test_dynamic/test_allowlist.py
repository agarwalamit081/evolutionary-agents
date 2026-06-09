"""Tests for the dynamic tool allowlist."""

from __future__ import annotations

from src.tools.dynamic.allowlist import ALLOWED_MODULES, get_materializer_namespace


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
