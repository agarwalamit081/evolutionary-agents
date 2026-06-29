"""Tests for the opt-in Lean 4 formal-verification builtin (#17).

The handler is gated exactly like ``git_clone``: flag off OR ``lean`` binary
absent → a clear ``DISABLED:`` no-op; enabled + binary present → a bounded
``lean`` type-check. The subprocess invocation lives in ``_check_with_lean`` so
these tests stub the external binary without touching the asyncio module.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from src.tools.builtin import ALL_TOOL_DEFINITIONS, TOOL_ANNOTATIONS
from src.tools.builtin.lean4_runner import TOOL_DEFINITION, lean4_runner


def _patch_settings(monkeypatch: pytest.MonkeyPatch, enabled: bool, timeout_s: int = 5) -> None:
    """Patch the lazily-read get_settings the handler resolves at call time."""
    monkeypatch.setattr(
        "src.config.settings.get_settings",
        lambda: SimpleNamespace(lean4=SimpleNamespace(enabled=enabled, timeout_s=timeout_s)),
    )


def _patch_lean(monkeypatch: pytest.MonkeyPatch, present: bool) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/lean" if present else None)


class TestLean4RunnerGating:
    async def test_disabled_when_flag_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Flag off short-circuits before the binary probe (mirrors git_clone)."""
        _patch_settings(monkeypatch, enabled=False)
        result = await lean4_runner("theorem t : True := by trivial")
        assert result.startswith("DISABLED:")
        assert "LEAN4_ENABLED=false" in result

    async def test_disabled_when_binary_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Enabled but no 'lean' binary → a clear toolchain-absent DISABLED message."""
        _patch_settings(monkeypatch, enabled=True)
        _patch_lean(monkeypatch, present=False)
        result = await lean4_runner("theorem t : True := by trivial")
        assert result.startswith("DISABLED:")
        assert "lean" in result.lower()

    async def test_empty_code_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_settings(monkeypatch, enabled=True)
        _patch_lean(monkeypatch, present=True)
        result = await lean4_runner("   ")
        assert result == "ERROR: empty lean code"


class TestLean4RunnerCheck:
    async def test_type_check_ok_returns_ok_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_settings(monkeypatch, enabled=True, timeout_s=5)
        _patch_lean(monkeypatch, present=True)

        async def _ok(_bin: str, _code: str, _timeout: int) -> tuple[int, str]:
            return 0, ""

        monkeypatch.setattr("src.tools.builtin.lean4_runner._check_with_lean", _ok)
        payload = json.loads(await lean4_runner("theorem t : True := by trivial"))
        assert payload["status"] == "ok"
        assert payload["returncode"] == 0

    async def test_type_check_error_returns_error_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_settings(monkeypatch, enabled=True)
        _patch_lean(monkeypatch, present=True)

        async def _err(_bin: str, _code: str, _timeout: int) -> tuple[int, str]:
            return 1, "check.lean:1:5: error: unknown identifier 'foo'"

        monkeypatch.setattr("src.tools.builtin.lean4_runner._check_with_lean", _err)
        payload = json.loads(await lean4_runner("foo"))
        assert payload["status"] == "error"
        assert payload["returncode"] == 1
        assert "unknown identifier" in payload["output"]

    async def test_timeout_returns_error_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_settings(monkeypatch, enabled=True, timeout_s=1)
        _patch_lean(monkeypatch, present=True)

        async def _hang(_bin: str, _code: str, _timeout: int) -> tuple[int, str]:
            raise asyncio.TimeoutError()

        monkeypatch.setattr("src.tools.builtin.lean4_runner._check_with_lean", _hang)
        result = await lean4_runner("theorem t : True := by sorry")
        assert result.startswith("ERROR:")
        assert "timed out" in result


class TestLean4RunnerRegistration:
    def test_registered_in_catalog(self) -> None:
        assert "lean4_runner" in {d["name"] for d in ALL_TOOL_DEFINITIONS}

    def test_unique_name_and_description(self) -> None:
        names = [d["name"] for d in ALL_TOOL_DEFINITIONS]
        assert names.count("lean4_runner") == 1
        descs = [d["description"] for d in ALL_TOOL_DEFINITIONS]
        assert descs.count(TOOL_DEFINITION["description"]) == 1

    def test_has_annotation_entry(self) -> None:
        assert "lean4_runner" in TOOL_ANNOTATIONS

    def test_definition_shape(self) -> None:
        assert TOOL_DEFINITION["name"] == "lean4_runner"
        assert TOOL_DEFINITION["handler"] is lean4_runner
        assert TOOL_DEFINITION["cacheable"] is False
        assert TOOL_DEFINITION["parameters"]["required"] == ["lean_code"]
        assert "lean_code" in TOOL_DEFINITION["parameters"]["properties"]
