"""Unit tests for scripts/diagnose_codegen.py — the JSON-validity + handler-tail
helpers. Hermetic: no LLM calls, no DB, no keys."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT_NAME = "diagnose_codegen"
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / f"{_SCRIPT_NAME}.py"
_spec = importlib.util.spec_from_file_location(_SCRIPT_NAME, _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
dcg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dcg)  # type: ignore[union-attr]


class TestDiagnoseCodegenHelpers:
    def test_is_direct_json_true_for_valid_object(self) -> None:
        assert dcg._is_direct_json('{"tool_name": "x", "handler_code": "y"}') is True

    def test_is_direct_json_false_for_prose(self) -> None:
        assert dcg._is_direct_json("not json at all") is False

    def test_is_direct_json_false_for_markdown_fenced(self) -> None:
        # Fenced JSON is not directly parseable (the generator's extractor strips
        # fences first); the diagnostic's direct-JSON flag must reflect that.
        assert dcg._is_direct_json('```json\n{"a": 1}\n```') is False

    def test_handler_tail_returns_last_line(self) -> None:
        handler = "async def f() -> str:\n    return 'ok'"
        assert dcg._handler_tail(handler) == "    return 'ok'"

    def test_handler_tail_truncates_long_lines(self) -> None:
        long_line = "    return " + "x" * 200
        assert len(dcg._handler_tail(f"async def f():\n{long_line}")) <= 80

    def test_handler_tail_none_input(self) -> None:
        assert dcg._handler_tail(None) == "(no handler)"
