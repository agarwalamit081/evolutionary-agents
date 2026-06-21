"""Unit tests for scripts/backfill_capability_embeddings.py — pure helpers.

These test only the *pure* logic (no DB, no embedding API): the text-selection
preference and the api-only store decision. The DB/embedding path is an external
dependency exercised elsewhere; the helpers are hermetic and deterministic.

The script lives in ``scripts/`` (not the ``src`` package), so it is loaded via
``importlib`` — the same convention as ``test_backfill_cold_embeddings.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT_NAME = "backfill_capability_embeddings"
_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "backfill_capability_embeddings.py"
)
_spec = importlib.util.spec_from_file_location(_SCRIPT_NAME, _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
backfill = importlib.util.module_from_spec(_spec)
sys.modules[_SCRIPT_NAME] = backfill
_spec.loader.exec_module(backfill)


# ─── select_embedding_text ───────────────────────────────────────────


class TestSelectEmbeddingText:
    """Text selection prefers capability_text, then name+description, else None."""

    def test_prefers_capability_text(self) -> None:
        """An explicit capability_text wins over name/description synthesis."""
        out = backfill.select_embedding_text(
            "official capability blurb", "tool", "a description"
        )
        assert out == "official capability blurb"

    def test_falls_back_to_name_and_description(self) -> None:
        """No capability_text → embed "{name}: {description}"."""
        out = backfill.select_embedding_text(None, "web_search", "Searches the web")
        assert out == "web_search: Searches the web"

    def test_falls_back_when_capability_text_blank(self) -> None:
        """A whitespace-only capability_text is treated as empty (falls through)."""
        out = backfill.select_embedding_text("   \n\t ", "tool", "desc")
        assert out == "tool: desc"

    def test_returns_none_when_all_empty(self) -> None:
        """No usable text anywhere → None (row is skipped, left NULL)."""
        assert backfill.select_embedding_text(None, None, None) is None

    def test_returns_none_when_name_or_description_missing(self) -> None:
        """Synthesis needs both name AND description; missing either → None."""
        assert backfill.select_embedding_text(None, "name", None) is None
        assert backfill.select_embedding_text(None, None, "desc") is None

    def test_returns_none_for_blank_name_or_description(self) -> None:
        """Whitespace-only name/description cannot synthesize → None."""
        assert backfill.select_embedding_text(None, "  ", "desc") is None
        assert backfill.select_embedding_text(None, "name", "  ") is None

    def test_blank_capability_text_with_no_synthesis_is_none(self) -> None:
        """Blank capability_text and nothing to synthesize from → None."""
        assert backfill.select_embedding_text("   ", None, None) is None


# ─── should_store ────────────────────────────────────────────────────


class TestShouldStore:
    """Only "api" vectors are stored; hash/None/unknown are skipped."""

    def test_true_for_api(self) -> None:
        assert backfill.should_store("api") is True

    def test_false_for_hash(self) -> None:
        """Hash vectors are not semantically meaningful — never stored."""
        assert backfill.should_store("hash") is False

    def test_false_for_none(self) -> None:
        assert backfill.should_store(None) is False

    def test_false_for_unknown_source(self) -> None:
        """Any source other than "api" (e.g. a future label) is not stored."""
        assert backfill.should_store("anything-else") is False

    @pytest.mark.parametrize("source", ["api"])
    def test_only_api_is_truthy(self, source: str) -> None:
        assert backfill.should_store(source) is True
