"""Tests for the arxiv_search builtin — deterministic, no network.

The real ``arxiv`` SDK is an HTTP client, so unit tests mock the dependency
presence check (``importlib.util.find_spec``) and the blocking search
function (``_arxiv_search_sync``) rather than hitting arxiv.org. The
``_result_to_dict`` projection is exercised directly with a fake Result.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.tools.builtin.arxiv_search import (
    _MAX_RESULTS,
    _result_to_dict,
    arxiv_search,
)
from src.tools.builtin.arxiv_search import TOOL_DEFINITION as ARXIV_DEF


def _fake_result(**overrides: object) -> SimpleNamespace:
    """Build a SimpleNamespace mirroring the arxiv.Result attribute surface."""
    base: dict[str, object] = {
        "title": "Attention Is All You Need",
        "authors": [SimpleNamespace(name="A. Vaswani"), SimpleNamespace(name="N. Shazeer")],
        "summary": "We propose a new architecture...",
        "published": datetime(2017, 6, 12, tzinfo=timezone.utc),
        "pdf_url": "http://arxiv.org/pdf/1706.03762v1",
        "entry_id": "http://arxiv.org/abs/1706.03762v1",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _patch_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the lazy presence check report `arxiv` installed (hermetic)."""
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: object())


class TestResultToDict:
    def test_full_projection(self) -> None:
        d = _result_to_dict(_fake_result())
        assert d["title"] == "Attention Is All You Need"
        assert d["authors"] == ["A. Vaswani", "N. Shazeer"]
        assert d["summary"] == "We propose a new architecture..."
        assert d["published"] == "2017-06-12T00:00:00+00:00"
        assert d["pdf_url"].endswith("1706.03762v1")
        assert d["entry_id"].endswith("1706.03762v1")

    def test_missing_fields_coerced_not_raised(self) -> None:
        # A bare namespace with NO result attrs → empty strings/lists, not AttributeError.
        d = _result_to_dict(SimpleNamespace())
        assert d["title"] == ""
        assert d["authors"] == []
        assert d["summary"] == ""
        assert d["published"] == ""
        assert d["pdf_url"] == ""
        assert d["entry_id"] == ""


class TestArxivSearchHandler:
    @pytest.mark.asyncio
    async def test_empty_query_rejected(self) -> None:
        out = await arxiv_search(query="   ")
        assert out.startswith("ERROR: empty arxiv query")

    @pytest.mark.asyncio
    async def test_package_missing_graceful(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(importlib.util, "find_spec", lambda _: None)
        out = await arxiv_search(query="transformers")
        assert out.startswith("ERROR: arxiv package not installed")

    @pytest.mark.asyncio
    async def test_returns_json_array_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_present(monkeypatch)
        canned = [
            _result_to_dict(_fake_result()),
            _result_to_dict(_fake_result(title="Second Paper")),
        ]
        with patch("src.tools.builtin.arxiv_search._arxiv_search_sync", return_value=canned):
            out = await arxiv_search(query="transformer attention", max_results=5)
        rows = json.loads(out)
        assert isinstance(rows, list)
        assert len(rows) == 2
        assert rows[0]["title"] == "Attention Is All You Need"
        assert rows[1]["title"] == "Second Paper"
        assert "published" in rows[0]

    @pytest.mark.asyncio
    async def test_no_results_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_present(monkeypatch)
        with patch("src.tools.builtin.arxiv_search._arxiv_search_sync", return_value=[]):
            out = await arxiv_search(query="obscuratopic")
        assert out.startswith("No arxiv results found")

    @pytest.mark.asyncio
    async def test_sync_failure_returns_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_present(monkeypatch)
        with patch(
            "src.tools.builtin.arxiv_search._arxiv_search_sync",
            side_effect=RuntimeError("HTTP 503"),
        ):
            out = await arxiv_search(query="anything")
        assert out.startswith("ERROR: arxiv search failed")

    @pytest.mark.asyncio
    async def test_max_results_clamped_to_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, int] = {}

        def fake_sync(_query: str, max_results: int) -> list[dict[str, object]]:
            captured["max"] = max_results
            return []

        _patch_present(monkeypatch)
        with patch("src.tools.builtin.arxiv_search._arxiv_search_sync", side_effect=fake_sync):
            await arxiv_search(query="q", max_results=9999)
        assert captured["max"] == _MAX_RESULTS


class TestRegistration:
    def test_definition_shape(self) -> None:
        assert ARXIV_DEF["name"] == "arxiv_search"
        assert ARXIV_DEF["handler"] is arxiv_search
        assert ARXIV_DEF["cacheable"] is True
        params = ARXIV_DEF["parameters"]
        assert params["type"] == "object"
        assert "query" in params["properties"]
        assert params["properties"]["max_results"]["default"] == 5
        assert params["required"] == ["query"]

    def test_registered_in_all_tool_definitions(self) -> None:
        from src.tools.builtin import ALL_TOOL_DEFINITIONS

        names = [d["name"] for d in ALL_TOOL_DEFINITIONS]
        assert "arxiv_search" in names
        # No duplicate names across the whole builtin set.
        assert len(names) == len(set(names))
