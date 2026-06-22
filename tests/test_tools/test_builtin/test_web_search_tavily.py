"""Unit tests for the Tavily adapter (S12 — full param surface).

The paid provider adapters share a uniform ``(client, key, query, max_results)``
signature; the Tavily-specific knobs (search_depth/topic/days/domain-filters/
score floor) are read from SearchSettings at call-time via ``_search_settings()``
so the signature stays uniform. These tests mock the HTTP layer with
httpx.MockTransport and assert each knob reaches the request body and that the
per-hit ``score`` filter drops low-relevance results before truncation.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from src.tools.builtin import web_search as ws


def _settings(**tavily: Any) -> SimpleNamespace:
    """Stand-in exposing only the TAVILY_* knobs the adapter reads."""
    base = dict(
        tavily_search_depth="basic",
        tavily_topic="general",
        tavily_days=3,
        tavily_include_domains="",
        tavily_exclude_domains="",
        tavily_min_score=0.0,
    )
    base.update(tavily)
    return SimpleNamespace(**base)


def _patch_settings(monkeypatch: pytest.MonkeyPatch, **tavily: Any) -> None:
    monkeypatch.setattr(ws, "_search_settings", lambda: _settings(**tavily))


def _run(body_capture: dict[str, Any], payload: dict[str, Any]) -> Any:
    """Build a MockTransport that records the POST body and returns ``payload``."""

    def handler(request: httpx.Request) -> httpx.Response:
        body_capture["body"] = json.loads(request.content) if request.content else {}
        return httpx.Response(200, json=payload)

    async def go() -> list[dict[str, str]]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await ws._tavily_fetch(client, "KEY", "the query", 5)

    return go


class TestTavilyRequestBody:
    """Each knob reaches the POST body; defaults are cheap/broad."""

    @pytest.mark.asyncio
    async def test_default_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_settings(monkeypatch)
        cap: dict[str, Any] = {}
        await _run(cap, {"results": []})()
        body = cap["body"]
        assert body["api_key"] == "KEY"
        assert body["query"] == "the query"
        assert body["max_results"] == 5
        assert body["search_depth"] == "basic"
        assert body["topic"] == "general"
        assert "days" not in body  # general topic → no days
        assert "include_domains" not in body  # empty list omitted, NOT []
        assert "exclude_domains" not in body

    @pytest.mark.asyncio
    async def test_advanced_depth(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_settings(monkeypatch, tavily_search_depth="advanced")
        cap: dict[str, Any] = {}
        await _run(cap, {"results": []})()
        assert cap["body"]["search_depth"] == "advanced"

    @pytest.mark.asyncio
    async def test_news_topic_sends_days(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_settings(monkeypatch, tavily_topic="news", tavily_days=7)
        cap: dict[str, Any] = {}
        await _run(cap, {"results": []})()
        assert cap["body"]["topic"] == "news"
        assert cap["body"]["days"] == 7

    @pytest.mark.asyncio
    async def test_general_topic_omits_days_even_when_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """days is a no-op/error for general topic → never sent there."""
        _patch_settings(monkeypatch, tavily_topic="general", tavily_days=7)
        cap: dict[str, Any] = {}
        await _run(cap, {"results": []})()
        assert "days" not in cap["body"]

    @pytest.mark.asyncio
    async def test_days_clamped_to_min_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_settings(monkeypatch, tavily_topic="news", tavily_days=0)
        cap: dict[str, Any] = {}
        await _run(cap, {"results": []})()
        assert cap["body"]["days"] == 1

    @pytest.mark.asyncio
    async def test_domain_lists_parsed_and_lowercased(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(
            monkeypatch,
            tavily_include_domains=" Arxiv.org , b.com ",
            tavily_exclude_domains="c.com",
        )
        cap: dict[str, Any] = {}
        await _run(cap, {"results": []})()
        assert cap["body"]["include_domains"] == ["arxiv.org", "b.com"]
        assert cap["body"]["exclude_domains"] == ["c.com"]

    @pytest.mark.asyncio
    async def test_empty_domain_lists_omitted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty list must be absent (not []), since [] would mean 'no domains'."""
        _patch_settings(monkeypatch, tavily_include_domains=" , ", tavily_exclude_domains="")
        cap: dict[str, Any] = {}
        await _run(cap, {"results": []})()
        assert "include_domains" not in cap["body"]
        assert "exclude_domains" not in cap["body"]

    @pytest.mark.asyncio
    async def test_invalid_depth_topic_fall_back_to_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Garbage enums coerce to the cheap defaults rather than a 4xx."""
        _patch_settings(
            monkeypatch,
            tavily_search_depth="ultra",
            tavily_topic="breaking",
        )
        cap: dict[str, Any] = {}
        await _run(cap, {"results": []})()
        assert cap["body"]["search_depth"] == "basic"
        assert cap["body"]["topic"] == "general"


class TestTavilyScoreFilter:
    """min_score drops low-relevance hits before the max_results cap."""

    @pytest.mark.asyncio
    async def test_filter_drops_below_floor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch, tavily_min_score=0.5)
        payload = {"results": [
            {"title": "good", "url": "https://a", "content": "c", "score": 0.9},
            {"title": "weak", "url": "https://b", "content": "c", "score": 0.2},
        ]}
        rows = await _run({}, payload)()
        assert [r["title"] for r in rows] == ["good"]

    @pytest.mark.asyncio
    async def test_zero_floor_keeps_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default min_score=0 keeps every result, even a missing-score hit."""
        _patch_settings(monkeypatch, tavily_min_score=0.0)
        payload = {"results": [
            {"title": "a", "url": "https://a", "content": "c", "score": 0.0},
            {"title": "b", "url": "https://b", "content": "c"},  # no score field
        ]}
        rows = await _run({}, payload)()
        assert {r["title"] for r in rows} == {"a", "b"}

    @pytest.mark.asyncio
    async def test_truncation_applies_after_filter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Filter runs first, then max_results bounds the surviving count."""
        _patch_settings(monkeypatch, tavily_min_score=0.5)

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"results": [
                {"title": f"t{i}", "url": f"https://{i}", "content": "c", "score": 0.9}
                for i in range(8)
            ]})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            rows = await ws._tavily_fetch(client, "KEY", "q", 3)

        assert len(rows) == 3
        assert [r["title"] for r in rows] == ["t0", "t1", "t2"]

    @pytest.mark.asyncio
    async def test_non_numeric_score_dropped_when_floor_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-numeric score is treated as below-floor and dropped (not a crash)."""
        _patch_settings(monkeypatch, tavily_min_score=0.3)
        payload = {"results": [
            {"title": "good", "url": "https://a", "content": "c", "score": 0.8},
            {"title": "junk", "url": "https://b", "content": "c", "score": "n/a"},
        ]}
        rows = await _run({}, payload)()
        assert [r["title"] for r in rows] == ["good"]
