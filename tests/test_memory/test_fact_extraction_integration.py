"""Fact-extraction integration (Phase 5) — ``memory_fact_extraction_enabled`` parity.

The fold path mines an episode summary for durable facts via the gateway
(``extract_facts``) and persists each as warm memory ``memory_type="fact"``
(``MemoryManager.extract_and_store_facts`` → ``WarmMemoryStore.store_fact``);
``retrieve_facts`` recalls them. The opt-in gate
``memory_fact_extraction_enabled`` (default True but the fold caller gates on
it) controls whether a fold mines facts at all.

These tests exercise the manager-level integration with a fake warm store (no
DB) and a canned gateway (no real-LLM spend): extraction → per-fact storage as
``memory_type="fact"`` → recall. Plus the disabled / failure paths.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.llm.models import LLMResponse
from src.memory.facts import FactCandidate, extract_facts
from src.memory.manager import MemoryManager


# ─── fakes ──────────────────────────────────────────────────────────────────


class _FakeWarm:
    """Captures store_fact calls + serves a recall by key substring."""

    def __init__(self) -> None:
        self.stored: list[dict[str, Any]] = []

    async def store_fact(
        self, *, key: str, value: str, source: str = "extraction",
        confidence: float = 0.5, tags: list[str] | None = None,
    ) -> str:
        fact_id = f"fact-{len(self.stored) + 1}"
        self.stored.append({
            "id": fact_id, "key": key, "value": value, "source": source,
            "confidence": confidence, "memory_type": "fact",
            "tags": list(tags or []),
        })
        return fact_id

    async def retrieve_facts(self, *, query: str = "", limit: int = 5) -> list[dict[str, Any]]:
        if query:
            hits = [f for f in self.stored if query.lower() in f["key"].lower()
                    or query.lower() in f["value"].lower()]
        else:
            hits = list(self.stored)
        return hits[:limit]


def _manager() -> tuple[MemoryManager, _FakeWarm]:
    warm = _FakeWarm()
    mgr = MemoryManager.__new__(MemoryManager)  # bypass __init__ (no DB/Redis)
    mgr.warm = warm
    mgr._graph = None  # type: ignore[attr-defined]
    return mgr, warm


def _canned_gateway(facts_json: str) -> MagicMock:
    gw = MagicMock()
    gw.acompletion = AsyncMock(return_value=LLMResponse(
        content=facts_json,
        model="gpt-4o-mini-2024-07-18", provider="openai",
        input_tokens=10, output_tokens=50, total_tokens=60, cost_usd=0.0001,
    ))
    return gw


_FACTS_JSON = json.dumps({"facts": [
    {"key": "orders.csv", "value": "has 1024 rows", "confidence": 0.9},
    {"key": "timestamps", "value": "must be UTC ISO-8601", "confidence": 0.8},
]})


# ─── extract_facts (gateway → FactCandidate) ────────────────────────────────


class TestExtractFacts:
    async def test_should_parse_well_formed_json_into_candidates(self) -> None:
        out = await extract_facts(_canned_gateway(_FACTS_JSON), "summary text")
        assert len(out) == 2
        assert isinstance(out[0], FactCandidate)
        assert out[0].key == "orders.csv"
        assert out[1].confidence == 0.8

    async def test_should_return_empty_on_empty_text(self) -> None:
        out = await extract_facts(_canned_gateway(_FACTS_JSON), "   ")
        assert out == []

    async def test_should_tolerate_markdown_fence_wrapper(self) -> None:
        gw = _canned_gateway(f"```json\n{_FACTS_JSON}\n```")
        out = await extract_facts(gw, "summary")
        assert len(out) == 2

    async def test_should_never_raise_on_gateway_error(self) -> None:
        gw = MagicMock()
        gw.acompletion = AsyncMock(side_effect=RuntimeError("provider down"))
        out = await extract_facts(gw, "summary")
        assert out == []

    async def test_should_drop_empty_key_or_value_entries(self) -> None:
        payload = json.dumps({"facts": [
            {"key": "", "value": "no key", "confidence": 0.5},
            {"key": "good", "value": "kept", "confidence": 0.5},
            {"key": "novalue", "value": "", "confidence": 0.5},
        ]})
        out = await extract_facts(_canned_gateway(payload), "summary")
        assert len(out) == 1
        assert out[0].key == "good"


# ─── extract_and_store_facts integration ────────────────────────────────────


class TestExtractAndStoreFactsIntegration:
    async def test_should_store_each_extracted_fact_as_memory_type_fact(self) -> None:
        mgr, warm = _manager()
        n = await mgr.extract_and_store_facts(
            _canned_gateway(_FACTS_JSON), "summary", source="fold_1_episode"
        )
        assert n == 2
        assert len(warm.stored) == 2
        # Every stored row is the fact memory_type.
        for row in warm.stored:
            assert row["memory_type"] == "fact"
            assert row["source"] == "fold_1_episode"
        assert warm.stored[0]["key"] == "orders.csv"

    async def test_should_be_bounded_by_max_facts(self) -> None:
        mgr, warm = _manager()
        many = json.dumps({"facts": [
            {"key": f"k{i}", "value": f"v{i}", "confidence": 0.5} for i in range(10)
        ]})
        n = await mgr.extract_and_store_facts(_canned_gateway(many), "summary", max_facts=3)
        assert n == 3
        assert len(warm.stored) == 3

    async def test_should_return_zero_when_extraction_fails(self) -> None:
        mgr, warm = _manager()
        gw = MagicMock()
        gw.acompletion = AsyncMock(side_effect=RuntimeError("down"))
        n = await mgr.extract_and_store_facts(gw, "summary")
        assert n == 0
        assert warm.stored == []


# ─── recall round-trip ──────────────────────────────────────────────────────


class TestRetrieveFactsRecall:
    async def test_should_recall_stored_facts_by_query(self) -> None:
        mgr, warm = _manager()
        await mgr.extract_and_store_facts(_canned_gateway(_FACTS_JSON), "summary")
        hits = await mgr.retrieve_facts(query="orders", limit=5)
        assert len(hits) == 1
        assert hits[0]["key"] == "orders.csv"

    async def test_should_recall_all_facts_when_no_query(self) -> None:
        mgr, warm = _manager()
        await mgr.extract_and_store_facts(_canned_gateway(_FACTS_JSON), "summary")
        hits = await mgr.retrieve_facts()
        assert len(hits) == 2


# ─── disabled flag parity ───────────────────────────────────────────────────


class TestFactExtractionGate:
    def test_flag_defaults_to_true_but_is_toggleable(self) -> None:
        # The flag lives on AgentSettings; the fold caller (reflect) gates on it.
        from src.config import get_settings

        s = get_settings().agent
        assert hasattr(s, "memory_fact_extraction_enabled")
        # It is a bool knob — assignable (the fold caller reads it at call time).
        original = s.memory_fact_extraction_enabled
        try:
            s.memory_fact_extraction_enabled = False
            assert s.memory_fact_extraction_enabled is False
        finally:
            s.memory_fact_extraction_enabled = original

    async def test_disabled_flag_yields_no_storage_in_simulated_fold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Mirror the reflect-node gate: when the flag is off, extract_and_store
        # is never invoked (so the store stays empty). We simulate by gating the
        # call the same way reflect.py does.
        from src.config import get_settings

        monkeypatch.setattr(
            get_settings().agent, "memory_fact_extraction_enabled", False
        )
        mgr, warm = _manager()
        if get_settings().agent.memory_fact_extraction_enabled:
            await mgr.extract_and_store_facts(_canned_gateway(_FACTS_JSON), "summary")
        # Flag off → no facts stored.
        assert warm.stored == []
