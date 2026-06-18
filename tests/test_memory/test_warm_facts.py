"""WarmMemoryStore fact tier (Phase 5): store_fact + retrieve_facts.

The store path is verified with a capturing fake session (asserts the WarmMemory
row carries memory_type="fact", extra_data provenance, and that the embedding is
built from "<key>: <value>"). retrieve_facts has two paths — semantic (generator
+ query + rows) and the fitness fallback (no generator / empty result) — both
exercised without a real pgvector (cosine_distance is built as a SQL expression,
not executed).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from src.db.models import WarmMemory
from src.memory.warm import WarmMemoryStore


class _FakeResult:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def all(self) -> list[Any]:
        return self._rows


class _FakeSession:
    """Captures add()/execute() — never touches a DB."""

    def __init__(self, rows: list[Any] | None = None) -> None:
        self.added: list[Any] = []
        self.executed: list[Any] = []
        self._rows = rows or []

    def add(self, obj: Any) -> None:
        self.added.append(obj)

    async def execute(self, stmt: Any) -> _FakeResult:
        self.executed.append(stmt)
        return _FakeResult(self._rows)

    async def commit(self) -> None:
        return None


class _FakeGen:
    """Records the text it was asked to embed; returns a constant vector."""

    def __init__(self) -> None:
        self.embedded: list[str] = []
        self.model = "test-embed"

    async def generate(self, text: str) -> list[float]:
        self.embedded.append(text)
        return [0.0] * 768


class TestStoreFact:
    @pytest.mark.asyncio
    async def test_writes_fact_row_with_metadata(self) -> None:
        session = _FakeSession()
        gen = _FakeGen()
        store = WarmMemoryStore(session, generator=gen)  # type: ignore[arg-type]

        mem_id = await store.store_fact(
            "row_count",
            "orders.csv has 1024 rows",
            source="fold_1_episode",
            confidence=0.8,
            tags=["data"],
        )

        assert mem_id  # uuid string returned
        assert len(session.added) == 2  # WarmMemory + MemoryEmbedding

        entry = session.added[0]
        assert isinstance(entry, WarmMemory)
        assert entry.memory_type == "fact"
        assert entry.title == "row_count"
        assert entry.content == "orders.csv has 1024 rows"
        assert entry.fitness_score == pytest.approx(0.8)
        assert "fact" in entry.tags and "data" in entry.tags
        assert entry.extra_data["source"] == "fold_1_episode"
        assert entry.extra_data["confidence"] == pytest.approx(0.8)
        assert entry.extra_data["fact_key"] == "row_count"
        # Embedding built from "<key>: <value>" for better recall.
        assert gen.embedded == ["row_count: orders.csv has 1024 rows"]

    @pytest.mark.asyncio
    async def test_confidence_is_clamped(self) -> None:
        session = _FakeSession()
        store = WarmMemoryStore(session, generator=_FakeGen())  # type: ignore[arg-type]

        await store.store_fact("k", "v", confidence=1.5)
        await store.store_fact("k2", "v2", confidence=-0.2)

        # Each store_fact appends [WarmMemory, MemoryEmbedding], so the two
        # WarmMemory rows are at indices 0 and 2.
        assert session.added[0].fitness_score == pytest.approx(1.0)
        assert session.added[2].fitness_score == pytest.approx(0.0)
        # extra_data confidence is clamped too.
        assert session.added[2].extra_data["confidence"] == pytest.approx(0.0)


class TestRetrieveFacts:
    @pytest.mark.asyncio
    async def test_fitness_fallback_when_no_generator(self) -> None:
        # No generator → semantic path skipped → fitness-ordered retrieve().
        session = _FakeSession()
        store = WarmMemoryStore(session, generator=None)  # type: ignore[arg-type]

        fake_rows = [
            {
                "id": "u1",
                "name": "row_count",
                "content": "1024 rows",
                "fitness_score": 0.9,
            }
        ]
        with patch.object(WarmMemoryStore, "retrieve", return_value=fake_rows):
            facts = await store.retrieve_facts(query="how many rows", limit=5)
        assert facts == [
            {
                "id": "u1",
                "key": "row_count",
                "value": "1024 rows",
                "source": "extraction",
                "confidence": 0.9,
            }
        ]

    @pytest.mark.asyncio
    async def test_semantic_path_returns_ranked_facts(self) -> None:
        # Generator + query → semantic join; session returns (WarmMemory, dist).
        entry = WarmMemory(
            memory_type="fact",
            title="schema",
            content="id,ts,amount",
            fitness_score=0.85,
            extra_data={"source": "fold_2_episode", "confidence": 0.85, "fact_key": "schema"},
            tags=["fact"],
        )
        session = _FakeSession(rows=[(entry, 0.2)])  # cosine distance 0.2
        gen = _FakeGen()
        store = WarmMemoryStore(session, generator=gen)  # type: ignore[arg-type]

        facts = await store.retrieve_facts(query="what columns", limit=5)

        assert len(facts) == 1
        f = facts[0]
        assert f["key"] == "schema"
        assert f["value"] == "id,ts,amount"
        assert f["source"] == "fold_2_episode"
        assert f["confidence"] == pytest.approx(0.85)
        assert f["similarity"] == pytest.approx(0.8)  # 1 - 0.2
        # The query was embedded exactly once.
        assert gen.embedded == ["what columns"]

    @pytest.mark.asyncio
    async def test_semantic_empty_falls_back_to_fitness(self) -> None:
        # Generator + query but no embedded fact rows → fall through to retrieve.
        session = _FakeSession(rows=[])  # no semantic matches
        store = WarmMemoryStore(session, generator=_FakeGen())  # type: ignore[arg-type]

        fallback_rows = [{"id": "u", "name": "k", "content": "v", "fitness_score": 0.4}]
        with patch.object(WarmMemoryStore, "retrieve", return_value=fallback_rows):
            facts = await store.retrieve_facts(query="anything", limit=5)
        assert len(facts) == 1
        assert facts[0]["key"] == "k"
