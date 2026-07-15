"""src.memory — per-tier cosine-similarity relevance threshold (Q82).

``search_by_embedding`` (cold) and the semantic paths of ``retrieve_skills`` /
``retrieve_facts`` (warm) now take ``min_similarity`` and drop rows whose
cosine similarity (``1 - distance``) falls below it. Rows are returned
distance-asc (similarity-desc), so the post-query Python filter keeps a
contiguous high-quality prefix — DB-agnostic. Default 0.0 = keep all.

cosine_distance is built as a SQL expression but never executed against
pgvector here (the fake session serves ordered row-sets), mirroring the
retrieve_skills / retrieve_facts test conventions.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from src.db.models import ColdMemory as ColdMemoryModel
from src.db.models import WarmMemory
from src.memory.cold import ColdMemory
from src.memory.warm import WarmMemoryStore


class _FakeResult:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def all(self) -> list[Any]:
        return self._rows

    def scalars(self) -> _FakeResult:
        return self


class _FakeSession:
    """Serves ordered row-sets per execute() call."""

    def __init__(self, row_sets: list[list[Any]] | None = None) -> None:
        self._row_sets = list(row_sets) if row_sets else []

    async def execute(self, _stmt: Any) -> _FakeResult:
        rows = self._row_sets.pop(0) if self._row_sets else []
        return _FakeResult(rows)

    async def commit(self) -> None:
        return None


class _FakeGen:
    def __init__(self) -> None:
        self.model = "test-embed"

    async def generate(self, _text: str) -> list[float]:
        return [0.0] * 768


class TestColdThreshold:
    @pytest.mark.asyncio
    async def test_min_similarity_drops_low_similarity_rows(self) -> None:
        m0 = ColdMemoryModel(id=uuid.uuid4(), episode_type="execution", content="high")
        m1 = ColdMemoryModel(id=uuid.uuid4(), episode_type="execution", content="mid")
        m2 = ColdMemoryModel(id=uuid.uuid4(), episode_type="execution", content="low")
        # cosine distances → similarities 0.9 / 0.5 / 0.1.
        session = _FakeSession(row_sets=[[(m0, 0.1), (m1, 0.5), (m2, 0.9)]])
        cold = ColdMemory(session, embedding_dim=768, generator=None)  # type: ignore[arg-type]

        out = await cold.search_by_embedding(
            query_embedding=[0.0] * 768, limit=5, min_similarity=0.6
        )

        # Only the >=0.6 row survives (0.9); the 0.5 and 0.1 are dropped.
        assert [r["content"] for r in out] == ["high"]
        assert out[0]["similarity"] == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_min_similarity_zero_keeps_all(self) -> None:
        m0 = ColdMemoryModel(id=uuid.uuid4(), episode_type="execution", content="high")
        m1 = ColdMemoryModel(id=uuid.uuid4(), episode_type="execution", content="low")
        session = _FakeSession(row_sets=[[(m0, 0.1), (m1, 0.9)]])
        cold = ColdMemory(session, embedding_dim=768, generator=None)  # type: ignore[arg-type]

        out = await cold.search_by_embedding(
            query_embedding=[0.0] * 768, limit=5, min_similarity=0.0
        )

        assert [r["content"] for r in out] == ["high", "low"]

    @pytest.mark.asyncio
    async def test_search_by_query_threads_min_similarity(self) -> None:
        # search_by_query owns the query embed then delegates to search_by_embedding
        # with the caller's min_similarity — verify the floor survives the handoff.
        m0 = ColdMemoryModel(id=uuid.uuid4(), episode_type="execution", content="keep")
        m1 = ColdMemoryModel(id=uuid.uuid4(), episode_type="execution", content="drop")
        session = _FakeSession(row_sets=[[(m0, 0.05), (m1, 0.8)]])
        cold = ColdMemory(session, embedding_dim=768, generator=_FakeGen())  # type: ignore[arg-type]

        out = await cold.search_by_query(query="anything", min_similarity=0.5)

        assert [r["content"] for r in out] == ["keep"]  # sim 0.95 kept, 0.2 dropped


class TestWarmSkillsThreshold:
    @pytest.mark.asyncio
    async def test_retrieve_skills_drops_below_min_similarity(self) -> None:
        e0 = WarmMemory(memory_type="skill", title="s0", content="keep", fitness_score=0.8)
        e1 = WarmMemory(memory_type="skill", title="s1", content="drop", fitness_score=0.8)
        session = _FakeSession(row_sets=[[(e0, 0.2), (e1, 0.9)]])  # sim 0.8 / 0.1
        store = WarmMemoryStore(session, generator=_FakeGen())  # type: ignore[arg-type]

        out = await store.retrieve_skills(query="anything", limit=5, min_similarity=0.5)

        assert [r["name"] for r in out] == ["s0"]
        assert out[0]["similarity"] == pytest.approx(0.8)


class TestWarmFactsThreshold:
    @pytest.mark.asyncio
    async def test_retrieve_facts_drops_below_min_similarity(self) -> None:
        e0 = WarmMemory(
            memory_type="fact", title="f0", content="keep", extra_data={"source": "fold"}
        )
        e1 = WarmMemory(
            memory_type="fact", title="f1", content="drop", extra_data={"source": "fold"}
        )
        session = _FakeSession(row_sets=[[(e0, 0.1), (e1, 0.95)]])  # sim 0.9 / 0.05
        store = WarmMemoryStore(session, generator=_FakeGen())  # type: ignore[arg-type]

        out = await store.retrieve_facts(query="anything", limit=5, min_similarity=0.5)

        assert [r["key"] for r in out] == ["f0"]
        assert out[0]["similarity"] == pytest.approx(0.9)
