"""WarmMemoryStore skill recall (findings-05 C): retrieve_skills.

Mirrors the fact-tier tests. retrieve_skills has two paths — semantic
(generator + query + ranked rows spanning skill/procedure/workflow) and the
fitness fallback (no generator / empty result, a single fitness-ordered pass
across the three capability types). The fitness fallback queries the session
directly (unlike retrieve_facts, which reuses retrieve()), so the fake session
serves ordered row-sets per execute() call. cosine_distance is built as a SQL
expression, never executed against pgvector.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.db.models import WarmMemory
from src.memory.warm import WarmMemoryStore


class _FakeResult:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def all(self) -> list[Any]:
        return self._rows

    def scalars(self) -> _FakeResult:
        # Fitness-fallback path calls result.scalars().all(); the fake session
        # is fed WarmMemory objects directly, so scalars() returns self.
        return self


class _FakeSession:
    """Captures add()/execute() — serves ordered row-sets per execute() call."""

    def __init__(self, row_sets: list[list[Any]] | None = None) -> None:
        self.added: list[Any] = []
        self.executed: list[Any] = []
        self._row_sets = list(row_sets) if row_sets else []

    def add(self, obj: Any) -> None:
        self.added.append(obj)

    async def execute(self, stmt: Any) -> _FakeResult:
        self.executed.append(stmt)
        rows = self._row_sets.pop(0) if self._row_sets else []
        return _FakeResult(rows)

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


class TestRetrieveSkills:
    @pytest.mark.asyncio
    async def test_fitness_fallback_when_no_generator(self) -> None:
        # No generator → semantic path skipped → fitness-ordered fallback.
        entry = WarmMemory(
            memory_type="skill",
            title="utc_normalize",
            content="pd.to_datetime(..., utc=True)",
            fitness_score=0.9,
            tags=["skill", "pandas"],
        )
        session = _FakeSession(row_sets=[[entry]])
        store = WarmMemoryStore(session, generator=None)  # type: ignore[arg-type]

        skills = await store.retrieve_skills(query="convert timestamps", limit=5)

        assert len(skills) == 1
        s = skills[0]
        assert s["type"] == "skill"
        assert s["name"] == "utc_normalize"
        assert s["content"] == "pd.to_datetime(..., utc=True)"
        assert s["fitness_score"] == pytest.approx(0.9)
        # Fallback rows carry no similarity (cosine distance was never computed).
        assert "similarity" not in s

    @pytest.mark.asyncio
    async def test_semantic_path_returns_ranked_skills(self) -> None:
        # Generator + query → semantic join; session returns (WarmMemory, dist).
        entry = WarmMemory(
            memory_type="procedure",
            title="dedup_by_canonical_url",
            content="canonicalize then groupby",
            fitness_score=0.7,
            tags=["procedure"],
        )
        session = _FakeSession(row_sets=[[(entry, 0.25)]])  # cosine distance 0.25
        gen = _FakeGen()
        store = WarmMemoryStore(session, generator=gen)  # type: ignore[arg-type]

        skills = await store.retrieve_skills(query="remove duplicate search results", limit=5)

        assert len(skills) == 1
        s = skills[0]
        assert s["type"] == "procedure"
        assert s["name"] == "dedup_by_canonical_url"
        assert s["content"] == "canonicalize then groupby"
        assert s["fitness_score"] == pytest.approx(0.7)
        assert s["similarity"] == pytest.approx(0.75)  # 1 - 0.25
        # The query was embedded exactly once.
        assert gen.embedded == ["remove duplicate search results"]

    @pytest.mark.asyncio
    async def test_semantic_spans_skill_procedure_workflow(self) -> None:
        # The three capability-shaped types all surface through one semantic pass.
        rows = [
            (WarmMemory(memory_type="skill", title="s", content="cs", fitness_score=0.5), 0.1),
            (WarmMemory(memory_type="workflow", title="w", content="cw", fitness_score=0.5), 0.4),
        ]
        session = _FakeSession(row_sets=[rows])
        store = WarmMemoryStore(session, generator=_FakeGen())  # type: ignore[arg-type]

        skills = await store.retrieve_skills(query="anything", limit=5)
        types = {s["type"] for s in skills}
        assert types == {"skill", "workflow"}

    @pytest.mark.asyncio
    async def test_semantic_empty_falls_back_to_fitness(self) -> None:
        # Generator + query but no embedded skill rows → fitness fallback.
        entry = WarmMemory(
            memory_type="workflow",
            title="plan_csv_pipeline",
            content="read, validate, aggregate",
            fitness_score=0.6,
        )
        # First execute (semantic) returns []; second (fallback) returns [entry].
        session = _FakeSession(row_sets=[[], [entry]])
        store = WarmMemoryStore(session, generator=_FakeGen())  # type: ignore[arg-type]

        skills = await store.retrieve_skills(query="build a csv report", limit=5)

        assert len(skills) == 1
        assert skills[0]["name"] == "plan_csv_pipeline"
        assert "similarity" not in skills[0]  # fallback row, no distance

    @pytest.mark.asyncio
    async def test_empty_query_uses_fitness_fallback(self) -> None:
        # No query text → semantic path skipped even with a generator present.
        entry = WarmMemory(
            memory_type="skill", title="k", content="c", fitness_score=0.8
        )
        gen = _FakeGen()
        session = _FakeSession(row_sets=[[entry]])
        store = WarmMemoryStore(session, generator=gen)  # type: ignore[arg-type]

        skills = await store.retrieve_skills(query="", limit=5)

        assert len(skills) == 1
        assert skills[0]["name"] == "k"
        # Empty query means the generator is never asked to embed.
        assert gen.embedded == []
