"""WarmMemoryStore fact tier (Phase 5): store_fact + retrieve_facts.

store_fact (A5) is now an ``ON CONFLICT (fact_key) DO UPDATE`` upsert — verified
by capturing the emitted statement and compiling it (the WarmMemory JSONB columns
block a real aiosqlite round-trip, so the contract is asserted at the compiled-SQL
layer — the statement that actually runs against Postgres). The migration round-
trip exercises the real DDL/conflict against pgvector. retrieve_facts has two
paths — semantic (generator + query + rows) and the fitness fallback (no
generator / empty result) — both exercised without a real pgvector
(cosine_distance is built as a SQL expression, not executed).
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import patch

import pytest
from sqlalchemy.dialects import postgresql

from src.db.models import WarmMemory
from src.memory.warm import WarmMemoryStore

# store_fact's upsert RETURNING-ed id is read off the result. We don't execute
# against a real DB, so _FakeResult.scalar_one returns this fixed id — the caller
# returns it and the refreshed embedding binds to it.
_FIXED_UUID = uuid.UUID("00000000-0000-0000-0000-000000000001")


class _FakeResult:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def all(self) -> list[Any]:
        return self._rows

    def scalar_one(self) -> Any:
        # store_fact reads the RETURNING-ed id here. Not executed against a real
        # DB, so hand back the fixed id the caller/embedding can use.
        return _FIXED_UUID


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

        assert mem_id  # surviving id returned
        # A5: the fact row is upserted via execute (ON CONFLICT), not add-ed.
        # With a generator, store_fact runs the upsert + the delete-stale-embedding,
        # then adds exactly the one refreshed embedding.
        assert len(session.executed) == 2
        assert len(session.added) == 1  # MemoryEmbedding only
        # JSONB values can't literal-bind at compile time, so assert the fact
        # row's values via the bound params (the statement that runs on Postgres).
        params = (
            session.executed[0]
            .compile(dialect=postgresql.dialect())
            .construct_params()
        )
        assert params["memory_type"] == "fact"
        assert params["title"] == "row_count"
        assert params["content"] == "orders.csv has 1024 rows"
        assert params["fact_key"] == "row_count"
        assert params["fitness_score"] == pytest.approx(0.8)
        assert params["tags"] == ["fact", "data"]
        assert params["extra_data"]["source"] == "fold_1_episode"
        assert params["extra_data"]["confidence"] == pytest.approx(0.8)
        assert params["extra_data"]["fact_key"] == "row_count"
        # Embedding built from "<key>: <value>" for better recall.
        assert gen.embedded == ["row_count: orders.csv has 1024 rows"]

    @pytest.mark.asyncio
    async def test_confidence_is_clamped(self) -> None:
        session = _FakeSession()
        store = WarmMemoryStore(session, generator=None)  # type: ignore[arg-type]

        await store.store_fact("k", "v", confidence=1.5)
        await store.store_fact("k2", "v2", confidence=-0.2)

        # No generator → each store_fact executes only the upsert (no embedding
        # work), so the two upsert statements are executed[0] and executed[1].
        assert len(session.executed) == 2
        p1 = (
            session.executed[0]
            .compile(dialect=postgresql.dialect())
            .construct_params()
        )
        p2 = (
            session.executed[1]
            .compile(dialect=postgresql.dialect())
            .construct_params()
        )
        # Clamped fitness (1.5 → 1.0, -0.2 → 0.0) seeds both the INSERT VALUES
        # fitness_score and the SET fitness_score (store_fact builds one value).
        assert p1["fitness_score"] == pytest.approx(1.0)
        assert p2["fitness_score"] == pytest.approx(0.0)
        # extra_data confidence is clamped too.
        assert p1["extra_data"]["confidence"] == pytest.approx(1.0)
        assert p2["extra_data"]["confidence"] == pytest.approx(0.0)


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


class TestStoreFactDedup:
    """A5: store_fact upserts on fact_key (ON CONFLICT DO UPDATE), not plain INSERT.

    Re-running a goal re-extracts the same fact → the existing row is updated,
    not duplicated. The dedup index is partial (memory_type='fact' AND
    expires_at IS NULL). The WarmMemory JSONB columns block a real aiosqlite
    round-trip, so the contract is asserted on the compiled statement that
    actually runs against Postgres (the real conflict is exercised by the
    migration round-trip on pgvector).
    """

    @pytest.mark.asyncio
    async def test_store_fact_emits_on_conflict_on_fact_key(self) -> None:
        session = _FakeSession()
        store = WarmMemoryStore(session, generator=None)  # type: ignore[arg-type]

        await store.store_fact("row_count", "orders.csv has 1024 rows", confidence=0.8)

        assert session.executed, "store_fact must execute an upsert statement"
        sql = str(
            session.executed[0].compile(dialect=postgresql.dialect())
        )
        # Plain INSERT that conflicts on fact_key …
        assert "INSERT INTO warm_memories" in sql
        assert "ON CONFLICT (fact_key)" in sql
        # … scoped to active facts (the partial-index predicate) …
        assert "memory_type = 'fact' AND expires_at IS NULL" in sql
        # … with an update clause and the surviving id returned.
        assert "DO UPDATE SET" in sql
        assert "RETURNING" in sql
        assert "warm_memories.id" in sql

    @pytest.mark.asyncio
    async def test_upsert_set_clause_refreshes_content_not_identity(self) -> None:
        """The conflict-update refreshes mutable fields but preserves identity."""
        session = _FakeSession()
        store = WarmMemoryStore(session, generator=None)  # type: ignore[arg-type]

        await store.store_fact("row_count", "1024 rows", confidence=0.9)

        # JSONB SET values render as param placeholders (param_N) and can't
        # literal-bind, so assert SET column NAMES on the non-literal SQL.
        sql = str(session.executed[0].compile(dialect=postgresql.dialect()))
        # SET refreshes the mutable fields …
        assert "content =" in sql
        assert "fitness_score =" in sql
        assert "extra_data =" in sql
        assert "tags =" in sql
        assert "updated_at =" in sql  # func.now()
        # … but NOT the identity columns (surviving row keeps id/title/fact_key).
        # "id =" would only appear if id were in the SET clause; it isn't — id is
        # only in the INSERT column list and RETURNING.
        assert "id =" not in sql

    @pytest.mark.asyncio
    async def test_store_fact_refreshes_embedding_on_surviving_id(self) -> None:
        """The RETURNING-ed id flows back and the embedding is refreshed in place."""
        session = _FakeSession()
        gen = _FakeGen()
        store = WarmMemoryStore(session, generator=gen)  # type: ignore[arg-type]

        returned = await store.store_fact("schema", "id,ts,amount")

        # RETURNING-ed id is returned to the caller.
        assert returned == str(_FIXED_UUID)
        # Embedding built from "<key>: <value>" and bound to the surviving id.
        assert gen.embedded == ["schema: id,ts,amount"]
        assert len(session.added) == 1  # the refreshed MemoryEmbedding
        assert session.added[0].memory_id == _FIXED_UUID
        # The stale embedding is deleted before the new one is added.
        assert len(session.executed) == 2  # upsert + delete-stale-embedding
        del_sql = str(session.executed[1].compile())
        assert "DELETE FROM memory_embeddings" in del_sql
