"""Tests that embeddings are persisted on cold/warm store (§10.2).

cold.store() populates ``ColdMemory.embedding`` (the column
``search_by_embedding`` queries) when no embedding is passed. warm.store()
writes a ``memory_embeddings`` row — the FK-correct home
(``memory_embeddings.memory_id`` → ``warm_memories.id``) — that the review found
empty. A generator is injected via MemoryManager in production; these tests
construct the stores directly with a mock generator.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.db.models import MemoryEmbedding, WarmMemory
from src.memory.cold import ColdMemory
from src.memory.warm import WarmMemoryStore


def _mock_session() -> MagicMock:
    session = MagicMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    return session


def _mock_generator(vec: list[float], model: str = "text-embedding-3-small") -> MagicMock:
    gen = MagicMock()
    gen.generate = AsyncMock(return_value=vec)
    gen.model = model
    return gen


class TestColdMemoryEmbedding:
    """cold.store() generates and persists an embedding (§10.2)."""

    @pytest.mark.asyncio
    async def test_generates_embedding_when_none_passed(self) -> None:
        gen = _mock_generator([0.1] * 768)
        session = _mock_session()
        cold = ColdMemory(session=session, generator=gen)

        await cold.store(episode_type="execution", content="hello world")

        gen.generate.assert_awaited_once_with("hello world")
        added = session.add.call_args.args[0]
        assert added.embedding == [0.1] * 768
        assert added.content == "hello world"

    @pytest.mark.asyncio
    async def test_respects_explicit_embedding(self) -> None:
        """A caller-supplied embedding is used as-is (no generation)."""
        gen = _mock_generator([0.9] * 768)
        session = _mock_session()
        cold = ColdMemory(session=session, generator=gen)

        explicit = [0.5] * 768
        await cold.store(episode_type="execution", content="x", embedding=explicit)

        gen.generate.assert_not_awaited()
        assert session.add.call_args.args[0].embedding == explicit

    @pytest.mark.asyncio
    async def test_without_generator_stays_backward_compatible(self) -> None:
        """No generator → embedding stays None (pre-existing behavior)."""
        session = _mock_session()
        cold = ColdMemory(session=session)

        await cold.store(episode_type="execution", content="x")

        assert session.add.call_args.args[0].embedding is None


class TestColdMemorySemanticSearch:
    """cold.search_by_query embeds the query and delegates to search_by_embedding.

    Semantic recall is the path MemoryManager.retrieve_context now uses for the
    query (Phase C). It owns query embedding so callers need no generator ref.
    """

    @pytest.mark.asyncio
    async def test_embeds_query_and_delegates(self) -> None:
        """The query is embedded, then search_by_embedding is called with that vector."""
        vec = [0.3] * 768
        gen = _mock_generator(vec)
        cold = ColdMemory(session=_mock_session(), generator=gen)
        cold.search_by_embedding = AsyncMock(return_value=[
            {"id": "c1", "content": "match", "similarity": 0.88},
        ])

        results = await cold.search_by_query(query="find me", limit=3)

        gen.generate.assert_awaited_once_with("find me")
        cold.search_by_embedding.assert_awaited_once_with(
            query_embedding=vec, limit=3, min_importance=0.0, episode_type=None,
            min_similarity=0.0,
        )
        assert results == [{"id": "c1", "content": "match", "similarity": 0.88}]

    @pytest.mark.asyncio
    async def test_no_generator_returns_empty(self) -> None:
        """Without a generator, semantic search is unavailable → empty list."""
        cold = ColdMemory(session=_mock_session())
        assert await cold.search_by_query(query="q") == []

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self) -> None:
        """An empty query short-circuits — no embedding, no DB hit."""
        gen = _mock_generator([0.1] * 768)
        cold = ColdMemory(session=_mock_session(), generator=gen)
        cold.search_by_embedding = AsyncMock(return_value=[{"id": "x"}])

        assert await cold.search_by_query(query="") == []
        gen.generate.assert_not_awaited()
        cold.search_by_embedding.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_embedding_failure_returns_empty(self) -> None:
        """An embedding error must not propagate — retrieval degrades to empty."""
        gen = MagicMock()
        gen.generate = AsyncMock(side_effect=RuntimeError("embed service down"))
        cold = ColdMemory(session=_mock_session(), generator=gen)
        cold.search_by_embedding = AsyncMock(return_value=[{"id": "x"}])

        assert await cold.search_by_query(query="q") == []
        cold.search_by_embedding.assert_not_awaited()


class TestWarmMemoryEmbedding:
    """warm.store() writes a memory_embeddings row (§10.2)."""

    @pytest.mark.asyncio
    async def test_writes_memory_embedding_row(self) -> None:
        gen = _mock_generator([0.2] * 768, model="custom-embed-model")
        session = _mock_session()
        warm = WarmMemoryStore(session=session, generator=gen)

        await warm.store(memory_type="skill", name="my-skill", content="do thing")

        assert session.add.call_count == 2  # WarmMemory + MemoryEmbedding
        added = [c.args[0] for c in session.add.call_args_list]
        warm_rows = [o for o in added if isinstance(o, WarmMemory)]
        emb_rows = [o for o in added if isinstance(o, MemoryEmbedding)]
        assert len(warm_rows) == 1
        assert len(emb_rows) == 1
        assert emb_rows[0].embedding == [0.2] * 768
        assert emb_rows[0].embedding_model == "custom-embed-model"
        # The embedding row FKs the warm memory's id.
        assert emb_rows[0].memory_id == warm_rows[0].id

    @pytest.mark.asyncio
    async def test_without_generator_skips_embedding_row(self) -> None:
        """No generator → only the WarmMemory row is written (backward compatible)."""
        session = _mock_session()
        warm = WarmMemoryStore(session=session)

        await warm.store(memory_type="skill", name="s", content="c")

        assert session.add.call_count == 1
        assert isinstance(session.add.call_args.args[0], WarmMemory)
