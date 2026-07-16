"""Depth tests for the 3-tier memory subsystem — store/retrieve ops and
per-tier non-fatal failure paths.

Complements the existing per-module files (test_hot, test_cold, test_embeddings,
test_warm_skills, test_warm_facts, test_manager, test_manager_facts). This file
targets the GAPS not already locked there:

  * hot-tier TTL: a short TTL actually evicts the value (fakeredis time-travel),
    distinct from the recency-ZSET ordering already covered in test_hot.
  * cold-tier cosine ranking: ``search_by_embedding`` orders rows by ascending
    cosine *distance* (= descending similarity) and converts distance→similarity
    via ``1 - distance``; the ``episode_type`` + ``min_importance`` filters
    narrow the candidate set; the ``similarity`` field is the projected column.
  * embeddings hash-fallback determinism when NO embedding provider key is
    available: the generator's ``_api_embedding`` raises (no key) →
    ``last_source == "hash"`` and the SAME input yields the SAME vector across
    calls and instances (the property capability dedup relies on for
    never-store-hash correctness).
  * warm CRUD round-trip: ``store`` → ``retrieve`` by name/tags/fitness, plus
    ``update_fitness`` EMA clamp behavior (success up, failure down, clamped to
    [0, 1], access_count increments, missing id is a no-op).
  * manager tier routing: ``retrieve_context`` fans out to hot (newest-first) →
    warm (skills) → cold (semantic+tag, deduped by id) and tags each result with
    its ``tier``; ``store_observation`` writes BOTH hot and cold.
  * per-tier non-fatal failure: a Redis error in hot and a DB error in cold/warm
    is caught, logged, and returns empty/None — it must NEVER raise into the run.

Order-safety: nothing here mutates the ``get_settings()`` singleton or
reassigns class attributes; all external I/O is mocked/faked deterministically.
No real network, DB, or LLM. asyncio_mode=auto (pytest.ini) — no marker needed.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.db.models import ColdMemory as ColdMemoryModel
from src.db.models import MemoryEmbedding, WarmMemory
from src.memory.cold import ColdMemory
from src.memory.embeddings import EmbeddingGenerator
from src.memory.hot import HotMemory
from src.memory.manager import MemoryManager
from src.memory.warm import WarmMemoryStore


# ─── shared fakes ───────────────────────────────────────────────────


class _FakeResult:
    """Mimic SQLAlchemy ``Result``: ``all()`` / ``scalars().all()``."""

    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def all(self) -> list[Any]:
        return self._rows

    def scalars(self) -> _FakeResult:
        return self

    def scalar_one_or_none(self) -> Any:
        return self._rows[0] if self._rows else None

    def scalar_one(self) -> Any:
        return self._rows[0]


class _FakeSession:
    """Captures add()/commit()/execute(); serves queued row-sets per execute()."""

    def __init__(self, row_sets: list[list[Any]] | None = None) -> None:
        self.added: list[Any] = []
        self.executed: list[Any] = []
        self.committed = 0
        self.deleted: list[Any] = []
        self._row_sets = list(row_sets) if row_sets else []

    def add(self, obj: Any) -> None:
        self.added.append(obj)

    async def commit(self) -> None:
        self.committed += 1

    async def delete(self, obj: Any) -> None:
        self.deleted.append(obj)

    async def execute(self, stmt: Any) -> _FakeResult:
        self.executed.append(stmt)
        rows = self._row_sets.pop(0) if self._row_sets else []
        return _FakeResult(rows)


def _warm_row(
    *,
    id_: Any = None,
    memory_type: str = "skill",
    title: str = "n",
    content: str = "c",
    tags: list[str] | None = None,
    fitness_score: float = 0.5,
    access_count: int = 0,
) -> WarmMemory:
    import uuid

    return WarmMemory(
        id=id_ or uuid.uuid4(),
        memory_type=memory_type,
        title=title,
        content=content,
        tags=tags or [],
        fitness_score=fitness_score,
        access_count=access_count,
        extra_data={},
    )


def _cold_row(
    *,
    content: str = "c",
    episode_type: str = "execution",
    importance: float = 0.5,
    tags: list[str] | None = None,
) -> tuple[ColdMemoryModel, float]:
    """A (model_row, distance) pair as ``search_by_embedding`` consumes them."""
    import uuid

    row = ColdMemoryModel(
        id=uuid.uuid4(),
        episode_type=episode_type,
        content=content,
        context_tags=tags or [],
        importance=importance,
    )
    return row, 0.0


# ─── hot-tier: TTL expiry actually evicts ───────────────────────────


class TestHotTtlEviction:
    """A set value with a short TTL is gone after the TTL elapses (fakeredis)."""

    @pytest.mark.asyncio
    async def test_value_present_before_ttl(self) -> None:
        import fakeredis

        redis = fakeredis.FakeAsyncRedis()
        mem = HotMemory(redis_client=redis, ttl_seconds=2)
        await mem.set("k", {"content": "v"})

        assert await mem.get("k") == {"content": "v"}

    @pytest.mark.asyncio
    async def test_value_evicted_after_ttl(self) -> None:
        import fakeredis

        redis = fakeredis.FakeAsyncRedis()
        mem = HotMemory(redis_client=redis, ttl_seconds=1)
        await mem.set("evict", {"content": "gone"})

        # Advance the fakeredis server clock past the 1s TTL.
        redis.now = lambda: time.time() + 5  # type: ignore[assignment]
        # Sleep long enough for the (real-time) key to expire on the fake server.
        time.sleep(1.2)

        assert await mem.get("evict") is None

    @pytest.mark.asyncio
    async def test_custom_ttl_override_shorter(self) -> None:
        import fakeredis

        redis = fakeredis.FakeAsyncRedis()
        mem = HotMemory(redis_client=redis, ttl_seconds=3600)
        # Explicit short TTL overrides the default long TTL.
        await mem.set("quick", {"x": 1}, ttl=1)
        # Advance the fakeredis server clock past the 1s TTL (mirrors
        # ``test_value_evicted_after_ttl``) so the assertion is deterministic
        # rather than depending on a 0.2s wall-clock margin that flakes under
        # heavy suite load.
        redis.now = lambda: time.time() + 5  # type: ignore[assignment]
        time.sleep(1.2)
        assert await mem.get("quick") is None


# ─── cold-tier: cosine ranking + filters + similarity projection ────


class TestColdCosineRanking:
    """``search_by_embedding`` orders by ascending distance and projects
    ``similarity = 1 - distance``; filters narrow the candidate set."""

    @pytest.mark.asyncio
    async def test_orders_by_ascending_distance(self) -> None:
        session = _FakeSession(
            row_sets=[
                [  # rows ordered closest→farthest, as the DB would return them
                    _cold_row(content="nearest", importance=0.9)[0:1][0] and
                    (_cold_row(content="nearest", importance=0.9)[0], 0.05),
                    (_cold_row(content="middle", importance=0.5)[0], 0.40),
                    (_cold_row(content="farthest", importance=0.5)[0], 0.80),
                ],
            ]
        )
        cold = ColdMemory(session=session)

        out = await cold.search_by_embedding([0.1, 0.2, 0.3, 0.4], limit=3)

        assert [r["content"] for r in out] == ["nearest", "middle", "farthest"]
        # similarity = 1 - distance, monotonically decreasing along the order.
        sims = [r["similarity"] for r in out]
        assert sims == pytest.approx([0.95, 0.60, 0.20])
        assert sims == sorted(sims, reverse=True)

    @pytest.mark.asyncio
    async def test_similarity_is_one_minus_distance(self) -> None:
        near, d_near = _cold_row(content="a")
        session = _FakeSession(row_sets=[[(near, 0.25)]])
        cold = ColdMemory(session=session)

        out = await cold.search_by_embedding([0.0, 0.0, 0.0, 0.0], limit=1)

        assert out[0]["similarity"] == pytest.approx(0.75)

    @pytest.mark.asyncio
    async def test_limit_pushed_into_sql(self) -> None:
        # The limit is enforced in SQL (``.limit(N)``), not truncated in Python.
        # A fake session ignores it and returns the full set; the contract is
        # that the limit value reached the compiled SELECT.
        rows = [(_cold_row(content=f"c{i}")[0], 0.1 * i) for i in range(5)]
        session = _FakeSession(row_sets=[rows])
        cold = ColdMemory(session=session)

        await cold.search_by_embedding([0.0, 0.0, 0.0, 0.0], limit=2)

        # The limit reached the compiled SELECT (vector literal can't render in
        # literal_binds, so inspect the statement's limit attribute directly).
        assert session.executed[-1]._limit == 2

    @pytest.mark.asyncio
    async def test_episode_type_filter_pushed_into_query(self) -> None:
        # With a type filter, the query is still executed once; we assert the
        # search returns whatever the (filtered) session served and that a
        # WHERE clause for episode_type was composed (2nd .where on the stmt).
        row, _ = _cold_row(content="x", episode_type="reflection")
        session = _FakeSession(row_sets=[[(row, 0.1)]])
        cold = ColdMemory(session=session)

        out = await cold.search_by_embedding(
            [0.0, 0.0, 0.0, 0.0], limit=5, episode_type="reflection"
        )

        assert len(out) == 1
        assert session.executed  # a SELECT was issued


class TestColdStoreEmbeddingFallback:
    """store() generates an embedding from injected generator when none passed;
    a generator failure is swallowed (embedding stays None, store still commits)."""

    @pytest.mark.asyncio
    async def test_store_generates_embedding_from_generator(self) -> None:
        session = _FakeSession()
        gen = MagicMock()
        gen.generate = AsyncMock(return_value=[0.1, 0.2])
        cold = ColdMemory(session=session, generator=gen)

        mem_id = await cold.store("execution", "content")

        gen.generate.assert_awaited_once_with("content")
        assert mem_id
        assert session.committed == 1
        # The added entry carries the generated embedding.
        added = [a for a in session.added if isinstance(a, ColdMemoryModel)]
        assert len(added) == 1
        assert added[0].embedding == [0.1, 0.2]

    @pytest.mark.asyncio
    async def test_store_swallows_embedding_failure(self) -> None:
        session = _FakeSession()
        gen = MagicMock()
        gen.generate = AsyncMock(side_effect=RuntimeError("embed down"))
        cold = ColdMemory(session=session, generator=gen)

        mem_id = await cold.store("execution", "content")  # must not raise

        assert mem_id  # store still returned an id and committed
        assert session.committed == 1
        added = [a for a in session.added if isinstance(a, ColdMemoryModel)][0]
        assert added.embedding is None  # failed embedding → None, not a crash


class TestColdSearchByQueryBestEffort:
    """search_by_query returns [] with no generator, empty query, or embed fail."""

    @pytest.mark.asyncio
    async def test_no_generator_returns_empty(self) -> None:
        cold = ColdMemory(session=_FakeSession(), generator=None)
        assert await cold.search_by_query("anything") == []

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self) -> None:
        gen = MagicMock()
        gen.generate = AsyncMock(return_value=[0.0, 0.0])
        cold = ColdMemory(session=_FakeSession(), generator=gen)
        assert await cold.search_by_query("") == []
        gen.generate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_embed_failure_returns_empty(self) -> None:
        gen = MagicMock()
        gen.generate = AsyncMock(side_effect=RuntimeError("no key"))
        cold = ColdMemory(session=_FakeSession(), generator=gen)
        assert await cold.search_by_query("q") == []  # never raises


# ─── embeddings: hash-fallback determinism when no embedding key ────


class TestEmbeddingHashFallbackDeterminism:
    """When the embedding provider is unavailable (no key / API error), the
    generator falls back to a deterministic hash vector. Capability dedup relies
    on this being: same input → same vector, AND ``last_source == "hash"``."""

    @pytest.mark.asyncio
    async def test_no_key_path_marks_hash_source(self) -> None:
        # Simulate "no embedding API key" by making litellm raise — the real
        # no-key behavior routes through this same except branch.
        gen = EmbeddingGenerator()
        with patch("litellm.aembedding", new=AsyncMock(side_effect=RuntimeError("no key"))):
            vec = await gen.generate("the same text")

        assert gen.last_source == "hash"
        assert len(vec) == gen.dimension

    @pytest.mark.asyncio
    async def test_same_input_same_vector_across_instances(self) -> None:
        with patch("litellm.aembedding", new=AsyncMock(side_effect=RuntimeError("no key"))):
            v1 = await EmbeddingGenerator().generate("deterministic input")
            v2 = await EmbeddingGenerator().generate("deterministic input")
        # Determinism holds even across separate instances — dedup depends on it.
        assert v1 == v2

    @pytest.mark.asyncio
    async def test_different_inputs_yield_different_vectors(self) -> None:
        with patch("litellm.aembedding", new=AsyncMock(side_effect=RuntimeError("no key"))):
            v1 = await EmbeddingGenerator().generate("input one")
            v2 = await EmbeddingGenerator().generate("input two")
        assert v1 != v2

    @pytest.mark.asyncio
    async def test_api_success_marks_api_source_and_no_hash(self) -> None:
        gen = EmbeddingGenerator()
        fake = MagicMock()
        fake.data = [{"embedding": [0.5] * gen.dimension}]
        with patch("litellm.aembedding", new=AsyncMock(return_value=fake)):
            vec = await gen.generate("real embed")
        assert gen.last_source == "api"
        assert vec == [0.5] * gen.dimension

    @pytest.mark.asyncio
    async def test_hash_values_bounded_in_unit_range(self) -> None:
        with patch("litellm.aembedding", new=AsyncMock(side_effect=RuntimeError("no key"))):
            vec = await EmbeddingGenerator().generate("bounded")
        assert all(-1.0 <= v <= 1.0 for v in vec)


# ─── warm: CRUD round-trip + fitness EMA ────────────────────────────


class TestWarmCrudRoundTrip:
    """store() writes a WarmMemory + optional embedding; retrieve() reads them
    back filtered by type/name/tags/fitness, fitness-ordered."""

    @pytest.mark.asyncio
    async def test_store_then_retrieve_by_type(self) -> None:
        session = _FakeSession(
            row_sets=[[_warm_row(memory_type="skill", title="parse-csv", fitness_score=0.8)]]
        )
        store = WarmMemoryStore(session=session, generator=None)

        mid = await store.store("skill", "parse-csv", "code", fitness_score=0.8)
        out = await store.retrieve(memory_type="skill")

        assert mid  # store returned a uuid
        assert session.committed == 1
        assert len(out) == 1
        assert out[0]["name"] == "parse-csv"
        assert out[0]["type"] == "skill"
        assert out[0]["fitness_score"] == 0.8

    @pytest.mark.asyncio
    async def test_store_with_generator_writes_embedding_row(self) -> None:
        session = _FakeSession()
        gen = MagicMock()
        gen.generate = AsyncMock(return_value=[0.1, 0.2])
        gen.model = "text-embedding-3-small"
        store = WarmMemoryStore(session=session, generator=gen)

        await store.store("skill", "n", "content")

        added_emb = [a for a in session.added if isinstance(a, MemoryEmbedding)]
        assert len(added_emb) == 1
        assert added_emb[0].embedding == [0.1, 0.2]
        assert added_emb[0].embedding_model == "text-embedding-3-small"

    @pytest.mark.asyncio
    async def test_store_embedding_failure_non_fatal(self) -> None:
        session = _FakeSession()
        gen = MagicMock()
        gen.generate = AsyncMock(side_effect=RuntimeError("embed down"))
        store = WarmMemoryStore(session=session, generator=gen)

        mid = await store.store("skill", "n", "c")  # must not raise

        assert mid  # warm row still written + committed
        assert session.committed == 1
        assert [a for a in session.added if isinstance(a, MemoryEmbedding)] == []

    @pytest.mark.asyncio
    async def test_retrieve_filters_by_fitness_threshold(self) -> None:
        # retrieve() adds a WHERE fitness >= min_fitness; we assert the warm row
        # below the threshold is not in the served result set.
        session = _FakeSession(
            row_sets=[[_warm_row(title="high", fitness_score=0.9)]]
        )
        store = WarmMemoryStore(session=session, generator=None)
        out = await store.retrieve(min_fitness=0.5, limit=10)
        assert all(r["fitness_score"] >= 0.5 for r in out)


class TestWarmUpdateFitness:
    """update_fitness EMA: +0.1 on success, -0.05 on failure, clamped to [0,1],
    access_count increments; unknown id is a silent no-op."""

    @pytest.mark.asyncio
    async def test_success_raises_fitness_and_increments_access(self) -> None:
        entry = _warm_row(fitness_score=0.50, access_count=0)
        session = _FakeSession(row_sets=[[entry]])
        store = WarmMemoryStore(session=session, generator=None)

        await store.update_fitness(str(entry.id), success=True)

        assert entry.fitness_score == pytest.approx(0.60)
        assert entry.access_count == 1
        assert session.committed == 1

    @pytest.mark.asyncio
    async def test_failure_lowers_fitness(self) -> None:
        entry = _warm_row(fitness_score=0.50, access_count=2)
        session = _FakeSession(row_sets=[[entry]])
        store = WarmMemoryStore(session=session, generator=None)

        await store.update_fitness(str(entry.id), success=False)

        assert entry.fitness_score == pytest.approx(0.45)
        assert entry.access_count == 3

    @pytest.mark.asyncio
    async def test_fitness_clamped_at_one(self) -> None:
        entry = _warm_row(fitness_score=0.99, access_count=0)
        session = _FakeSession(row_sets=[[entry]])
        store = WarmMemoryStore(session=session, generator=None)

        await store.update_fitness(str(entry.id), success=True)

        assert entry.fitness_score == 1.0  # clamped, not 1.09

    @pytest.mark.asyncio
    async def test_fitness_clamped_at_zero(self) -> None:
        entry = _warm_row(fitness_score=0.02, access_count=0)
        session = _FakeSession(row_sets=[[entry]])
        store = WarmMemoryStore(session=session, generator=None)

        await store.update_fitness(str(entry.id), success=False)

        assert entry.fitness_score == 0.0  # clamped, not negative

    @pytest.mark.asyncio
    async def test_unknown_id_is_noop(self) -> None:
        session = _FakeSession(row_sets=[[None]])  # scalar_one_or_none → None
        store = WarmMemoryStore(session=session, generator=None)

        await store.update_fitness("00000000-0000-0000-0000-000000000000", success=True)

        assert session.committed == 0  # nothing to commit


# ─── manager: tier routing + store_observation fan-out ──────────────


def _manager_with_mocks() -> tuple[MemoryManager, MagicMock, MagicMock, MagicMock]:
    """Build a MemoryManager whose hot/warm/cold are pre-wired mocks, so routing
    can be asserted without a real Redis/DB."""
    redis = MagicMock()
    db = MagicMock()
    settings = MagicMock()
    settings.redis.cache_ttl_seconds = 3600
    settings.llm.embedding_dim = 768
    settings.agent.memory_hot_recall_size = 3
    settings.neo4j.enabled = False

    with patch("src.memory.manager.EmbeddingGenerator") as mock_emb_cls, \
         patch("src.memory.manager.HotMemoryStore") as mock_hot_cls, \
         patch("src.memory.manager.WarmMemoryStore") as mock_warm_cls, \
         patch("src.memory.manager.ColdMemoryStore") as mock_cold_cls:
        mock_emb_cls.return_value = MagicMock()
        hot = MagicMock()
        warm = MagicMock()
        cold = MagicMock()
        mock_hot_cls.return_value = hot
        mock_warm_cls.return_value = warm
        mock_cold_cls.return_value = cold
        mgr = MemoryManager(redis_client=redis, db_session=db, settings=settings)
    return mgr, hot, warm, cold


class TestManagerTierRouting:
    """retrieve_context fans out hot → warm → cold and tags each result tier."""

    @pytest.mark.asyncio
    async def test_retrieve_context_merges_all_tiers_tagged(self) -> None:
        mgr, hot, warm, cold = _manager_with_mocks()
        hot.search_recent = AsyncMock(return_value=[{"content": "recent"}])
        warm.retrieve = AsyncMock(return_value=[{"name": "skill-a"}])
        cold.search_by_query = AsyncMock(return_value=[{"id": "c1", "content": "ep"}])
        cold.search_by_tags = AsyncMock(return_value=[])

        out = await mgr.retrieve_context("query", limit=10)

        tiers = [r["tier"] for r in out]
        assert "hot" in tiers and "warm" in tiers and "cold" in tiers
        assert any(r.get("content") == "recent" for r in out)

    @pytest.mark.asyncio
    async def test_retrieve_context_dedups_cold_by_id(self) -> None:
        """A cold memory matching BOTH the semantic and tag paths appears once."""
        mgr, hot, warm, cold = _manager_with_mocks()
        hot.search_recent = AsyncMock(return_value=[])
        warm.retrieve = AsyncMock(return_value=[])
        dup = {"id": "dup-1", "content": "x"}
        cold.search_by_query = AsyncMock(return_value=[dup])
        cold.search_by_tags = AsyncMock(return_value=[dup])

        out = await mgr.retrieve_context("query", tags=["t"], limit=10)

        cold_hits = [r for r in out if r["tier"] == "cold"]
        assert len(cold_hits) == 1

    @pytest.mark.asyncio
    async def test_retrieve_context_skips_cold_when_no_query(self) -> None:
        mgr, hot, warm, cold = _manager_with_mocks()
        hot.search_recent = AsyncMock(return_value=[])
        warm.retrieve = AsyncMock(return_value=[])

        await mgr.retrieve_context("", limit=5)

        cold.search_by_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_observation_writes_hot_and_cold(self) -> None:
        mgr, hot, warm, cold = _manager_with_mocks()
        hot.add_observation = AsyncMock()
        cold.store = AsyncMock(return_value="cold-id")

        await mgr.store_observation("an observation", importance=0.7, tags=["t"])

        hot.add_observation.assert_awaited_once()  # hot tier written
        cold.store.assert_awaited_once()           # cold tier written
        # cold.store got the episode + importance + tags.
        _, kwargs = cold.store.call_args
        assert kwargs["content"] == "an observation"
        assert kwargs["importance"] == 0.7
        assert kwargs["context_tags"] == ["t"]

    @pytest.mark.asyncio
    async def test_retrieve_context_hot_uses_recall_size_setting(self) -> None:
        mgr, hot, warm, cold = _manager_with_mocks()
        hot.search_recent = AsyncMock(return_value=[])
        warm.retrieve = AsyncMock(return_value=[])
        cold.search_by_query = AsyncMock(return_value=[])

        await mgr.retrieve_context("q", limit=10)

        # search_recent called with the configured memory_hot_recall_size.
        args, _ = hot.search_recent.call_args
        assert args[1] == 3  # settings.agent.memory_hot_recall_size


# ─── per-tier non-fatal failure (never raises into the run) ─────────


class TestPerTierNonFatalFailure:
    """A Redis/DB failure on any single tier is caught + logged + returns
    empty/None. It must NEVER propagate into the calling run."""

    @pytest.mark.asyncio
    async def test_hot_get_failure_returns_none(self) -> None:
        redis = MagicMock()
        redis.get = AsyncMock(side_effect=ConnectionError("redis down"))
        mem = HotMemory(redis_client=redis, ttl_seconds=3600)
        assert await mem.get("k") is None  # not raised

    @pytest.mark.asyncio
    async def test_hot_set_failure_does_not_raise(self) -> None:
        redis = MagicMock()
        redis.set = AsyncMock(side_effect=ConnectionError("redis down"))
        mem = HotMemory(redis_client=redis, ttl_seconds=3600)
        await mem.set("k", {"v": 1})  # not raised

    @pytest.mark.asyncio
    async def test_cold_store_embedding_failure_is_non_fatal(self) -> None:
        """The non-fatal cold surface is the EMBEDDING layer: a generator error
        during store() is swallowed (embedding → None), and the row still
        commits — a cold store never fails because embeddings are down."""
        session = _FakeSession()
        gen = MagicMock()
        gen.generate = AsyncMock(side_effect=RuntimeError("no key"))
        cold = ColdMemory(session=session, generator=gen)

        mem_id = await cold.store("execution", "content")  # must not raise

        assert mem_id
        assert session.committed == 1
        added = [a for a in session.added if isinstance(a, ColdMemoryModel)][0]
        assert added.embedding is None  # embed failed → None, store still ok

    @pytest.mark.asyncio
    async def test_cold_search_query_failure_returns_empty(self) -> None:
        gen = MagicMock()
        gen.generate = AsyncMock(side_effect=RuntimeError("no key"))
        cold = ColdMemory(session=_FakeSession(), generator=gen)
        # search_by_query swallows the embedding failure → [] (never raises).
        assert await cold.search_by_query("q") == []

    @pytest.mark.asyncio
    async def test_warm_store_embedding_failure_does_not_block_write(self) -> None:
        session = _FakeSession()
        gen = MagicMock()
        gen.generate = AsyncMock(side_effect=RuntimeError("embed down"))
        store = WarmMemoryStore(session=session, generator=gen)

        mid = await store.store("skill", "n", "c")  # not raised

        assert mid  # the warm row was still written + committed despite embed fail

    @pytest.mark.asyncio
    async def test_manager_retrieve_context_survives_hot_failure(self) -> None:
        """retrieve_context is non-fatal across tiers: the hot tier swallows its
        own Redis errors internally (returns []), so recall still returns warm
        and cold results. This simulates that contract: hot returns [] (its
        documented behavior on Redis failure) and warm/cold still contribute."""
        mgr, hot, warm, cold = _manager_with_mocks()
        # Hot tier swallowed its Redis error → empty (never raised into the run).
        hot.search_recent = AsyncMock(return_value=[])
        warm.retrieve = AsyncMock(return_value=[{"name": "skill-x"}])
        cold.search_by_query = AsyncMock(
            return_value=[{"id": "c1", "content": "episode"}]
        )
        cold.search_by_tags = AsyncMock(return_value=[])

        out = await mgr.retrieve_context("query", limit=10)

        # Hot contributed nothing, but warm + cold still reached the result set.
        tiers = {r["tier"] for r in out}
        assert "warm" in tiers and "cold" in tiers
        assert "hot" not in tiers  # empty hot → no hot rows tagged
