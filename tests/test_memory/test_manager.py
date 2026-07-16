"""Tests for src.memory.manager — MemoryManager across all 3 tiers."""

from __future__ import annotations

from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.memory.manager import MemoryManager


@pytest.fixture
def mock_redis() -> MagicMock:
    """Create a mock Redis client."""
    return MagicMock()


@pytest.fixture
def mock_db_session() -> MagicMock:
    """Create a mock AsyncSession."""
    session = MagicMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.execute = AsyncMock()
    session.delete = AsyncMock()
    return session


@pytest.fixture
def mock_settings() -> MagicMock:
    """Create mock Settings with required nested groups."""
    settings = MagicMock()
    settings.redis.cache_ttl_seconds = 3600
    settings.budget.max_cost_usd = 10.0
    # Embedding config read by MemoryManager (§10.2).
    settings.llm.embedding_dim = 768
    # A4: hot-recall count read by retrieve_context (settings.agent.memory_hot_recall_size).
    settings.agent.memory_hot_recall_size = 3
    # I3: Neo4j graph mirror stays OFF in unit tests (default-off in prod too) so
    # store_skill/store_fact don't attempt a real driver/connectivity check.
    settings.neo4j.enabled = False
    # Phase 9 (Q81-84): memory recall/consolidate knobs read by retrieve_context /
    # retrieve_facts / retrieve_skills / consolidate — defaults reproduce today's
    # behavior so existing assertions stay valid.
    settings.memory.recall_min_similarity = 0.0
    settings.memory.consolidate_max_age_days = 90
    settings.memory.consolidate_min_importance = 0.1
    return settings


@pytest.fixture
def mock_hot() -> MagicMock:
    """Create a mock HotMemoryStore."""
    hot = MagicMock()
    hot.set = AsyncMock()
    hot.add_observation = AsyncMock()
    hot.get = AsyncMock(return_value=None)
    hot.search = AsyncMock(return_value=[])
    hot.search_recent = AsyncMock(return_value=[])
    hot.delete = AsyncMock()
    hot.clear = AsyncMock(return_value=0)
    return hot


@pytest.fixture
def mock_warm() -> MagicMock:
    """Create a mock WarmMemoryStore."""
    warm = MagicMock()
    warm.store = AsyncMock(return_value="skill-uuid-1234")
    warm.retrieve = AsyncMock(return_value=[])
    warm.update_fitness = AsyncMock()
    return warm


@pytest.fixture
def mock_cold() -> MagicMock:
    """Create a mock ColdMemoryStore."""
    cold = MagicMock()
    cold.store = AsyncMock(return_value="cold-uuid-5678")
    cold.search_by_tags = AsyncMock(return_value=[])
    cold.search_by_query = AsyncMock(return_value=[])
    cold.consolidate = AsyncMock(return_value=0)
    return cold


class _MockedManager(MemoryManager):
    """MemoryManager test double exposing injected tier mocks for assertions.

    The ``manager`` fixture attaches ``_mock_hot``/``_mock_warm``/``_mock_cold``
    to a real MemoryManager; this subclass declares them so pyright accepts the
    access without adding test-only attributes to the production class.
    """

    _mock_hot: MagicMock
    _mock_warm: MagicMock
    _mock_cold: MagicMock


@pytest.fixture
def manager(
    mock_redis: MagicMock,
    mock_db_session: MagicMock,
    mock_settings: MagicMock,
    mock_hot: MagicMock,
    mock_warm: MagicMock,
    mock_cold: MagicMock,
) -> _MockedManager:
    """Create a MemoryManager with all tier mocks injected."""
    with (
        patch("src.memory.manager.HotMemoryStore", return_value=mock_hot),
        patch("src.memory.manager.WarmMemoryStore", return_value=mock_warm),
        patch("src.memory.manager.ColdMemoryStore", return_value=mock_cold),
    ):
        mgr = cast(_MockedManager, MemoryManager(
            redis_client=mock_redis,
            db_session=mock_db_session,
            settings=mock_settings,
        ))
    # Attach mocks for assertion access
    mgr._mock_hot = mock_hot
    mgr._mock_warm = mock_warm
    mgr._mock_cold = mock_cold
    return mgr


class TestMemoryManagerInit:
    """Tests for MemoryManager.__init__."""

    def test_initializes_hot_tier(
        self,
        mock_redis: MagicMock,
        mock_db_session: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Hot tier should be initialized with redis_client and TTL from settings."""
        with patch("src.memory.manager.HotMemoryStore") as hot_cls:
            MemoryManager(
                redis_client=mock_redis,
                db_session=mock_db_session,
                settings=mock_settings,
            )
            hot_cls.assert_called_once_with(
                redis_client=mock_redis,
                ttl_seconds=mock_settings.redis.cache_ttl_seconds,
            )

    def test_initializes_warm_tier(
        self,
        mock_redis: MagicMock,
        mock_db_session: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Warm tier is initialized with db_session and an embedding generator."""
        with patch("src.memory.manager.WarmMemoryStore") as warm_cls:
            MemoryManager(
                redis_client=mock_redis,
                db_session=mock_db_session,
                settings=mock_settings,
            )
            warm_cls.assert_called_once()
            kwargs = warm_cls.call_args.kwargs
            assert kwargs["session"] is mock_db_session
            assert "generator" in kwargs  # EmbeddingGenerator wired (§10.2)

    def test_initializes_cold_tier(
        self,
        mock_redis: MagicMock,
        mock_db_session: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Cold tier is initialized with db_session and a generator."""
        with patch("src.memory.manager.ColdMemoryStore") as cold_cls:
            MemoryManager(
                redis_client=mock_redis,
                db_session=mock_db_session,
                settings=mock_settings,
            )
            cold_cls.assert_called_once()
            kwargs = cold_cls.call_args.kwargs
            assert kwargs["session"] is mock_db_session
            assert "generator" in kwargs  # EmbeddingGenerator wired (§10.2)

    def test_stores_settings_reference(
        self,
        mock_redis: MagicMock,
        mock_db_session: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Settings should be stored internally."""
        with (
            patch("src.memory.manager.HotMemoryStore"),
            patch("src.memory.manager.WarmMemoryStore"),
            patch("src.memory.manager.ColdMemoryStore"),
        ):
            mgr = MemoryManager(
                redis_client=mock_redis,
                db_session=mock_db_session,
                settings=mock_settings,
            )
            assert mgr._settings is mock_settings


class TestStoreObservation:
    """Tests for MemoryManager.store_observation()."""

    @pytest.mark.asyncio
    async def test_stores_to_hot_tier(self, manager: _MockedManager) -> None:
        """store_observation should call hot.set with observation data."""
        await manager.store_observation(
            content="agent performed task",
            importance=0.7,
            tags=["execution", "success"],
            episode_type="execution",
        )

        manager._mock_hot.add_observation.assert_called_once()
        call = manager._mock_hot.add_observation.call_args
        # Key starts with "obs:execution:"
        assert call[1]["key"].startswith("obs:execution:")
        assert call[1]["value"]["content"] == "agent performed task"
        assert call[1]["value"]["tags"] == ["execution", "success"]
        assert call[1]["ttl"] == 3600

    @pytest.mark.asyncio
    async def test_stores_to_cold_tier(self, manager: _MockedManager) -> None:
        """store_observation should call cold.store with episode data."""
        await manager.store_observation(
            content="agent reflected",
            importance=0.8,
            tags=["reflection"],
            episode_type="reflection",
        )

        manager._mock_cold.store.assert_called_once_with(
            episode_type="reflection",
            content="agent reflected",
            importance=0.8,
            context_tags=["reflection"],
        )

    @pytest.mark.asyncio
    async def test_default_importance(self, manager: _MockedManager) -> None:
        """Default importance should be 0.5."""
        await manager.store_observation(content="default test")

        call = manager._mock_cold.store.call_args
        assert call[1]["importance"] == 0.5

    @pytest.mark.asyncio
    async def test_default_tags_empty_list(self, manager: _MockedManager) -> None:
        """When tags is None, hot tier should receive empty list."""
        await manager.store_observation(content="no tags")

        hot_call = manager._mock_hot.add_observation.call_args
        assert hot_call[1]["value"]["tags"] == []

        cold_call = manager._mock_cold.store.call_args
        assert cold_call[1]["context_tags"] is None

    @pytest.mark.asyncio
    async def test_default_episode_type(self, manager: _MockedManager) -> None:
        """Default episode_type should be 'execution'."""
        await manager.store_observation(content="default type")

        hot_call = manager._mock_hot.add_observation.call_args
        assert hot_call[1]["key"].startswith("obs:execution:")

        cold_call = manager._mock_cold.store.call_args
        assert cold_call[1]["episode_type"] == "execution"


class TestStoreSkill:
    """Tests for MemoryManager.store_skill()."""

    @pytest.mark.asyncio
    async def test_stores_to_warm_tier(self, manager: _MockedManager) -> None:
        """store_skill should call warm.store with correct parameters."""
        await manager.store_skill(
            name="code_review",
            content="Review code for quality",
            skill_type="procedure",
            tags=["quality", "review"],
        )

        manager._mock_warm.store.assert_called_once_with(
            memory_type="procedure",
            name="code_review",
            content="Review code for quality",
            tags=["quality", "review"],
            fitness_score=0.5,
        )

    @pytest.mark.asyncio
    async def test_returns_uuid(self, manager: _MockedManager) -> None:
        """store_skill should return the UUID from warm.store."""
        result = await manager.store_skill(
            name="test_skill",
            content="test content",
        )

        assert result == "skill-uuid-1234"

    @pytest.mark.asyncio
    async def test_default_skill_type(self, manager: _MockedManager) -> None:
        """Default skill_type should be 'procedure'."""
        await manager.store_skill(name="s1", content="c1")

        call = manager._mock_warm.store.call_args
        assert call[1]["memory_type"] == "procedure"

    @pytest.mark.asyncio
    async def test_default_tags_none(self, manager: _MockedManager) -> None:
        """When tags is None, warm.store should receive None."""
        await manager.store_skill(name="s1", content="c1")

        call = manager._mock_warm.store.call_args
        assert call[1]["tags"] is None


class TestRetrieveContext:
    """Tests for MemoryManager.retrieve_context()."""

    @pytest.mark.asyncio
    async def test_aggregates_from_all_tiers_with_tags(
        self, manager: _MockedManager
    ) -> None:
        """retrieve_context should aggregate results from hot, warm, and cold tiers."""
        manager._mock_hot.search_recent = AsyncMock(return_value=[
            {"content": "recent obs"},
        ])
        manager._mock_warm.retrieve = AsyncMock(return_value=[
            {"id": "w1", "name": "skill1"},
        ])
        manager._mock_cold.search_by_tags = AsyncMock(return_value=[
            {"id": "c1", "content": "cold memory"},
        ])

        results = await manager.retrieve_context(
            query="test query",
            tags=["execution"],
            limit=10,
        )

        # Should have results from all 3 tiers
        tiers = {r["tier"] for r in results}
        assert "hot" in tiers
        assert "warm" in tiers
        assert "cold" in tiers

    @pytest.mark.asyncio
    async def test_respects_limit(self, manager: _MockedManager) -> None:
        """retrieve_context should truncate results to the specified limit."""
        manager._mock_hot.search_recent = AsyncMock(return_value=[
            {"content": f"hot-{i}"} for i in range(5)
        ])
        manager._mock_warm.retrieve = AsyncMock(return_value=[
            {"id": f"w{i}"} for i in range(5)
        ])
        # No tags so cold not queried

        results = await manager.retrieve_context(
            query="test",
            tags=None,
            limit=3,
        )

        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_no_tags_runs_semantic_but_skips_tag_search(
        self, manager: _MockedManager
    ) -> None:
        """With no tags, tag-based cold search is skipped but semantic search runs.

        The query now drives semantic cold recall even without tags; only the
        explicit tag filter remains tag-gated.
        """
        manager._mock_hot.search_recent = AsyncMock(return_value=[])
        manager._mock_warm.retrieve = AsyncMock(return_value=[])

        await manager.retrieve_context(query="semantic query", tags=None, limit=5)

        manager._mock_cold.search_by_tags.assert_not_called()
        manager._mock_cold.search_by_query.assert_called_once_with(
            query="semantic query", limit=3, min_similarity=0.0
        )

    @pytest.mark.asyncio
    async def test_semantic_cold_results_flow_through(
        self, manager: _MockedManager
    ) -> None:
        """Semantic cold recall results appear as cold-tier items with similarity."""
        manager._mock_hot.search_recent = AsyncMock(return_value=[])
        manager._mock_warm.retrieve = AsyncMock(return_value=[])
        manager._mock_cold.search_by_query = AsyncMock(return_value=[
            {"id": "c1", "content": "semantically relevant", "similarity": 0.9},
        ])

        results = await manager.retrieve_context(query="q", tags=None, limit=5)

        cold = [r for r in results if r["tier"] == "cold"]
        assert len(cold) == 1
        assert cold[0]["id"] == "c1"
        assert cold[0]["similarity"] == 0.9

    @pytest.mark.asyncio
    async def test_semantic_and_tag_cold_results_deduped(
        self, manager: _MockedManager
    ) -> None:
        """A memory returned by both semantic and tag search appears once."""
        manager._mock_hot.search_recent = AsyncMock(return_value=[])
        manager._mock_warm.retrieve = AsyncMock(return_value=[])
        shared = {"id": "c1", "content": "both paths match"}
        manager._mock_cold.search_by_query = AsyncMock(return_value=[shared])
        manager._mock_cold.search_by_tags = AsyncMock(return_value=[shared])

        results = await manager.retrieve_context(query="q", tags=["t"], limit=10)

        cold_ids = [r["id"] for r in results if r["tier"] == "cold"]
        assert cold_ids == ["c1"]  # deduped, not duplicated

    @pytest.mark.asyncio
    async def test_hot_recall_uses_obs_prefix_and_configured_size(
        self, manager: _MockedManager
    ) -> None:
        """A4: hot recall goes through search_recent with the 'obs:' prefix and
        the configured recall size (not the legacy unordered search('obs:*')[:2])."""
        manager._mock_hot.search_recent = AsyncMock(return_value=[])
        manager._mock_warm.retrieve = AsyncMock(return_value=[])

        await manager.retrieve_context(query="test", tags=None, limit=5)

        manager._mock_hot.search_recent.assert_called_once_with("obs:", 3)
        manager._mock_hot.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_warm_retrieve_with_tags_and_min_fitness(
        self, manager: _MockedManager
    ) -> None:
        """Warm tier retrieve should pass tags and min_fitness=0.3."""
        manager._mock_hot.search_recent = AsyncMock(return_value=[])
        manager._mock_warm.retrieve = AsyncMock(return_value=[])
        manager._mock_cold.search_by_tags = AsyncMock(return_value=[])

        await manager.retrieve_context(
            query="test",
            tags=["python"],
            limit=5,
        )

        manager._mock_warm.retrieve.assert_called_once_with(
            tags=["python"],
            min_fitness=0.3,
            limit=3,
        )

    @pytest.mark.asyncio
    async def test_cold_search_with_tags(self, manager: _MockedManager) -> None:
        """Cold tier should search by tags when tags are provided."""
        manager._mock_hot.search_recent = AsyncMock(return_value=[])
        manager._mock_warm.retrieve = AsyncMock(return_value=[])
        manager._mock_cold.search_by_tags = AsyncMock(return_value=[])

        await manager.retrieve_context(
            query="test",
            tags=["agent", "tool"],
            limit=10,
        )

        manager._mock_cold.search_by_tags.assert_called_once_with(
            tags=["agent", "tool"],
            limit=3,
        )

    @pytest.mark.asyncio
    async def test_results_include_tier_field(self, manager: _MockedManager) -> None:
        """Each result dict should include a 'tier' field."""
        manager._mock_hot.search_recent = AsyncMock(return_value=[
            {"content": "hot item"},
        ])
        manager._mock_warm.retrieve = AsyncMock(return_value=[
            {"id": "w1", "name": "skill"},
        ])

        results = await manager.retrieve_context(query="test", limit=5)

        for r in results:
            assert "tier" in r
            assert r["tier"] in ("hot", "warm", "cold")


class TestUpdateSkillFitness:
    """Tests for MemoryManager.update_skill_fitness()."""

    @pytest.mark.asyncio
    async def test_delegates_to_warm(self, manager: _MockedManager) -> None:
        """update_skill_fitness should call warm.update_fitness."""
        await manager.update_skill_fitness("skill-id-123", success=True)

        manager._mock_warm.update_fitness.assert_called_once_with(
            "skill-id-123", True
        )

    @pytest.mark.asyncio
    async def test_passes_failure(self, manager: _MockedManager) -> None:
        """update_skill_fitness should pass success=False correctly."""
        await manager.update_skill_fitness("skill-id-456", success=False)

        manager._mock_warm.update_fitness.assert_called_once_with(
            "skill-id-456", False
        )


class TestConsolidate:
    """Tests for MemoryManager.consolidate()."""

    @pytest.mark.asyncio
    async def test_returns_consolidation_stats(self, manager: _MockedManager) -> None:
        """consolidate should return stats with cold_deleted count."""
        manager._mock_cold.consolidate = AsyncMock(return_value=5)

        stats = await manager.consolidate()

        assert stats == {"cold_deleted": 5}

    @pytest.mark.asyncio
    async def test_calls_cold_consolidate_with_defaults(
        self, manager: _MockedManager
    ) -> None:
        """consolidate should call cold.consolidate with max_age_days=90 and min_importance=0.1."""
        manager._mock_cold.consolidate = AsyncMock(return_value=0)

        await manager.consolidate()

        manager._mock_cold.consolidate.assert_called_once_with(
            max_age_days=90,
            min_importance=0.1,
        )

    @pytest.mark.asyncio
    async def test_zero_deleted(self, manager: _MockedManager) -> None:
        """When no memories are old enough, cold_deleted should be 0."""
        manager._mock_cold.consolidate = AsyncMock(return_value=0)

        stats = await manager.consolidate()

        assert stats["cold_deleted"] == 0
