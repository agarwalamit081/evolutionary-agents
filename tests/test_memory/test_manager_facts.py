"""MemoryManager fact-tier wrappers (Phase 5): store/retrieve delegation +
extract_and_store_facts best-effort contract.

The manager is built with MagicMock tier deps (EmbeddingGenerator tolerates
them) and ``manager.warm`` is swapped for a MagicMock/AsyncMock so the
delegation and the never-raises / per-fact-resilience guarantees are asserted
without a DB.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.memory.facts import FactCandidate
from src.memory.manager import MemoryManager


def _manager() -> tuple[MemoryManager, MagicMock]:
    """A MemoryManager with cheap MagicMock deps and a mocked warm tier.

    Returns ``(manager, warm_mock)`` so tests assert against the warm tier's
    AsyncMock attributes directly (pyright otherwise types ``manager.warm`` as
    the production ``WarmMemoryStore``).
    """
    settings = MagicMock()
    settings.redis.cache_ttl_seconds = 3600
    settings.llm.embedding_dim = 768
    mgr = MemoryManager(
        redis_client=MagicMock(),  # type: ignore[arg-type]
        db_session=MagicMock(),  # type: ignore[arg-type]
        settings=settings,  # type: ignore[arg-type]
    )
    warm = MagicMock()
    warm.store_fact = AsyncMock(return_value="fact-uuid")
    warm.retrieve_facts = AsyncMock(
        return_value=[{"key": "k", "value": "v", "confidence": 0.5}]
    )
    warm.update_fitness = AsyncMock()
    mgr.warm = warm  # type: ignore[assignment]
    return mgr, warm


class TestStoreRetrieveFactDelegation:
    @pytest.mark.asyncio
    async def test_store_fact_delegates_to_warm(self) -> None:
        mgr, warm = _manager()
        uid = await mgr.store_fact(
            "row_count", "1024 rows", source="fold_1", confidence=0.8, tags=["data"]
        )
        assert uid == "fact-uuid"
        warm.store_fact.assert_awaited_once()
        kwargs = warm.store_fact.call_args.kwargs
        assert kwargs["key"] == "row_count"
        assert kwargs["value"] == "1024 rows"
        assert kwargs["source"] == "fold_1"
        assert kwargs["confidence"] == pytest.approx(0.8)
        assert kwargs["tags"] == ["data"]

    @pytest.mark.asyncio
    async def test_retrieve_facts_delegates_to_warm(self) -> None:
        mgr, warm = _manager()
        facts = await mgr.retrieve_facts(query="how many rows", limit=4)
        assert facts == [{"key": "k", "value": "v", "confidence": 0.5}]
        warm.retrieve_facts.assert_awaited_once()
        assert warm.retrieve_facts.call_args.kwargs["query"] == "how many rows"
        assert warm.retrieve_facts.call_args.kwargs["limit"] == 4


class TestExtractAndStoreFacts:
    @pytest.mark.asyncio
    async def test_extracts_and_persists_each_fact(self) -> None:
        mgr, warm = _manager()
        candidates = [
            FactCandidate(key="a", value="va", confidence=0.8),
            FactCandidate(key="b", value="vb", confidence=0.6),
        ]
        gateway = MagicMock()
        with patch(
            "src.memory.facts.extract_facts",
            new=AsyncMock(return_value=candidates),
        ):
            stored = await mgr.extract_and_store_facts(
                gateway, "summary", source="fold_1_episode", max_facts=5
            )
        assert stored == 2
        assert warm.store_fact.await_count == 2
        # Provenance source is threaded through to each store.
        for call in warm.store_fact.await_args_list:
            assert call.kwargs["source"] == "fold_1_episode"

    @pytest.mark.asyncio
    async def test_extraction_failure_returns_zero_and_never_raises(self) -> None:
        mgr, warm = _manager()
        with patch(
            "src.memory.facts.extract_facts",
            new=AsyncMock(side_effect=RuntimeError("gateway down")),
        ):
            stored: int = await mgr.extract_and_store_facts(MagicMock(), "summary")
        assert stored == 0
        warm.store_fact.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_one_bad_store_does_not_abort_the_rest(self) -> None:
        mgr, warm = _manager()
        # 2nd store raises; the 1st and 3rd must still succeed.
        warm.store_fact = AsyncMock(
            side_effect=["ok-1", RuntimeError("write failed"), "ok-3"]
        )
        candidates = [
            FactCandidate(key="a", value="va"),
            FactCandidate(key="b", value="vb"),
            FactCandidate(key="c", value="vc"),
        ]
        with patch(
            "src.memory.facts.extract_facts",
            new=AsyncMock(return_value=candidates),
        ):
            stored = await mgr.extract_and_store_facts(MagicMock(), "summary")
        assert stored == 2  # only the failed one is not counted
