"""Tests for prompt-cache hit/miss stats (Phase 3, B5 / M7a).

Covers ``PromptCache.stats`` and the ``LLMGateway.cache_stats`` accessor.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config.settings import Settings
from src.llm.cache import PromptCache
from src.llm.gateway import LLMGateway
from src.llm.models import LLMResponse


# ─── Helpers ─────────────────────────────────────────────────────────


def _make_settings() -> Settings:
    """Settings with default Redis config (no .env required)."""
    return Settings()


def _make_response() -> LLMResponse:
    """A sample LLMResponse for populating the cache."""
    return LLMResponse(
        content="Paris is the capital of France.",
        model="gpt-4o-mini-2024-07-18",
        provider="openai",
        input_tokens=20,
        output_tokens=10,
        total_tokens=30,
        cost_usd=0.0001,
        finish_reason="stop",
    )


def _async_iter(items: list[str]) -> Any:
    """Build an async generator (the shape redis.scan_iter returns)."""

    async def _gen() -> Any:
        for item in items:
            yield item

    return _gen()


def _make_redis(store: dict[str, str] | None = None) -> MagicMock:
    """A dict-backed mock redis with get/set and an empty scan_iter."""
    store = {} if store is None else store
    redis_mock = MagicMock()
    redis_mock.get = AsyncMock(side_effect=lambda k: store.get(k))

    def _store_set(k: str, v: str, **kw: Any) -> None:
        del kw  # cache.set passes ex=<ttl>; the mock store ignores it
        store.__setitem__(k, v)

    redis_mock.set = AsyncMock(side_effect=_store_set)
    redis_mock.scan_iter = MagicMock(return_value=_async_iter([]))
    return redis_mock


def _messages(content: str = "hi") -> list[dict[str, Any]]:
    return [{"role": "user", "content": content}]


# ─── PromptCache.stats ───────────────────────────────────────────────


class TestCacheStats:
    """Counters on PromptCache.get and the derived stats() snapshot."""

    @pytest.mark.asyncio
    async def test_miss_then_hit_increments_counters(self) -> None:
        """A miss increments _misses; a subsequent hit increments _hits."""
        cache = PromptCache(_make_redis({}), _make_settings())
        msgs = _messages()

        assert await cache.get(msgs, "gpt-4o-mini-2024-07-18", 0.5) is None
        await cache.set(_make_response(), msgs, "gpt-4o-mini-2024-07-18", 0.5)
        hit = await cache.get(msgs, "gpt-4o-mini-2024-07-18", 0.5)

        assert hit is not None
        assert cache._misses == 1
        assert cache._hits == 1

    @pytest.mark.asyncio
    async def test_hit_rate_reflects_ratio(self) -> None:
        """hit_rate is hits / (hits + misses)."""
        cache = PromptCache(_make_redis({}), _make_settings())
        msgs = _messages()

        await cache.get(msgs, "gpt-4o-mini-2024-07-18", 0.5)  # miss
        await cache.set(_make_response(), msgs, "gpt-4o-mini-2024-07-18", 0.5)
        await cache.get(msgs, "gpt-4o-mini-2024-07-18", 0.5)  # hit

        stats = await cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    @pytest.mark.asyncio
    async def test_size_est_counts_live_entries(self) -> None:
        """size_est is the SCAN count over the cache key prefix."""
        redis_mock = _make_redis({})
        redis_mock.scan_iter = MagicMock(
            return_value=_async_iter(
                [
                    "turing:llm_cache:gpt-4o-mini-2024-07-18:a",
                    "turing:llm_cache:gpt-4o-mini-2024-07-18:b",
                    "turing:llm_cache:claude-sonnet-4-6:c",
                ]
            )
        )
        cache = PromptCache(redis_mock, _make_settings())

        stats = await cache.stats()
        assert stats["size_est"] == 3
        redis_mock.scan_iter.assert_called_once_with(match="turing:llm_cache:*")

    @pytest.mark.asyncio
    async def test_stats_zero_when_no_lookups(self) -> None:
        """A fresh cache reports zeros and a 0.0 hit_rate."""
        cache = PromptCache(_make_redis({}), _make_settings())

        stats = await cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_get_exception_does_not_count(self) -> None:
        """A Redis error during get is neither a hit nor a miss."""
        redis_mock = _make_redis({})
        redis_mock.get = AsyncMock(side_effect=ConnectionError("Redis down"))
        cache = PromptCache(redis_mock, _make_settings())

        assert await cache.get(_messages(), "gpt-4o-mini-2024-07-18", 0.5) is None
        assert cache._hits == 0
        assert cache._misses == 0

    @pytest.mark.asyncio
    async def test_size_est_degrades_to_zero_on_scan_error(self) -> None:
        """A Redis error during SCAN yields size_est=0, not an exception."""
        redis_mock = _make_redis({})
        redis_mock.scan_iter = MagicMock(side_effect=ConnectionError("Redis down"))
        cache = PromptCache(redis_mock, _make_settings())

        stats = await cache.stats()
        assert stats["size_est"] == 0


# ─── LLMGateway.cache_stats ──────────────────────────────────────────


class TestGatewayCacheStats:
    """The gateway accessor guards for a missing cache and delegates otherwise."""

    @pytest.mark.asyncio
    async def test_disabled_cache_returns_zeros(self) -> None:
        """A gateway with no wired cache reports all-zero stats."""
        gateway = LLMGateway(_make_settings())

        stats = await gateway.cache_stats()
        assert stats == {"hits": 0, "misses": 0, "hit_rate": 0.0, "size_est": 0}

    @pytest.mark.asyncio
    async def test_delegates_to_wired_cache(self) -> None:
        """cache_stats returns the wired cache's live counters."""
        settings = _make_settings()
        cache = PromptCache(_make_redis({}), settings)
        gateway = LLMGateway(settings)
        gateway.set_cache(cache)

        msgs = _messages()
        await cache.set(_make_response(), msgs, "gpt-4o-mini-2024-07-18", 0.5)
        await cache.get(msgs, "gpt-4o-mini-2024-07-18", 0.5)  # hit
        await cache.get(msgs, "claude-sonnet-4-6", 0.5)  # miss (different key)

        stats = await gateway.cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


# ─── A1 (Phase 3.5): production gateway ↔ PromptCache wiring seam ────────────


class TestSetCacheSeam:
    """``gateway.set_cache`` is the seam ``execute_run`` wires on every run.

    The exact-match LLM cache is attached on the PRODUCTION path:
    ``execute_run`` builds a ``PromptCache`` over the run's Redis client and calls
    ``gateway.set_cache(cache)`` (runner.py:351); the worker's
    ``default_agent_executor`` reuses ``execute_run``, so the battery path has the
    cache attached. These lock the seam itself — ``_cache`` is ``None`` by default
    and the exact object is attached on ``set_cache`` — so the wiring is real, not
    dormant (the original "dormant on the worker path" misread this guards against).

    CAVEAT: the cache is exact-match on (messages, model, temperature, max_tokens)
    (cache._make_cache_key). In a verify loop the prompt grows every cycle, so
    cache hits there are ≈ 0 by design. The per-call input-token win is
    provider-native PREFIX caching (A2) observed via record_prompt_cache_tokens,
    not this exact-match cache.
    """

    def test_cache_is_none_by_default(self) -> None:
        gateway = LLMGateway(_make_settings())
        assert gateway._cache is None

    def test_set_cache_attaches_the_exact_object(self) -> None:
        gateway = LLMGateway(_make_settings())
        cache = PromptCache(_make_redis({}), _make_settings())
        gateway.set_cache(cache)
        assert gateway._cache is cache

    def test_set_cache_replaces_a_prior_cache(self) -> None:
        gateway = LLMGateway(_make_settings())
        first = PromptCache(_make_redis({}), _make_settings())
        second = PromptCache(_make_redis({}), _make_settings())
        gateway.set_cache(first)
        gateway.set_cache(second)
        assert gateway._cache is second
