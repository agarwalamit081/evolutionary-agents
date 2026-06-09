"""Tests for src.llm.cache — Redis-backed prompt cache."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config.settings import Settings
from src.llm.cache import PromptCache
from src.llm.models import LLMResponse


# ─── Helpers ─────────────────────────────────────────────────────────


def _make_settings() -> Settings:
    """Create Settings with default Redis config (no .env required)."""
    return Settings()


def _make_redis() -> MagicMock:
    """Create a mock Redis client with dict-backed get/set."""
    store: dict[str, str] = {}
    redis_mock = MagicMock()
    redis_mock.get = AsyncMock(side_effect=lambda k: store.get(k))
    redis_mock.set = AsyncMock(side_effect=lambda k, v, **kw: store.__setitem__(k, v))
    redis_mock.delete = AsyncMock(side_effect=lambda *keys: [store.pop(k, None) for k in keys])
    redis_mock.scan_iter = MagicMock()
    return redis_mock


def _make_response(**overrides: Any) -> LLMResponse:
    """Create a sample LLMResponse for cache testing."""
    defaults = {
        "content": "Paris is the capital of France.",
        "model": "gpt-4o-mini-2024-07-18",
        "provider": "openai",
        "input_tokens": 20,
        "output_tokens": 10,
        "total_tokens": 30,
        "cost_usd": 0.0001,
        "finish_reason": "stop",
    }
    defaults.update(overrides)
    return LLMResponse(**defaults)


# ─── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def settings() -> Settings:
    return _make_settings()


@pytest.fixture
def mock_redis() -> MagicMock:
    return _make_redis()


@pytest.fixture
def cache(mock_redis: MagicMock, settings: Settings) -> PromptCache:
    return PromptCache(mock_redis, settings)


@pytest.fixture
def sample_messages() -> list[dict[str, Any]]:
    return [{"role": "user", "content": "What is the capital of France?"}]


# ─── Test _make_cache_key ────────────────────────────────────────────


class TestMakeCacheKey:
    """Tests for PromptCache._make_cache_key determinism and uniqueness."""

    def test_same_inputs_produce_same_key(
        self, cache: PromptCache, sample_messages: list[dict[str, Any]]
    ) -> None:
        """Identical inputs always produce the same cache key."""
        key1 = cache._make_cache_key(sample_messages, "gpt-4o-mini-2024-07-18", 0.5, 100)
        key2 = cache._make_cache_key(sample_messages, "gpt-4o-mini-2024-07-18", 0.5, 100)
        assert key1 == key2

    def test_different_messages_produce_different_keys(self, cache: PromptCache) -> None:
        """Different message content produces different keys."""
        msgs_a = [{"role": "user", "content": "Hello"}]
        msgs_b = [{"role": "user", "content": "Goodbye"}]
        key_a = cache._make_cache_key(msgs_a, "gpt-4o-mini-2024-07-18", 0.5)
        key_b = cache._make_cache_key(msgs_b, "gpt-4o-mini-2024-07-18", 0.5)
        assert key_a != key_b

    def test_different_models_produce_different_keys(
        self, cache: PromptCache, sample_messages: list[dict[str, Any]]
    ) -> None:
        """Different model IDs produce different keys."""
        key_a = cache._make_cache_key(sample_messages, "gpt-4o-mini-2024-07-18", 0.5)
        key_b = cache._make_cache_key(sample_messages, "claude-sonnet-4-6", 0.5)
        assert key_a != key_b

    def test_different_temperature_produces_different_keys(
        self, cache: PromptCache, sample_messages: list[dict[str, Any]]
    ) -> None:
        """Different temperatures produce different keys."""
        key_a = cache._make_cache_key(sample_messages, "gpt-4o-mini-2024-07-18", 0.3)
        key_b = cache._make_cache_key(sample_messages, "gpt-4o-mini-2024-07-18", 0.9)
        assert key_a != key_b

    def test_different_max_tokens_produces_different_keys(
        self, cache: PromptCache, sample_messages: list[dict[str, Any]]
    ) -> None:
        """Different max_tokens produce different keys."""
        key_a = cache._make_cache_key(sample_messages, "gpt-4o-mini-2024-07-18", 0.5, 100)
        key_b = cache._make_cache_key(sample_messages, "gpt-4o-mini-2024-07-18", 0.5, 200)
        assert key_a != key_b

    def test_key_starts_with_prefix(
        self, cache: PromptCache, sample_messages: list[dict[str, Any]]
    ) -> None:
        """Cache key begins with the configured prefix."""
        key = cache._make_cache_key(sample_messages, "gpt-4o-mini-2024-07-18", 0.5)
        assert key.startswith("turing:llm_cache:")

    def test_key_contains_model_name(
        self, cache: PromptCache, sample_messages: list[dict[str, Any]]
    ) -> None:
        """Cache key includes the model name for easier debugging."""
        key = cache._make_cache_key(sample_messages, "gpt-4o-mini-2024-07-18", 0.5)
        assert "gpt-4o-mini-2024-07-18" in key


# ─── Test get ────────────────────────────────────────────────────────


class TestCacheGet:
    """Tests for PromptCache.get (cache hit, miss, error)."""

    @pytest.mark.asyncio
    async def test_cache_miss_returns_none(
        self, cache: PromptCache, sample_messages: list[dict[str, Any]]
    ) -> None:
        """When the key is not in Redis, get returns None."""
        result = await cache.get(sample_messages, "gpt-4o-mini-2024-07-18", 0.5)
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_hit_returns_llm_response(
        self, cache: PromptCache, mock_redis: MagicMock, sample_messages: list[dict[str, Any]]
    ) -> None:
        """When the key is in Redis, get returns an LLMResponse with cached=True."""
        response = _make_response()
        key = cache._make_cache_key(sample_messages, "gpt-4o-mini-2024-07-18", 0.5)

        # Pre-populate the mock store via the set mock's side effect
        response_data = {
            "content": response.content,
            "model": response.model,
            "provider": response.provider,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_tokens": response.total_tokens,
            "cost_usd": response.cost_usd,
            "finish_reason": response.finish_reason,
        }
        # Inject into the dict-backed store
        store: dict[str, str] = {}
        store[key] = json.dumps(response_data)
        mock_redis.get = AsyncMock(side_effect=lambda k: store.get(k))

        result = await cache.get(sample_messages, "gpt-4o-mini-2024-07-18", 0.5)

        assert result is not None
        assert isinstance(result, LLMResponse)
        assert result.cached is True
        assert result.content == "Paris is the capital of France."
        assert result.model == "gpt-4o-mini-2024-07-18"

    @pytest.mark.asyncio
    async def test_cache_hit_with_max_tokens(
        self, cache: PromptCache, mock_redis: MagicMock, sample_messages: list[dict[str, Any]]
    ) -> None:
        """Cache hit works when max_tokens is provided."""
        response = _make_response()
        key = cache._make_cache_key(sample_messages, "gpt-4o-mini-2024-07-18", 0.5, 256)

        response_data = {
            "content": response.content,
            "model": response.model,
            "provider": response.provider,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_tokens": response.total_tokens,
            "cost_usd": response.cost_usd,
            "finish_reason": response.finish_reason,
        }
        store: dict[str, str] = {key: json.dumps(response_data)}
        mock_redis.get = AsyncMock(side_effect=lambda k: store.get(k))

        result = await cache.get(sample_messages, "gpt-4o-mini-2024-07-18", 0.5, 256)
        assert result is not None
        assert result.cached is True

    @pytest.mark.asyncio
    async def test_parse_error_returns_none(
        self, cache: PromptCache, mock_redis: MagicMock, sample_messages: list[dict[str, Any]]
    ) -> None:
        """When cached data is corrupt JSON, get returns None (not an exception)."""
        key = cache._make_cache_key(sample_messages, "gpt-4o-mini-2024-07-18", 0.5)
        store: dict[str, str] = {key: "not valid json {{{"}
        mock_redis.get = AsyncMock(side_effect=lambda k: store.get(k))

        result = await cache.get(sample_messages, "gpt-4o-mini-2024-07-18", 0.5)
        assert result is None

    @pytest.mark.asyncio
    async def test_redis_error_returns_none(
        self, cache: PromptCache, mock_redis: MagicMock, sample_messages: list[dict[str, Any]]
    ) -> None:
        """When Redis raises an exception, get returns None gracefully."""
        mock_redis.get = AsyncMock(side_effect=ConnectionError("Redis down"))

        result = await cache.get(sample_messages, "gpt-4o-mini-2024-07-18", 0.5)
        assert result is None


# ─── Test set ────────────────────────────────────────────────────────


class TestCacheSet:
    """Tests for PromptCache.set (store with TTL)."""

    @pytest.mark.asyncio
    async def test_set_calls_redis_set_with_ttl(
        self, cache: PromptCache, mock_redis: MagicMock, sample_messages: list[dict[str, Any]]
    ) -> None:
        """set stores the response in Redis with the configured TTL."""
        response = _make_response()

        await cache.set(response, sample_messages, "gpt-4o-mini-2024-07-18", 0.5)

        mock_redis.set.assert_awaited_once()
        call_args = mock_redis.set.call_args
        key = call_args.args[0]
        value_json = call_args.args[1]
        ex = call_args.kwargs.get("ex")

        # Verify the key is well-formed
        assert key.startswith("turing:llm_cache:")

        # Verify the stored JSON contains expected fields
        data = json.loads(value_json)
        assert data["content"] == "Paris is the capital of France."
        assert data["model"] == "gpt-4o-mini-2024-07-18"
        assert data["provider"] == "openai"
        assert data["input_tokens"] == 20

        # Verify TTL is set from settings
        assert ex == _make_settings().redis.cache_ttl_seconds

    @pytest.mark.asyncio
    async def test_set_with_max_tokens(
        self, cache: PromptCache, mock_redis: MagicMock, sample_messages: list[dict[str, Any]]
    ) -> None:
        """set works correctly when max_tokens is provided."""
        response = _make_response()

        await cache.set(response, sample_messages, "gpt-4o-mini-2024-07-18", 0.5, 512)

        mock_redis.set.assert_awaited_once()
        key = mock_redis.set.call_args.args[0]
        # Key should incorporate max_tokens=512
        assert "gpt-4o-mini-2024-07-18" in key

    @pytest.mark.asyncio
    async def test_set_handles_redis_error_gracefully(
        self, cache: PromptCache, mock_redis: MagicMock, sample_messages: list[dict[str, Any]]
    ) -> None:
        """set does not raise when Redis fails; it logs and returns."""
        response = _make_response()
        mock_redis.set = AsyncMock(side_effect=ConnectionError("Redis unavailable"))

        # Should not raise
        await cache.set(response, sample_messages, "gpt-4o-mini-2024-07-18", 0.5)

    @pytest.mark.asyncio
    async def test_set_does_not_store_cached_flag(
        self, cache: PromptCache, mock_redis: MagicMock, sample_messages: list[dict[str, Any]]
    ) -> None:
        """The stored JSON should not include the 'cached' flag (set=True on retrieval)."""
        response = _make_response()

        await cache.set(response, sample_messages, "gpt-4o-mini-2024-07-18", 0.5)

        value_json = mock_redis.set.call_args.args[1]
        data = json.loads(value_json)
        assert "cached" not in data


# ─── Test invalidate ─────────────────────────────────────────────────


class TestCacheInvalidate:
    """Tests for PromptCache.invalidate (pattern-based deletion)."""

    @pytest.mark.asyncio
    async def test_invalidate_deletes_matching_keys(
        self, cache: PromptCache, mock_redis: MagicMock
    ) -> None:
        """invalidate scans for matching keys and deletes them."""
        # scan_iter is used with 'async for', so return an async iterable
        async def _async_iter(items: list[str]):
            for item in items:
                yield item

        keys = [
            "turing:llm_cache:gpt-4o-mini-2024-07-18:abc123",
            "turing:llm_cache:gpt-4o-mini-2024-07-18:def456",
        ]
        mock_redis.scan_iter = MagicMock(return_value=_async_iter(keys))

        await cache.invalidate("gpt-4o-mini-2024-07-18*")

        mock_redis.delete.assert_awaited_once()
        # delete(*keys) — each key is a separate positional arg
        assert mock_redis.delete.call_args.args == tuple(keys)

    @pytest.mark.asyncio
    async def test_invalidate_no_matching_keys_skips_delete(
        self, cache: PromptCache, mock_redis: MagicMock
    ) -> None:
        """When no keys match, delete is not called."""
        async def _empty_async_iter():
            return
            yield  # noqa: unreachable — makes this an async generator

        mock_redis.scan_iter = MagicMock(return_value=_empty_async_iter())

        await cache.invalidate("nonexistent-model*")

        mock_redis.delete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_invalidate_default_pattern_matches_all(
        self, cache: PromptCache, mock_redis: MagicMock
    ) -> None:
        """Default pattern '*' scans all cache entries."""
        async def _async_iter(items: list[str]):
            for item in items:
                yield item

        keys = [
            "turing:llm_cache:model-a:abc",
            "turing:llm_cache:model-b:def",
            "turing:llm_cache:model-c:ghi",
        ]
        mock_redis.scan_iter = MagicMock(return_value=_async_iter(keys))

        await cache.invalidate()

        # scan_iter should have been called with the full prefix + '*'
        mock_redis.scan_iter.assert_called_once_with(match="turing:llm_cache:*")
        mock_redis.delete.assert_awaited_once()
        # delete(*keys) — each key is a separate positional arg
        assert mock_redis.delete.call_args.args == tuple(keys)

    @pytest.mark.asyncio
    async def test_invalidate_handles_redis_error_gracefully(
        self, cache: PromptCache, mock_redis: MagicMock
    ) -> None:
        """invalidate does not raise when Redis fails; it logs and returns."""
        mock_redis.scan_iter = MagicMock(side_effect=ConnectionError("Redis down"))

        # Should not raise
        await cache.invalidate("*")
