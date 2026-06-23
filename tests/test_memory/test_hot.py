"""Tests for src.memory.hot — HotMemory Redis ephemeral cache."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.memory.hot import HotMemory


@pytest.fixture
def redis_client() -> MagicMock:
    """Create a mock Redis client for unit testing."""
    return MagicMock()


@pytest.fixture
def hot_memory(redis_client: MagicMock) -> HotMemory:
    """Create a HotMemory instance with default TTL."""
    return HotMemory(redis_client=redis_client, ttl_seconds=3600)


class TestHotMemoryInit:
    """Tests for HotMemory.__init__."""

    def test_default_prefix(self, redis_client: MagicMock) -> None:
        """Prefix should be 'turing:hot:'."""
        mem = HotMemory(redis_client=redis_client, ttl_seconds=3600)
        assert mem._prefix == "turing:hot:"

    def test_default_ttl(self, redis_client: MagicMock) -> None:
        """Default TTL should be 3600 seconds."""
        mem = HotMemory(redis_client=redis_client, ttl_seconds=3600)
        assert mem._ttl == 3600

    def test_custom_ttl(self, redis_client: MagicMock) -> None:
        """Custom TTL should be stored."""
        mem = HotMemory(redis_client=redis_client, ttl_seconds=7200)
        assert mem._ttl == 7200

    def test_redis_client_stored(self, redis_client: MagicMock) -> None:
        """Redis client reference should be stored."""
        mem = HotMemory(redis_client=redis_client, ttl_seconds=3600)
        assert mem._redis is redis_client


class TestHotMemoryGet:
    """Tests for HotMemory.get()."""

    @pytest.mark.asyncio
    async def test_get_existing_key_returns_parsed_json(
        self, hot_memory: HotMemory, redis_client: MagicMock
    ) -> None:
        """When key exists, return the parsed JSON dict."""
        stored = json.dumps({"content": "hello", "tags": ["a"]})
        redis_client.get = AsyncMock(return_value=stored.encode())

        result = await hot_memory.get("test_key")
        assert result == {"content": "hello", "tags": ["a"]}
        redis_client.get.assert_called_once_with("turing:hot:test_key")

    @pytest.mark.asyncio
    async def test_get_existing_key_string_response(
        self, hot_memory: HotMemory, redis_client: MagicMock
    ) -> None:
        """When Redis returns a string (not bytes), still parse it."""
        stored = json.dumps({"content": "world"})
        redis_client.get = AsyncMock(return_value=stored)

        result = await hot_memory.get("str_key")
        assert result == {"content": "world"}

    @pytest.mark.asyncio
    async def test_get_missing_key_returns_none(
        self, hot_memory: HotMemory, redis_client: MagicMock
    ) -> None:
        """When key is missing, return None."""
        redis_client.get = AsyncMock(return_value=None)

        result = await hot_memory.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_json_parse_error_returns_none(
        self, hot_memory: HotMemory, redis_client: MagicMock
    ) -> None:
        """When stored value is invalid JSON, return None (caught by except)."""
        redis_client.get = AsyncMock(return_value=b"not-valid-json{{{")

        result = await hot_memory.get("bad_json")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_redis_error_returns_none(
        self, hot_memory: HotMemory, redis_client: MagicMock
    ) -> None:
        """When Redis raises an exception, return None gracefully."""
        redis_client.get = AsyncMock(side_effect=ConnectionError("Redis down"))

        result = await hot_memory.get("error_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_uses_prefixed_key(
        self, hot_memory: HotMemory, redis_client: MagicMock
    ) -> None:
        """get() must prepend the 'turing:hot:' prefix to the key."""
        redis_client.get = AsyncMock(return_value=None)
        await hot_memory.get("mykey")
        redis_client.get.assert_called_once_with("turing:hot:mykey")


class TestHotMemorySet:
    """Tests for HotMemory.set()."""

    @pytest.mark.asyncio
    async def test_set_stores_with_ttl(
        self, hot_memory: HotMemory, redis_client: MagicMock
    ) -> None:
        """set() should store the JSON-serialized value with default TTL."""
        redis_client.set = AsyncMock(return_value=True)
        value = {"content": "test", "count": 42}

        await hot_memory.set("key1", value)

        redis_client.set.assert_called_once()
        call_args = redis_client.set.call_args
        assert call_args[0][0] == "turing:hot:key1"
        assert json.loads(call_args[0][1]) == value
        assert call_args[1]["ex"] == 3600

    @pytest.mark.asyncio
    async def test_set_custom_ttl_override(
        self, hot_memory: HotMemory, redis_client: MagicMock
    ) -> None:
        """set() with ttl override should use the provided TTL."""
        redis_client.set = AsyncMock(return_value=True)

        await hot_memory.set("key2", {"data": "val"}, ttl=1800)

        call_args = redis_client.set.call_args
        assert call_args[1]["ex"] == 1800

    @pytest.mark.asyncio
    async def test_set_json_serialization_with_default(
        self, hot_memory: HotMemory, redis_client: MagicMock
    ) -> None:
        """set() should serialize values with json.dumps(default=str) for non-serializable types."""
        redis_client.set = AsyncMock(return_value=True)
        from datetime import datetime

        value = {"ts": datetime(2025, 1, 1, 12, 0, 0), "name": "test"}

        await hot_memory.set("key3", value)

        call_args = redis_client.set.call_args
        stored = call_args[0][1]
        parsed = json.loads(stored)
        assert parsed["name"] == "test"
        # datetime should be serialized to string via default=str
        assert isinstance(parsed["ts"], str)

    @pytest.mark.asyncio
    async def test_set_uses_prefixed_key(
        self, hot_memory: HotMemory, redis_client: MagicMock
    ) -> None:
        """set() must prepend the 'turing:hot:' prefix to the key."""
        redis_client.set = AsyncMock(return_value=True)

        await hot_memory.set("abc", {"v": 1})

        call_args = redis_client.set.call_args
        assert call_args[0][0] == "turing:hot:abc"

    @pytest.mark.asyncio
    async def test_set_redis_error_does_not_raise(
        self, hot_memory: HotMemory, redis_client: MagicMock
    ) -> None:
        """set() should swallow Redis errors without raising."""
        redis_client.set = AsyncMock(side_effect=ConnectionError("Redis down"))

        # Should not raise
        await hot_memory.set("error_key", {"data": "val"})


class TestHotMemoryDelete:
    """Tests for HotMemory.delete()."""

    @pytest.mark.asyncio
    async def test_delete_calls_redis_delete(
        self, hot_memory: HotMemory, redis_client: MagicMock
    ) -> None:
        """delete() should call redis.delete with the prefixed key."""
        redis_client.delete = AsyncMock(return_value=1)

        await hot_memory.delete("mykey")

        redis_client.delete.assert_called_once_with("turing:hot:mykey")

    @pytest.mark.asyncio
    async def test_delete_redis_error_does_not_raise(
        self, hot_memory: HotMemory, redis_client: MagicMock
    ) -> None:
        """delete() should swallow Redis errors without raising."""
        redis_client.delete = AsyncMock(side_effect=ConnectionError("Redis down"))

        await hot_memory.delete("error_key")
        # Should not raise


class TestHotMemorySearch:
    """Tests for HotMemory.search()."""

    @pytest.mark.asyncio
    async def test_search_returns_matching_results(
        self, hot_memory: HotMemory, redis_client: MagicMock
    ) -> None:
        """search() should return parsed JSON for each matching key."""
        keys = [b"turing:hot:obs:exec:1", b"turing:hot:obs:exec:2"]

        async def mock_scan_iter(match: str = "") -> Any:
            for k in keys:
                yield k

        redis_client.scan_iter = mock_scan_iter
        redis_client.get = AsyncMock(
            side_effect=[
                json.dumps({"content": "first"}).encode(),
                json.dumps({"content": "second"}).encode(),
            ]
        )

        results = await hot_memory.search("obs:*")

        assert len(results) == 2
        assert results[0] == {"content": "first"}
        assert results[1] == {"content": "second"}

    @pytest.mark.asyncio
    async def test_search_empty_results(
        self, hot_memory: HotMemory, redis_client: MagicMock
    ) -> None:
        """search() with no matches returns an empty list."""

        async def mock_scan_iter(match: str = "") -> Any:
            return
            yield  # Make this an async generator

        redis_client.scan_iter = mock_scan_iter

        results = await hot_memory.search("nonexistent:*")

        assert results == []

    @pytest.mark.asyncio
    async def test_search_skips_none_values(
        self, hot_memory: HotMemory, redis_client: MagicMock
    ) -> None:
        """search() should skip keys where redis.get returns None."""

        async def mock_scan_iter(match: str = "") -> Any:
            yield b"turing:hot:obs:1"

        redis_client.scan_iter = mock_scan_iter
        redis_client.get = AsyncMock(return_value=None)

        results = await hot_memory.search("obs:*")

        assert results == []

    @pytest.mark.asyncio
    async def test_search_redis_error_returns_empty(
        self, hot_memory: HotMemory, redis_client: MagicMock
    ) -> None:
        """search() should return empty list on Redis error."""

        def broken_scan_iter(match: str = "") -> Any:
            raise ConnectionError("Redis down")

        redis_client.scan_iter = broken_scan_iter

        results = await hot_memory.search("obs:*")

        assert results == []

    @pytest.mark.asyncio
    async def test_search_uses_prefix_in_pattern(
        self, hot_memory: HotMemory, redis_client: MagicMock
    ) -> None:
        """search() should prepend prefix to the scan pattern."""

        captured_match: list[str] = []

        async def mock_scan_iter(match: str = "") -> Any:
            captured_match.append(match)
            return
            yield

        redis_client.scan_iter = mock_scan_iter

        await hot_memory.search("obs:*")

        assert captured_match == ["turing:hot:obs:*"]


class TestHotMemoryClear:
    """Tests for HotMemory.clear()."""

    @pytest.mark.asyncio
    async def test_clear_deletes_all_keys(
        self, hot_memory: HotMemory, redis_client: MagicMock
    ) -> None:
        """clear() should delete all keys with the prefix and return count."""
        keys = [b"turing:hot:a", b"turing:hot:b", b"turing:hot:c"]

        async def mock_scan_iter(match: str = "") -> Any:
            for k in keys:
                yield k

        redis_client.scan_iter = mock_scan_iter
        redis_client.delete = AsyncMock(return_value=3)

        count = await hot_memory.clear()

        assert count == 3
        redis_client.delete.assert_called_once_with(*keys)

    @pytest.mark.asyncio
    async def test_clear_no_keys_returns_zero(
        self, hot_memory: HotMemory, redis_client: MagicMock
    ) -> None:
        """clear() with no keys should return 0 without calling delete."""

        async def mock_scan_iter(match: str = "") -> Any:
            return
            yield

        redis_client.scan_iter = mock_scan_iter

        count = await hot_memory.clear()

        assert count == 0
        redis_client.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_clear_redis_error_returns_zero(
        self, hot_memory: HotMemory, redis_client: MagicMock
    ) -> None:
        """clear() should return 0 on Redis error."""

        def broken_scan_iter(match: str = "") -> Any:
            raise ConnectionError("Redis down")

        redis_client.scan_iter = broken_scan_iter

        count = await hot_memory.clear()

        assert count == 0


@pytest.fixture
def fake_async_redis() -> Any:
    """A fresh real in-memory async Redis per test (fakeredis 2.36).

    A MagicMock cannot prove ZSET ordering — add_observation/search_recent are
    exercised against a real sorted-set so the recency contract is genuine.
    """
    import fakeredis

    return fakeredis.FakeAsyncRedis()


class TestHotMemoryRecency:
    """A4: recency-ranked recall via the observation ZSET (newest-first).

    Replaces the legacy unordered hot.search('obs:*')[:2]. The ZSET is scored
    by time (add_observation) and search_recent returns highest-score members
    first (ZREVRANGE), skipping any whose STRING has TTL-expired.
    """

    @pytest.mark.asyncio
    async def test_search_recent_returns_by_recency(self, fake_async_redis: Any) -> None:
        """Four observations → search_recent returns the newest N, descending."""
        mem = HotMemory(redis_client=fake_async_redis, ttl_seconds=3600)
        # Explicit increasing scores → deterministic order, independent of
        # sub-microsecond time.time() collisions in a tight insertion loop.
        await mem.add_observation("obs:exec:1", {"content": "oldest"}, score=1.0)
        await mem.add_observation("obs:exec:2", {"content": "old"}, score=2.0)
        await mem.add_observation("obs:exec:3", {"content": "new"}, score=3.0)
        await mem.add_observation("obs:exec:4", {"content": "newest"}, score=4.0)

        results = await mem.search_recent("obs:", limit=2)

        assert [r["content"] for r in results] == ["newest", "new"]

    @pytest.mark.asyncio
    async def test_add_observation_also_writes_string(self, fake_async_redis: Any) -> None:
        """add_observation writes the keyed STRING too (back-compat with set())."""
        mem = HotMemory(redis_client=fake_async_redis, ttl_seconds=3600)
        await mem.add_observation("obs:exec:9", {"content": "x"}, score=1.0)

        val = await mem.get("obs:exec:9")
        assert val == {"content": "x"}

    @pytest.mark.asyncio
    async def test_search_recent_skips_expired_string(self, fake_async_redis: Any) -> None:
        """A ZSET member whose STRING has TTL-expired is skipped, not returned
        as None/garbage — search_recent yields the next live member instead."""
        mem = HotMemory(redis_client=fake_async_redis, ttl_seconds=3600)
        await mem.add_observation("obs:exec:1", {"content": "stale"}, score=1.0)
        await mem.add_observation("obs:exec:2", {"content": "live"}, score=2.0)
        # Simulate the STRING expiring while the ZSET member lingers.
        await fake_async_redis.delete("turing:hot:obs:exec:1")

        results = await mem.search_recent("obs:", limit=2)

        assert results == [{"content": "live"}]

    @pytest.mark.asyncio
    async def test_search_recent_respects_limit(self, fake_async_redis: Any) -> None:
        """search_recent caps at `limit` even when more members exist."""
        mem = HotMemory(redis_client=fake_async_redis, ttl_seconds=3600)
        for i in range(5):
            await mem.add_observation(f"obs:exec:{i}", {"content": str(i)}, score=float(i))

        results = await mem.search_recent("obs:", limit=3)

        assert [r["content"] for r in results] == ["4", "3", "2"]

