"""Tests for src.tools.result_cache and the execute-node cache routing.

Covers:
- ``ToolResultCache``: deterministic keys, hit/miss, TTL propagation, the
  disabled no-op path, transparent-miss on Redis failure, ``aclose`` ownership,
  and ``from_settings`` wiring.
- ``_execute_tool_call`` routing: cacheable flag honored, errors never cached,
  and mutating (non-cacheable) tools never touch the cache.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.graph.nodes.execute import _execute_tool_call
from src.tools.registry import ToolRegistry
from src.tools.result_cache import ToolResultCache


# ─── Helpers ─────────────────────────────────────────────────────────


def _settings(redis_url: str, *, enabled: bool, ttl: int) -> SimpleNamespace:
    """Build a minimal settings-like object for ``from_settings``."""
    return SimpleNamespace(
        redis=SimpleNamespace(redis_url=redis_url),
        tool_cache=SimpleNamespace(tool_cache_enabled=enabled, tool_cache_ttl_seconds=ttl),
    )


def _tc(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Build an OpenAI-style tool_call dict (function.name + arguments)."""
    return {"function": {"name": name, "arguments": json.dumps(args)}}


async def _read_handler(query: str) -> str:
    """A minimal idempotent handler for cacheable-tool tests."""
    return f"results for {query}"


async def _write_handler(**kwargs: Any) -> str:
    """A minimal mutating handler for non-cacheable-tool tests.

    Accepts arbitrary kwargs (``file_path`` etc.) so the cache-routing logic —
    not the handler signature — is what's under test.
    """
    return f"wrote {len(kwargs)} field(s)"


# ─── ToolResultCache.make_key ────────────────────────────────────────


class TestMakeKey:
    """Cache keys must be deterministic and canonical."""

    def test_prefixed_sha1(self) -> None:
        """Key is namespaced under the project prefix and is a hex digest."""
        key = ToolResultCache.make_key("web_search", {"query": "x"})
        assert key.startswith("turing:toolcache:")
        digest = key.removeprefix("turing:toolcache:")
        assert len(digest) == 40  # sha1 hex digest length
        assert all(c in "0123456789abcdef" for c in digest)

    def test_deterministic(self) -> None:
        """Same (name, args) → same key."""
        a = ToolResultCache.make_key("web_search", {"query": "rust async"})
        b = ToolResultCache.make_key("web_search", {"query": "rust async"})
        assert a == b

    def test_canonical_arg_ordering(self) -> None:
        """Dict key order must not change the key (sort_keys=True)."""
        a = ToolResultCache.make_key("web_search", {"query": "x", "max_results": 5})
        b = ToolResultCache.make_key("web_search", {"max_results": 5, "query": "x"})
        assert a == b

    def test_different_tools_different_keys(self) -> None:
        """Different tool name → different key."""
        assert ToolResultCache.make_key("web_search", {"q": "x"}) != ToolResultCache.make_key(
            "file_reader", {"q": "x"}
        )

    def test_different_args_different_keys(self) -> None:
        """Different arg value → different key."""
        assert ToolResultCache.make_key("web_search", {"query": "a"}) != ToolResultCache.make_key(
            "web_search", {"query": "b"}
        )


# ─── ToolResultCache.get / set ───────────────────────────────────────


class TestGetSet:
    """Hit/miss/TTL behavior with an injected mock client."""

    @pytest.mark.asyncio
    async def test_get_hit_returns_parsed(self) -> None:
        """A stored JSON value is parsed back into a dict on hit."""
        client = MagicMock()
        client.get = AsyncMock(return_value=json.dumps({"success": True, "output": "ok"}).encode())
        cache = ToolResultCache(redis_client=client)  # type: ignore[arg-type]

        result = await cache.get("web_search", {"query": "x"})
        assert result == {"success": True, "output": "ok"}
        client.get.assert_awaited_once_with(ToolResultCache.make_key("web_search", {"query": "x"}))

    @pytest.mark.asyncio
    async def test_get_miss_returns_none(self) -> None:
        """A Redis None (key absent) yields a None miss."""
        client = MagicMock()
        client.get = AsyncMock(return_value=None)
        cache = ToolResultCache(redis_client=client)  # type: ignore[arg-type]

        assert await cache.get("web_search", {"query": "x"}) is None

    @pytest.mark.asyncio
    async def test_set_writes_with_ttl(self) -> None:
        """set persists JSON and forwards the configured TTL via ``ex=``."""
        client = MagicMock()
        client.set = AsyncMock(return_value=True)
        cache = ToolResultCache(redis_client=client, ttl_seconds=7200)  # type: ignore[arg-type]

        await cache.set("web_search", {"query": "x"}, {"success": True, "output": "ok"})
        client.set.assert_awaited_once()
        args, kwargs = client.set.call_args
        assert args[0] == ToolResultCache.make_key("web_search", {"query": "x"})
        assert json.loads(args[1]) == {"success": True, "output": "ok"}
        assert kwargs["ex"] == 7200  # TTL propagated

    @pytest.mark.asyncio
    async def test_roundtrip_get_after_set(self) -> None:
        """A value written via set is retrievable via get through a shared store."""
        store: dict[str, bytes] = {}

        async def _store_set(key: str, value: str, ex: int = 0) -> bool:
            store[key] = value.encode() if isinstance(value, str) else value
            return True

        client = MagicMock()
        client.get = AsyncMock(side_effect=lambda k: store.get(k))
        client.set = _store_set
        cache = ToolResultCache(redis_client=client)  # type: ignore[arg-type]

        assert await cache.get("web_search", {"query": "y"}) is None
        await cache.set("web_search", {"query": "y"}, {"success": True, "output": "hit"})
        assert await cache.get("web_search", {"query": "y"}) == {"success": True, "output": "hit"}


# ─── Disabled + failure-degradation ──────────────────────────────────


class TestDisabledAndFailure:
    """The cache must never break a run: disabled = no-op, failure = miss."""

    @pytest.mark.asyncio
    async def test_disabled_get_is_noop(self) -> None:
        """When disabled, get short-circuits to None without touching Redis."""
        client = MagicMock()
        client.get = AsyncMock()
        cache = ToolResultCache(redis_client=client, enabled=False)  # type: ignore[arg-type]

        assert await cache.get("web_search", {"query": "x"}) is None
        client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_disabled_set_is_noop(self) -> None:
        """When disabled, set is a no-op."""
        client = MagicMock()
        client.set = AsyncMock()
        cache = ToolResultCache(redis_client=client, enabled=False)  # type: ignore[arg-type]

        await cache.set("web_search", {"query": "x"}, {"success": True})
        client.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_failure_is_transparent_miss(self) -> None:
        """A Redis error on get returns None, never raises."""
        client = MagicMock()
        client.get = AsyncMock(side_effect=RuntimeError("redis down"))
        cache = ToolResultCache(redis_client=client)  # type: ignore[arg-type]

        assert await cache.get("web_search", {"query": "x"}) is None

    @pytest.mark.asyncio
    async def test_set_failure_never_raises(self) -> None:
        """A Redis error on set is swallowed, never propagated."""
        client = MagicMock()
        client.set = AsyncMock(side_effect=RuntimeError("redis down"))
        cache = ToolResultCache(redis_client=client)  # type: ignore[arg-type]

        await cache.set("web_search", {"query": "x"}, {"success": True})  # must not raise


# ─── aclose ownership ────────────────────────────────────────────────


class TestAclose:
    """aclose closes clients the cache owns, leaves injected ones alone."""

    @pytest.mark.asyncio
    async def test_closes_owned_client(self) -> None:
        """A client built lazily by the cache is closed on aclose."""
        cache = ToolResultCache(redis_url="redis://localhost:6379/0")
        injected = MagicMock()
        injected.aclose = AsyncMock()
        cache._client = injected  # simulate an internally-built client
        cache._owns_client = True

        await cache.aclose()
        injected.aclose.assert_awaited_once()
        assert cache._client is None

    @pytest.mark.asyncio
    async def test_does_not_close_injected_client(self) -> None:
        """A client passed in from outside is not closed (caller owns it)."""
        client = MagicMock()
        client.aclose = AsyncMock()
        cache = ToolResultCache(redis_client=client)  # type: ignore[arg-type]

        await cache.aclose()
        client.aclose.assert_not_called()
        # Injected client remains intact for the caller.
        assert cache._client is client


# ─── from_settings ───────────────────────────────────────────────────


class TestFromSettings:
    """from_settings reads the Redis URL and ToolCacheSettings."""

    def test_reads_redis_url_and_defaults(self) -> None:
        """Redis URL + tool_cache settings are mapped onto the constructor."""
        settings = _settings("redis://cache:6379/1", enabled=True, ttl=1800)
        cache = ToolResultCache.from_settings(settings)  # type: ignore[arg-type]

        assert cache._redis_url == "redis://cache:6379/1"
        assert cache._ttl == 1800
        assert cache._enabled is True

    def test_disabled_flag_propagates(self) -> None:
        """tool_cache_enabled=False disables the cache."""
        settings = _settings("redis://cache:6379/0", enabled=False, ttl=3600)
        cache = ToolResultCache.from_settings(settings)  # type: ignore[arg-type]

        assert cache._enabled is False


# ─── Execute-node cache routing (_execute_tool_call) ─────────────────


class TestExecuteToolCallRouting:
    """The cacheable flag, error-suppression, and mutating-tool rules in execute."""

    @pytest.fixture
    def registry(self) -> ToolRegistry:
        """A registry with one cacheable and one non-cacheable tool."""
        reg = ToolRegistry()
        reg.register(
            name="web_search",
            handler=_read_handler,
            description="cacheable",
            cacheable=True,
        )
        reg.register(
            name="file_writer",
            handler=_write_handler,
            description="mutating, never cached",
            cacheable=False,
        )
        return reg

    @pytest.mark.asyncio
    async def test_cache_hit_skips_handler(
        self, registry: ToolRegistry
    ) -> None:
        """A cacheable tool hit returns the cached result without invoking the handler."""
        handler_spy = MagicMock(wraps=_read_handler)
        registry.register(
            name="web_search",
            handler=handler_spy,
            description="cacheable",
            cacheable=True,
        )
        cache = MagicMock()
        cache.get = AsyncMock(return_value={"success": True, "output": "cached!"})
        cache.set = AsyncMock()

        result = await _execute_tool_call(_tc("web_search", {"query": "x"}), registry, cache)  # type: ignore[arg-type]

        assert result.success is True
        assert result.output == "cached!"
        assert result.metadata.get("cached") is True
        handler_spy.assert_not_called()
        cache.set.assert_not_called()  # hit → no write

    @pytest.mark.asyncio
    async def test_miss_caches_success(self, registry: ToolRegistry) -> None:
        """A cacheable tool miss invokes the handler and caches the successful result."""
        cache = MagicMock()
        cache.get = AsyncMock(return_value=None)
        cache.set = AsyncMock()

        result = await _execute_tool_call(_tc("web_search", {"query": "x"}), registry, cache)  # type: ignore[arg-type]

        assert result.success is True
        assert result.output == "results for x"
        cache.get.assert_awaited_once()
        cache.set.assert_awaited_once()
        # The cached payload mirrors the ToolResult (success + output).
        payload = cache.set.call_args.args[2]
        assert payload["success"] is True
        assert payload["output"] == "results for x"

    @pytest.mark.asyncio
    async def test_handler_error_never_cached(
        self, registry: ToolRegistry
    ) -> None:
        """A failing handler returns an error ToolResult and is NOT cached."""

        async def _boom(query: str) -> str:
            raise RuntimeError(f"upstream 500 for {query}")

        registry.register(
            name="web_search",
            handler=_boom,
            description="cacheable but failing",
            cacheable=True,
        )
        cache = MagicMock()
        cache.get = AsyncMock(return_value=None)
        cache.set = AsyncMock()

        result = await _execute_tool_call(_tc("web_search", {"query": "x"}), registry, cache)  # type: ignore[arg-type]

        assert result.success is False
        assert "upstream 500" in (result.error or "")
        cache.set.assert_not_called()  # errors are never cached

    @pytest.mark.asyncio
    async def test_mutating_tool_never_cached(
        self, registry: ToolRegistry
    ) -> None:
        """A non-cacheable (mutating) tool bypasses the cache entirely."""
        cache = MagicMock()
        cache.get = AsyncMock()
        cache.set = AsyncMock()

        result = await _execute_tool_call(
            _tc("file_writer", {"file_path": "a.txt"}), registry, cache  # type: ignore[arg-type]
        )

        assert result.success is True
        cache.get.assert_not_called()
        cache.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalid_args_returns_error_without_cache(
        self, registry: ToolRegistry
    ) -> None:
        """Malformed tool-call JSON yields a clean error and skips the cache."""
        cache = MagicMock()
        cache.get = AsyncMock()
        cache.set = AsyncMock()
        bad_tc = {"function": {"name": "web_search", "arguments": "{not json"}}

        result = await _execute_tool_call(bad_tc, registry, cache)  # type: ignore[arg-type]

        assert result.success is False
        assert "Invalid arguments" in (result.error or "")
        cache.get.assert_not_called()
        cache.set.assert_not_called()
