"""Redis-backed prompt cache for LLM responses."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import redis.asyncio as aioredis
from loguru import logger

from src.config.settings import Settings
from src.llm.models import LLMResponse
from src.observability.metrics import record_cache_lookup


class PromptCache:
    """Caches LLM responses in Redis with configurable TTL.

    Cache keys are deterministic hashes of (messages, model, temperature, max_tokens).
    """

    def __init__(self, redis_client: aioredis.Redis, settings: Settings) -> None:
        self._redis = redis_client
        self._ttl = settings.redis.cache_ttl_seconds
        self._prefix = "turing:llm_cache:"
        # In-memory hit/miss counters (fast stats without a Redis round-trip).
        # The authoritative time-series lives in the Prometheus counters fed by
        # record_cache_lookup; these mirror them for a cheap stats() read.
        self._hits = 0
        self._misses = 0

    def _make_cache_key(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a deterministic cache key from request parameters."""
        payload = json.dumps(
            {"messages": messages, "model": model, "temperature": temperature, "max_tokens": max_tokens},
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
        return f"{self._prefix}{model}:{digest}"

    async def get(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int | None = None,
    ) -> LLMResponse | None:
        """Look up a cached response.

        Returns:
            The cached LLMResponse if found, None otherwise.
        """
        key = self._make_cache_key(messages, model, temperature, max_tokens)
        try:
            raw = await self._redis.get(key)
            if raw is None:
                self._misses += 1
                record_cache_lookup(model, hit=False)
                return None
            data = json.loads(raw)
            self._hits += 1
            record_cache_lookup(model, hit=True)
            return LLMResponse(cached=True, **data)
        except Exception as exc:
            logger.warning(f"Cache lookup failed for {key}: {exc}")
            return None

    async def stats(self) -> dict[str, Any]:
        """Return cache hit/miss counters and an estimated live entry count.

        ``hits``/``misses`` are in-memory counters maintained on every ``get``;
        ``hit_rate`` is ``hits / (hits + misses)`` (0.0 when no lookups yet).
        ``size_est`` is a Redis ``SCAN`` over this cache's key prefix, so it
        reflects the live cache size and degrades to 0 if Redis is unreachable.
        """
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": (self._hits / total) if total else 0.0,
            "size_est": await self._estimate_size(),
        }

    async def _estimate_size(self) -> int:
        """Count live cache entries via SCAN over the key prefix (best-effort)."""
        try:
            count = 0
            async for _ in self._redis.scan_iter(match=f"{self._prefix}*"):
                count += 1
            return count
        except Exception as exc:
            logger.warning(f"Cache size estimate failed: {exc}")
            return 0

    async def set(
        self,
        response: LLMResponse,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int | None = None,
    ) -> None:
        """Store an LLM response in the cache."""
        key = self._make_cache_key(messages, model, temperature, max_tokens)
        try:
            data = {
                "content": response.content,
                "model": response.model,
                "provider": response.provider,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "total_tokens": response.total_tokens,
                "cost_usd": response.cost_usd,
                "finish_reason": response.finish_reason,
            }
            await self._redis.set(key, json.dumps(data), ex=self._ttl)
        except Exception as exc:
            logger.warning(f"Cache write failed for {key}: {exc}")

    async def invalidate(self, pattern: str = "*") -> None:
        """Invalidate cache entries matching a pattern."""
        try:
            keys = []
            async for key in self._redis.scan_iter(match=f"{self._prefix}{pattern}"):
                keys.append(key)
            if keys:
                await self._redis.delete(*keys)
                logger.debug(f"Invalidated {len(keys)} cache entries matching '{pattern}'")
        except Exception as exc:
            logger.warning(f"Cache invalidation failed: {exc}")
