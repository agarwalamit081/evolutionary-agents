"""Redis-backed prompt cache for LLM responses."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import redis.asyncio as aioredis
from loguru import logger

from turing_agent.config.settings import Settings
from turing_agent.llm.models import LLMResponse


class PromptCache:
    """Caches LLM responses in Redis with configurable TTL.

    Cache keys are deterministic hashes of (messages, model, temperature, max_tokens).
    """

    def __init__(self, redis_client: aioredis.Redis, settings: Settings) -> None:
        self._redis = redis_client
        self._ttl = settings.redis.cache_ttl_seconds
        self._prefix = "turing:llm_cache:"

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
                return None
            data = json.loads(raw)
            return LLMResponse(cached=True, **data)
        except Exception as exc:
            logger.warning(f"Cache lookup failed for {key}: {exc}")
            return None

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
