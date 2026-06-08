"""Hot memory — Redis ephemeral cache for recent context."""

from __future__ import annotations

from typing import Any

import redis.asyncio as aioredis
from loguru import logger


class HotMemory:
    """Redis-backed ephemeral memory for recent task context.

    Stores short-lived data (recent messages, intermediate results)
    with configurable TTL. Data is lost on Redis restart.
    """

    def __init__(self, redis_client: aioredis.Redis, ttl_seconds: int = 3600) -> None:
        self._redis = redis_client
        self._ttl = ttl_seconds
        self._prefix = "turing:hot:"

    async def get(self, key: str) -> dict[str, Any] | None:
        """Retrieve a value from hot memory.

        Args:
            key: Memory key (without prefix).

        Returns:
            Dict value if found, None otherwise.
        """
        try:
            import json
            raw = await self._redis.get(f"{self._prefix}{key}")
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            logger.warning(f"Hot memory get failed for {key}: {exc}")
            return None

    async def set(self, key: str, value: dict[str, Any], ttl: int | None = None) -> None:
        """Store a value in hot memory.

        Args:
            key: Memory key.
            value: Dict value to store.
            ttl: Optional TTL override in seconds.
        """
        try:
            import json
            effective_ttl = ttl or self._ttl
            await self._redis.set(
                f"{self._prefix}{key}",
                json.dumps(value, default=str),
                ex=effective_ttl,
            )
        except Exception as exc:
            logger.warning(f"Hot memory set failed for {key}: {exc}")

    async def delete(self, key: str) -> None:
        """Delete a key from hot memory."""
        try:
            await self._redis.delete(f"{self._prefix}{key}")
        except Exception as exc:
            logger.warning(f"Hot memory delete failed for {key}: {exc}")

    async def search(self, pattern: str = "*") -> list[dict[str, Any]]:
        """Find entries matching a key pattern.

        Args:
            pattern: Glob pattern for key matching.

        Returns:
            List of matching values.
        """
        import json
        results: list[dict[str, Any]] = []
        try:
            async for key in self._redis.scan_iter(match=f"{self._prefix}{pattern}"):
                raw = await self._redis.get(key)
                if raw:
                    results.append(json.loads(raw))
        except Exception as exc:
            logger.warning(f"Hot memory search failed: {exc}")
        return results

    async def clear(self) -> int:
        """Clear all hot memory entries.

        Returns:
            Number of keys deleted.
        """
        count = 0
        try:
            keys = []
            async for key in self._redis.scan_iter(match=f"{self._prefix}*"):
                keys.append(key)
            if keys:
                count = await self._redis.delete(*keys)
        except Exception as exc:
            logger.warning(f"Hot memory clear failed: {exc}")
        return count
