"""Hot memory — Redis ephemeral cache for recent context."""

from __future__ import annotations

import time
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
        # Recency-ranked index of observation keys. ``search`` (STRING scan)
        # has no ordering guarantee; this ZSET lets retrieve_context recall the
        # most recent observations newest-first (ZREVRANGE by time-score).
        self._obs_zset_key = f"{self._prefix}obs"

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

    async def add_observation(
        self,
        key: str,
        value: dict[str, Any],
        ttl: int | None = None,
        score: float | None = None,
    ) -> None:
        """Store an observation in the STRING store AND index it for recency recall.

        Writes the keyed STRING (same as ``set`` — immediate keyed access) and
        adds the key to a recency-ranked ZSET so ``search_recent`` can return
        the newest observations first. The legacy ``search`` had no ordering,
        so ``retrieve_context`` had no deterministic "most recent" notion.

        Args:
            key: Observation key (e.g. "obs:execution:1234"), WITHOUT prefix.
            value: Dict value to store in the STRING.
            ttl: TTL seconds for the STRING; the ZSET TTL is refreshed to match
                so the whole index ages out 1h after the last observation.
            score: Optional explicit recency score (epoch seconds). Defaults to
                ``time.time()``; tests pass an explicit value for determinism
                (``time.time()`` can collide for sub-microsecond insertions).
        """
        # STRING (back-compat with set()) — immediate keyed access.
        await self.set(key=key, value=value, ttl=ttl)
        # ZSET index for newest-first recall. Best-effort: a Redis hiccup here
        # only degrades to the unordered search() path; the STRING is already
        # written, so the observation is never lost.
        try:
            member = f"{self._prefix}{key}"
            recency = score if score is not None else time.time()
            effective_ttl = ttl or self._ttl
            await self._redis.zadd(self._obs_zset_key, {member: recency})
            await self._redis.expire(self._obs_zset_key, effective_ttl)
        except Exception as exc:
            logger.warning(f"Hot memory observation index failed for {key}: {exc}")

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

    async def search_recent(
        self,
        member_prefix: str = "obs:",
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Recall the most recent observations, newest-first, via the recency ZSET.

        Replaces the unordered ``search("obs:*")`` recall. ``ZREVRANGE`` returns
        the highest-score (most recent) members first. Members whose STRING has
        already TTL-expired are skipped (the ZSET member can outlive its value);
        enough such members are fetched to still return up to ``limit`` live ones.

        Args:
            member_prefix: Only members whose key starts with this prefix are
                returned (the ZSET holds only ``obs:`` members today, but the
                filter keeps the contract explicit and forward-compatible).
            limit: Maximum number of live results to return.

        Returns:
            List of observation value dicts, newest-first.
        """
        import json

        results: list[dict[str, Any]] = []
        try:
            full_prefix = f"{self._prefix}{member_prefix}"
            # Over-fetch to tolerate stale (STRING-expired) ZSET members: fetch
            # up to limit+stale_window, then keep the first ``limit`` live ones.
            fetch_to = max(limit + 8, limit * 3) - 1
            members = await self._redis.zrevrange(self._obs_zset_key, 0, fetch_to)
            for raw_member in members:
                member = raw_member.decode() if isinstance(raw_member, bytes) else raw_member
                if not isinstance(member, str) or not member.startswith(full_prefix):
                    continue
                raw = await self._redis.get(member)
                if raw is None:
                    continue  # STRING expired; stale ZSET member — skip
                results.append(json.loads(raw))
                if len(results) >= limit:
                    break
        except Exception as exc:
            logger.warning(f"Hot memory search_recent failed: {exc}")
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
