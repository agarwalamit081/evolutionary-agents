"""Redis-backed result cache for idempotent tools (DeepAgent pattern).

Caches successful results of opt-in, read-only tools (``web_search``,
``file_reader``) keyed by ``sha1(tool_name + canonical args)`` so repeated
calls — within a run and across runs — skip the underlying network/disk work.

Best-effort by design: every Redis interaction is wrapped so that a cache
failure degrades to a transparent miss and never breaks a tool call. Only
successful results are cached; errors are never stored.

The client is lazily constructed from ``redis_url`` the first time it is
needed, so instantiating the cache (e.g. in ``build_task_graph``) never opens
a connection and is safe in tests without Redis.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import redis.asyncio as aioredis
from loguru import logger

if TYPE_CHECKING:
    from src.config.settings import Settings


class ToolResultCache:
    """Best-effort Redis cache for idempotent tool results.

    Args:
        redis_url: Redis connection URL. Used to lazily build a pooled client.
        ttl_seconds: Per-entry TTL.
        enabled: Master switch; when False, get/set are no-ops.
        redis_client: Optional pre-built client (e.g. a shared pool or a fake
            in tests). When provided it takes precedence over ``redis_url``.
    """

    _PREFIX = "turing:toolcache:"

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        ttl_seconds: int = 3600,
        enabled: bool = True,
        redis_client: aioredis.Redis | None = None,
    ) -> None:
        self._redis_url = redis_url
        self._ttl = ttl_seconds
        self._enabled = enabled
        # Sentinel: None = not yet built. ``_connect`` flips this to a client
        # or disables itself if construction fails.
        self._client: aioredis.Redis | None = redis_client
        self._owns_client = redis_client is None

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        redis_client: aioredis.Redis | None = None,
    ) -> ToolResultCache:
        """Build a cache from ``Settings`` (Redis URL + ToolCacheSettings)."""
        tc = getattr(settings, "tool_cache", None)
        return cls(
            redis_url=settings.redis.redis_url,
            ttl_seconds=tc.tool_cache_ttl_seconds if tc else 3600,
            enabled=bool(tc.tool_cache_enabled) if tc else True,
            redis_client=redis_client,
        )

    def _connect(self) -> aioredis.Redis | None:
        """Return the client, lazily building it. Returns None when disabled."""
        if not self._enabled:
            return None
        if self._client is None:
            try:
                self._client = aioredis.from_url(self._redis_url)
            except Exception as exc:  # noqa: BLE001 — never break a run
                logger.debug(f"ToolResultCache client init failed: {exc}")
                self._enabled = False
                return None
        return self._client

    @staticmethod
    def make_key(tool_name: str, args: dict[str, Any]) -> str:
        """Deterministic cache key from tool name + canonical args."""
        raw = f"{tool_name}|{json.dumps(args, sort_keys=True, default=str)}"
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
        return f"{ToolResultCache._PREFIX}{digest}"

    async def get(
        self,
        tool_name: str,
        args: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Look up a cached result. Returns the stored dict or None on miss."""
        client = self._connect()
        if client is None:
            return None
        try:
            raw = await client.get(self.make_key(tool_name, args))
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:  # noqa: BLE001 — transparent miss on failure
            logger.debug(f"ToolResultCache get failed ({tool_name}): {exc}")
            return None

    async def set(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """Store a successful result. No-op on failure."""
        client = self._connect()
        if client is None:
            return
        try:
            await client.set(
                self.make_key(tool_name, args),
                json.dumps(result, default=str),
                ex=self._ttl,
            )
        except Exception as exc:  # noqa: BLE001 — never break a run
            logger.debug(f"ToolResultCache set failed ({tool_name}): {exc}")

    async def aclose(self) -> None:
        """Close the client if this cache owns it."""
        if self._owns_client and self._client is not None:
            try:
                await self._client.aclose()
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"ToolResultCache close failed: {exc}")
            finally:
                self._client = None
