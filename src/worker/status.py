"""Per-run status store backed by a Redis hash (Phase 2b).

The worker writes each run's lifecycle (queued → running → completed/failed) to
``turing:run:{run_id}``; the API reads it for ``GET /runs/{run_id}`` so a client
can poll without the worker holding an open HTTP connection. Hash TTL bounded by
``WorkerSettings.status_ttl_seconds`` so the store self-cleans (Redis rule: never
unbounded growth). All values are str (Redis hashes have no nested types).
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import TYPE_CHECKING, cast

from src.worker.schema import JobStatus, RunStatus, utc_now_iso

if TYPE_CHECKING:
    import redis.asyncio as aioredis

    from src.config.settings import WorkerSettings


class RunStatusStore:
    """Read/write ``RunStatus`` records in Redis hashes."""

    _KEY_PREFIX = "turing:run:"  # hash key = prefix + run_id

    def __init__(
        self,
        redis_client: aioredis.Redis,
        settings: WorkerSettings,
    ) -> None:
        self._redis = redis_client
        self._ttl = settings.status_ttl_seconds

    def _key(self, run_id: str) -> str:
        return f"{self._KEY_PREFIX}{run_id}"

    async def put(self, status: RunStatus) -> None:
        """Overwrite the status hash (resets TTL). Never raises — best-effort."""
        try:
            key = self._key(status.run_id)
            await self._redis.hset(key, mapping=status.to_hash())  # type: ignore[arg-type]
            if self._ttl > 0:
                await self._redis.expire(key, self._ttl)
        except Exception:
            # Status is observability — a Redis hiccup must never break a run.
            pass

    async def get(self, run_id: str) -> RunStatus | None:
        """Current status, or ``None`` when no hash exists (unknown/expired run)."""
        try:
            # redis-py 7.4 types async ``hgetall`` as ``Union[Awaitable[dict],
            # dict]`` (a sync/async-shared-signature artifact); the async client
            # always returns a coroutine, so cast away the non-awaitable branch.
            mapping = await cast(
                "Awaitable[dict[str | bytes, str | bytes]]",
                self._redis.hgetall(self._key(run_id)),
            )
        except Exception:
            return None
        if not mapping:
            return None
        try:
            return RunStatus.from_hash(mapping)  # type: ignore[arg-type]
        except Exception:
            return None

    async def mark(
        self,
        run_id: str,
        thread_id: str,
        status: JobStatus,
        **fields: object,
    ) -> RunStatus:
        """Convenience: load-or-create, patch fields + status, persist, return.

        ``started_at``/``finished_at`` are stamped automatically on the
        running/completed/failed transitions. Extra ``fields`` (final_output,
        is_complete, iteration_count, error) are merged.
        """
        current = await self.get(run_id)
        base = current or RunStatus(run_id=run_id, thread_id=thread_id)
        base.thread_id = thread_id
        base.status = status
        for k, v in fields.items():
            if v is not None and hasattr(base, k):
                setattr(base, k, v)
        if status is JobStatus.RUNNING and not base.started_at:
            base.started_at = utc_now_iso()
        if status in (JobStatus.COMPLETED, JobStatus.FAILED) and not base.finished_at:
            base.finished_at = utc_now_iso()
        await self.put(base)
        return base
