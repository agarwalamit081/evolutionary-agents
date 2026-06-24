"""Per-run status store backed by a Redis hash (Phase 2b).

The worker writes each run's lifecycle (queued → running → completed/failed) to
``turing:run:{run_id}``; the API reads it for ``GET /runs/{run_id}`` so a client
can poll without the worker holding an open HTTP connection. Hash TTL bounded by
``WorkerSettings.status_ttl_s`` so the store self-cleans (Redis rule: never
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

    _KEY_PREFIX = "turing:run:"  # status hash key = prefix + run_id
    _CANCEL_PREFIX = "turing:runs:cancel:"  # cancel flag key = prefix + run_id

    def __init__(
        self,
        redis_client: aioredis.Redis,
        settings: WorkerSettings,
    ) -> None:
        self._redis = redis_client
        self._ttl = settings.status_ttl_s

    def _key(self, run_id: str) -> str:
        return f"{self._KEY_PREFIX}{run_id}"

    def _cancel_key(self, run_id: str) -> str:
        return f"{self._CANCEL_PREFIX}{run_id}"

    async def request_cancel(self, run_id: str, ttl: int | None = None) -> None:
        """Set the graceful-cancel flag (``POST /runs/{run_id}/cancel``).

        The worker's per-iteration progress callback polls ``is_cancelled`` and
        raises ``RunCancelled`` (~1-iteration latency). The flag is a bare key —
        its PRESENCE is the signal — so a repeat POST is a no-op (idempotent).
        TTL reuses ``status_ttl_s`` (a ``0`` disables it, as in tests) so the
        flag outlives any run yet still self-cleans; an explicit ``ttl`` wins.
        Best-effort — never raises: a Redis hiccup must not break the cancel
        path (the run-level timeout is the ultimate bound).
        """
        seconds = self._ttl if ttl is None else ttl
        try:
            if seconds > 0:
                await self._redis.set(self._cancel_key(run_id), "1", ex=seconds)  # type: ignore[arg-type]
            else:
                await self._redis.set(self._cancel_key(run_id), "1")  # type: ignore[arg-type]
        except Exception:
            pass

    async def is_cancelled(self, run_id: str) -> bool:
        """``True`` iff the cancel flag is set. Fails OPEN on Redis error.

        On a Redis error we return ``False`` (run continues): cancel is
        cooperative/best-effort, and a transient blip must not spuriously kill
        an in-flight run. If Redis is fully down the flag could not have been
        SET either (``request_cancel`` is best-effort), so fail-open keeps the
        two sides consistent; the run-level timeout remains the hard bound.
        """
        try:
            return bool(await self._redis.exists(self._cancel_key(run_id)))
        except Exception:
            return False

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
        # finished_at stamps every terminal transition: COMPLETED/FAILED plus the
        # resumable-terminal run-control statuses (TIMEOUT / BUDGET_EXHAUSTED /
        # CANCELLED) so the API reports when the run ended regardless of which
        # clean-stop path it took.
        if (
            status
            in (
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.TIMEOUT,
                JobStatus.BUDGET_EXHAUSTED,
                JobStatus.CANCELLED,
            )
            and not base.finished_at
        ):
            base.finished_at = utc_now_iso()
        await self.put(base)
        return base
