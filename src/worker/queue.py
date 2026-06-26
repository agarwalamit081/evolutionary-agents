"""Redis Streams run queue — the api↔worker seam (Phase 2b).

The API enqueues ``RunJob`` payloads to the ``turing:runs`` stream; worker
processes consume them through a consumer group (``turing-workers``). Delivery
is at-least-once: a consumer ``XACK``s only AFTER the run's checkpoint is
durable (see ``runner.RunConsumer``), so a crash between claim and ack leaves
the entry in the group's pending-entries list for ``reclaim_stale`` (XAUTOCLAIM)
to hand to another consumer — which then resumes from the last checkpoint
because ``thread_id = api-{run_id}`` is stable across redelivery.

All commands go through ``redis.asyncio``; nothing is string-interpolated.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.worker.schema import RunJob

if TYPE_CHECKING:
    import redis.asyncio as aioredis

    from src.config.settings import WorkerSettings


# (stream entry id, deserialized job). entry id is the XACK handle.
StreamEntry = tuple[str, RunJob]


def _decode_fields(fields: dict[bytes | str, bytes | str]) -> dict[str, str]:
    """Normalize a stream entry's fields to ``dict[str, str]`` (bytes-safe)."""
    out: dict[str, str] = {}
    for k, v in fields.items():
        key = k.decode() if isinstance(k, bytes) else str(k)
        val = v.decode() if isinstance(v, bytes) else str(v)
        out[key] = val
    return out


def _norm_token(value: Any) -> str:
    """Normalize a stored lease-lock value to ``str`` for token comparison.

    Under ``decode_responses=False`` (this client's mode) ``GET`` returns bytes or
    ``None``; a missing lock is treated as the empty string so it never equals a
    real (non-empty) token.
    """
    if value is None:
        return ""
    return value.decode() if isinstance(value, (bytes, bytearray)) else str(value)


def _entry_to_job(entry_id: Any, fields: dict[bytes | str, bytes | str]) -> StreamEntry:
    decoded = _decode_fields(fields)
    raw = decoded.get("job", "")
    if not raw:
        raise ValueError(f"stream entry {entry_id!r} has no 'job' field")
    # decode bytes entry ids (decode_responses=False) so the XACK handle is the
    # real id, not the ``"b'1-0'"`` repr that bare ``str()`` produces on bytes.
    eid = entry_id.decode() if isinstance(entry_id, bytes) else str(entry_id)
    return eid, RunJob.model_validate_json(raw)


class RunsQueue:
    """Producer + consumer surface over the ``turing:runs`` stream."""

    def __init__(self, redis_client: aioredis.Redis, settings: WorkerSettings) -> None:
        self._redis = redis_client
        self._s = settings
        self._stream = settings.runs_stream
        self._group = settings.group

    @property
    def stream(self) -> str:
        return self._stream

    @property
    def group(self) -> str:
        return self._group

    async def ensure_group(self) -> None:
        """Create the consumer group (idempotent). Reads from the stream start.

        ``id="0-0"`` so a group created AFTER messages exist still receives them;
        ``mkstream=True`` so the very first enqueue need not have pre-created the
        stream. BUSYGROUP (already exists) is swallowed — the desired end state.
        """
        from redis.exceptions import ResponseError

        try:
            await self._redis.xgroup_create(
                self._stream, self._group, id="0-0", mkstream=True
            )
            logger.info(f"Created consumer group {self._group!r} on {self._stream!r}")
        except ResponseError as exc:
            if "BUSYGROUP" in str(exc):
                return
            raise

    async def enqueue(self, job: RunJob) -> str:
        """Append a job to the stream. Returns the new entry id (XACK handle)."""
        entry_id = await self._redis.xadd(
            self._stream, {"job": job.model_dump_json()}
        )
        # decode bytes (decode_responses=False) so the XACK handle is the real
        # id, not the ``"b'…-0'"`` repr that bare ``str()`` yields on bytes.
        return entry_id.decode() if isinstance(entry_id, bytes) else str(entry_id)

    async def read_new(self, count: int | None = None) -> list[StreamEntry]:
        """Claim NEW (never-delivered) entries via ``XREADGROUP … >``.

        ``block`` is the configured ``block_ms``. Returns ``[]`` when nothing is
        available (each claimed entry is now in this consumer's pending list
        until ``ack`` removes it).
        """
        n = count if count is not None else self._s.read_batch_size
        raw = await self._redis.xreadgroup(
            self._group,
            self._s.consumer_name,
            {self._stream: ">"},
            count=n,
            block=self._s.block_ms,
        )
        return _flatten(raw)

    async def reclaim_stale(self, count: int | None = None) -> list[StreamEntry]:
        """Reclaim entries stuck idle > ``reclaim_min_idle_ms`` (crash recovery).

        ``XAUTOCLAIM`` hands pending entries from a dead/late consumer to this
        one. A worker that crashed after claiming but before acking is exactly
        the case this covers: the entry idles past the threshold and another
        worker picks up the run, resuming from the last checkpoint.
        """
        n = count if count is not None else self._s.read_batch_size
        raw = await self._redis.xautoclaim(
            self._stream,
            self._group,
            self._s.consumer_name,
            min_idle_time=self._s.reclaim_min_idle_ms,
            start_id="0-0",
            count=n,
        )
        return _autoclaim_flatten(raw)

    async def ack(self, entry_ids: Iterable[str]) -> int:
        """Acknowledge entries (remove from the group's pending list)."""
        ids = [eid for eid in entry_ids if eid]
        if not ids:
            return 0
        return int(await self._redis.xack(self._stream, self._group, *ids))

    async def delete_entry(self, entry_id: str) -> bool:
        """Terminal removal of one entry: ``XACK`` then ``XDEL`` (P1).

        Cancel (``POST /runs/{id}/cancel``) calls this the instant its flag is
        set so the pending entry can never be redelivered by ``reclaim_stale``
        (XAUTOCLAIM) to a peer worker — which would otherwise resume the run
        from its last checkpoint and burn tokens while the in-flight owner is
        still (cooperatively) winding down. ``XACK`` drops it from the group's
        pending-entries list (the only set ``XAUTOCLAIM`` re-hands out); ``XDEL``
        then drops it from the stream body so cancelled runs don't accumulate.

        Safe to race with the in-flight worker's own terminal ``ack``: ``XACK``
        on an already-acked entry is a 0-return no-op, and ``XDEL`` on an entry
        a consumer is mid-processing does not interrupt it (the record is gone
        but the in-memory job is unaffected — and the cancel flag stops that
        worker at its next iteration anyway). Best-effort: a Redis hiccup logs
        a WARNING and returns False (the worker's own ack + the run-level
        timeout remain the backstop).
        """
        if not entry_id:
            return False
        try:
            await self._redis.xack(self._stream, self._group, entry_id)
            await self._redis.xdel(self._stream, entry_id)
            return True
        except Exception as exc:  # noqa: BLE001 — best-effort; never break cancel
            logger.warning(f"delete_entry failed for {entry_id}: {exc}")
            return False

    async def pending_count(self) -> int:
        """Number of entries pending (delivered but not acked) in the group.

        redis-py returns the ``XPENDING`` summary as a tuple ``(count, …)``;
        some backends (fakeredis, Valkey forks) return a dict
        ``{"pending": count, …}``. Handle both so this never raises.
        """
        summary = await self._redis.xpending(self._stream, self._group)
        try:
            if isinstance(summary, dict):
                return int(summary.get("pending", 0))
            return int(summary[0])
        except (TypeError, IndexError, KeyError, ValueError):
            return 0

    async def record_attempt(self, run_id: str) -> int:
        """Increment + return the per-run delivery-attempt count (dead-letter gate).

        Keyed by ``run_id`` (stable across XAUTOCLAIM redelivery —
        ``thread_id = api-{run_id}``), so a job that keeps failing accumulates
        attempts whether it's retried by the same consumer or reclaimed by a peer.
        The consumer compares the returned count to
        ``dead_letter_max_attempts``: at the cap it XACKs + marks FAILED
        permanently instead of leaving the entry pending for yet another identical
        retry (Bug B — without this a DETERMINISTIC crash redelivered forever).
        TTL-bounded by ``status_ttl_s`` so the counter self-cleans with the run's
        status hash. Non-fatal best-effort: an INCR hiccup returns 0 (under-count
        → at worst one extra retry, never an infinite loop, because the executor
        exception itself is the retry signal).
        """
        key = f"{self._stream}:attempts:{run_id}"
        try:
            count = await self._redis.incr(key)
            # status_ttl_s == 0 means "no TTL" (the convention the status store uses
            # for tests): EXPIRE 0 would DELETE the key, resetting the counter every
            # call. Only set a TTL when one is actually configured.
            if self._s.status_ttl_s > 0:
                await self._redis.expire(key, self._s.status_ttl_s)
            return int(count)
        except Exception as exc:  # noqa: BLE001 — best-effort; redelivery still bounded by caller
            logger.warning(f"record_attempt INCR failed for {run_id}: {exc}")
            return 0

    # ─── Per-run lease lock (Bug C — concurrent double-claim) ──────────────
    #
    # ``reclaim_min_idle_ms`` (XAUTOCLAIM) gates crash recovery: a pending entry
    # idle past the threshold is reassigned to another consumer. But a normal run
    # outlasts the threshold, so a peer worker steals a STILL-HEALTHY in-flight
    # entry and processes the SAME goal a second time concurrently (observed live:
    # one entry claimed by both workers; acked=1 by the first, acked=0 by the
    # second). The lease lock is the hard guard: a worker acquires it SET-NX per
    # run_id before doing any work; a second claimant finds it held and SKIPS
    # (returns without acking — the rightful owner's XACK removes the entry
    # group-wide, so a skipped entry does not pile up). Renewed every ttl/3 while
    # the run is live, released on completion; a crash lets it expire so
    # ``reclaim_stale`` can then soundly hand the run to a peer.
    #
    # No Lua: the lupa-backed EVAL path is unavailable against fakeredis (which
    # the hermetic tests use), so the compare-and-set on release/renew is a
    # WATCH/MULTI transaction instead — fakeredis reuses the real ``redis.asyncio``
    # Pipeline, so the optimistic-lock semantics are faithfully exercised.

    def lock_key(self, run_id: str) -> str:
        """The per-run lease key, scoped to this stream so distinct streams (and
        the test stream) never collide."""
        return f"{self._stream}:lock:{run_id}"

    async def try_lock(self, run_id: str, token: str, ttl_s: int) -> bool:
        """Acquire the per-run lease atomically (``SET … NX EX``). Returns True
        iff this caller won the race.

        A single ``SET … NX EX`` is atomic on its own — no WATCH/Lua needed. The
        ``token`` is a unique-per-attempt secret the holder compares against on
        release/renew so a stale holder (whose TTL expired) cannot clobber a fresh
        owner's lock. ``ttl_s <= 0`` disables the lease (legacy single-worker
        behavior) — never do that in a multi-worker pool.
        """
        if ttl_s <= 0:
            return True  # lease disabled — behave as before the fix
        ok = await self._redis.set(self.lock_key(run_id), token, ex=int(ttl_s), nx=True)
        return bool(ok)

    async def renew_lock(self, run_id: str, token: str, ttl_s: int) -> bool:
        """Extend the lease ONLY while this holder still owns it (compare-and-expire).

        Returns False if the lock was lost (TTL expired + peer reacquired, or
        already released): the caller logs a WARNING — it does NOT abort the
        in-flight run, but another worker may then take it over once it ends. The
        TTL is the ultimate safety net, so a renewal hiccup never corrupts state.
        """
        if ttl_s <= 0:
            return True  # lease disabled
        from redis.exceptions import WatchError

        key = self.lock_key(run_id)
        try:
            async with self._redis.pipeline(transaction=True) as pipe:
                await pipe.watch(key)
                if _norm_token(await pipe.get(key)) != token:
                    await pipe.unwatch()
                    return False
                pipe.multi()
                pipe.expire(key, int(ttl_s))
                await pipe.execute()
                return True
        except WatchError:
            return False

    async def release_lock(self, run_id: str, token: str) -> bool:
        """Release the lease ONLY while this holder still owns it (compare-and-del).

        A stale holder (TTL expired, peer reacquired under a different token)
        returns False and MUST NOT delete the live owner's lock — so a long-since
        finished run's lingering release can never evict a peer that legitimately
        took over. Returns True iff this caller's token matched and the key was
        deleted.
        """
        from redis.exceptions import WatchError

        key = self.lock_key(run_id)
        try:
            async with self._redis.pipeline(transaction=True) as pipe:
                await pipe.watch(key)
                if _norm_token(await pipe.get(key)) != token:
                    await pipe.unwatch()
                    return False
                pipe.multi()
                pipe.delete(key)
                await pipe.execute()
                return True
        except WatchError:
            return False


def _flatten(raw: Any) -> list[StreamEntry]:
    """Normalize an ``XREADGROUP`` response into ``[(entry_id, RunJob), …]``.

    Shape: ``[[stream_name, [(id, {fields}), …]], …]`` or ``None``/``[]``.
    """
    entries: list[StreamEntry] = []
    if not raw:
        return entries
    for _stream, items in raw:
        for entry_id, fields in items:
            entries.append(_entry_to_job(entry_id, fields))
    return entries


def _autoclaim_flatten(raw: Any) -> list[StreamEntry]:
    """Normalize an ``XAUTOCLAIM`` response.

    redis-py 7.x returns ``(next_start_id, [(id, {fields}), …], deleted_ids)``;
    older/alternate builds may return a 2-tuple. Be defensive — never invent.
    """
    if not raw:
        return []
    claimed = raw[1] if len(raw) >= 2 else []
    entries: list[StreamEntry] = []
    for entry_id, fields in claimed or []:
        entries.append(_entry_to_job(entry_id, fields))
    return entries
