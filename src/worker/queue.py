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
