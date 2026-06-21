"""Live Redis integration test for the worker seam — skips if Redis is down.

Exercises the real ``XADD → XREADGROUP → XACK`` roundtrip and the real
redis-py ``XPENDING`` tuple shape (the hermetic fakeredis tests cover the dict
shape). Run against the compose stack (host :6380) or ``REDIS_URL``; skipped
when Redis is unreachable so the unit suite stays hermetic and CI-safe.
"""

from __future__ import annotations

import pytest
import redis.asyncio as aioredis

from src.config import get_settings
from src.worker.queue import RunsQueue
from src.worker.schema import RunJob

_LIVE_STREAM = "turing:runs:test-live"
_LIVE_GROUP = "turing-workers-test-live"


@pytest.fixture
async def live_redis():
    """A real Redis client, or skip the whole module if unreachable.

    Bounded connect/socket timeouts so a DOWN Redis fails fast into a skip
    instead of hanging the suite on the default (unbounded) connect.
    """
    settings = get_settings()
    client = aioredis.from_url(
        settings.redis.redis_url,
        socket_connect_timeout=2,
        socket_timeout=2,
    )
    try:
        await client.ping()
    except Exception as exc:
        await client.aclose()
        pytest.skip(f"Redis unreachable at {settings.redis.redis_url}: {exc}")
    yield client
    # Best-effort cleanup of our test stream + group.
    try:
        await client.delete(_LIVE_STREAM)
    except Exception:
        pass
    await client.aclose()


async def test_live_enqueue_claim_ack_roundtrip(live_redis) -> None:
    """Full seam on real Redis: enqueue → claim → pending=1 → ack → pending=0."""
    settings = get_settings().worker.model_copy(
        update={
            "runs_stream": _LIVE_STREAM,
            "group": _LIVE_GROUP,
            "block_ms": 100,
        }
    )
    q = RunsQueue(live_redis, settings)
    await live_redis.delete(_LIVE_STREAM)  # clean slate (drops any stale group)
    await q.ensure_group()

    job = RunJob(run_id="live1", goal="roundtrip on real redis")
    entry_id = await q.enqueue(job)
    claimed = await q.read_new()
    assert len(claimed) == 1
    assert claimed[0][1].run_id == "live1"

    # Real redis-py returns XPENDING summary as a tuple (count, …); pending_count
    # must read index 0 correctly here (the dict path is covered hermetically).
    assert await q.pending_count() == 1

    assert await q.ack([entry_id]) == 1
    assert await q.pending_count() == 0


async def test_live_ensure_group_idempotent(live_redis) -> None:
    settings = get_settings().worker.model_copy(
        update={"runs_stream": _LIVE_STREAM, "group": _LIVE_GROUP}
    )
    q = RunsQueue(live_redis, settings)
    await live_redis.delete(_LIVE_STREAM)
    await q.ensure_group()
    await q.ensure_group()  # BUSYGROUP swallowed on real Redis
    assert await q.pending_count() == 0
