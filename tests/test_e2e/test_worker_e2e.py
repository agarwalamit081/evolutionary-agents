"""Live e2e for the Phase-3 worker role (cost-bounded, @pytest.mark.e2e).

The hermetic ``tests/test_worker/test_role_split_integration.py`` proves the
api→worker→deliverable composition with a FAKE executor (no LLM, no cost). This
is its live counterpart: a worker drains a real RunJob off a live Redis stream
and runs the REAL ``default_agent_executor`` → ``execute_run`` (LLM +
AsyncPostgresSaver checkpointer + DB-backed memory/cost-tracker) to completion.

Skips unless the stack is up (Redis reachable) AND DEEPSEEK_API_KEY is set (the
default provider). Run it explicitly:
    python -m pytest tests/test_e2e/test_worker_e2e.py -m e2e -s

Scope / cost discipline: a trivial one-step goal, ``no_evolution=True``, a low
iteration cap — this run is a few cheap-model calls, well under the ~$3 bound.
The richer signals called out by the plan — evolution firing IN the worker, and
checkpoint resume across a worker kill+restart — are validated elsewhere:
evolution-in-worker by the battery-04 golden runs + the Phase-4 evolve→execute
edge; checkpoint resume by the AsyncPostgresSaver fix (P2b) and the hermetic
redelivery test (FAILED → reclaim_stale → retry). This test isolates the one
thing only a live run can prove: execute_run actually runs end-to-end inside a
worker against real dependencies.
"""

from __future__ import annotations

import os
import uuid

import pytest
import redis.asyncio as aioredis

from src.config import get_settings
from src.worker.executors import default_agent_executor
from src.worker.queue import RunsQueue
from src.worker.runner import RunConsumer
from src.worker.schema import JobStatus, RunJob
from src.worker.status import RunStatusStore

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not os.environ.get("DEEPSEEK_API_KEY"),
        reason="Requires DEEPSEEK_API_KEY (default provider) + a live stack for worker e2e",
    ),
]

_LIVE_STREAM = "turing:runs:e2e-worker"
_LIVE_GROUP = "turing-workers-e2e-worker"


@pytest.fixture
async def live_redis():
    """A real Redis client, or skip the module if unreachable (bounded timeout)."""
    settings = get_settings()
    client = aioredis.from_url(
        settings.redis.redis_url, socket_connect_timeout=2, socket_timeout=2
    )
    try:
        await client.ping()  # type: ignore[union-attr]  # redis.asyncio stub returns sync bool
    except Exception as exc:
        await client.aclose()
        pytest.skip(f"Redis unreachable at {settings.redis.redis_url}: {exc}")
    yield client
    try:
        await client.delete(_LIVE_STREAM)
    except Exception:  # noqa: BLE001 — best-effort cleanup
        pass
    await client.aclose()


@pytest.mark.asyncio
async def test_worker_runs_real_execute_run_to_completion(live_redis) -> None:
    """A real worker drain runs execute_run end-to-end → status COMPLETED."""
    ws = get_settings().worker.model_copy(
        update={
            "runs_stream": _LIVE_STREAM,
            "group": _LIVE_GROUP,
            "block_ms": 100,
            "reclaim_min_idle_ms": 0,
        }
    )
    run_id = f"p3-e2e-{uuid.uuid4().hex[:8]}"
    queue = RunsQueue(live_redis, ws)
    await live_redis.delete(_LIVE_STREAM)  # clean slate (drops any stale group)
    await queue.ensure_group()

    # Trivial deliverable goal, cheap + no evolution → a few cheap-model calls.
    job = RunJob(
        run_id=run_id,
        goal="Write the exact text 'hello from worker' to a file results/worker_e2e.txt and confirm it.",
        max_iterations=4,
        no_evolution=True,
    )
    await queue.enqueue(job)

    store = RunStatusStore(live_redis, ws)
    consumer = RunConsumer(queue, store, default_agent_executor, ws)

    acked = await consumer.run_once()
    assert acked == 1, "the worker should have drained and acked the enqueued job"

    status = await store.get(run_id)
    assert status is not None, "run status was never written"
    assert status.status == JobStatus.COMPLETED, f"run did not complete: {status.error}"
    # final_output is non-empty → execute_run produced a real answer (the robust
    # completion signal; a brittle file-path deliverable check is the golden
    # battery's job, not this worker-seam e2e's).
    assert status.final_output, "completed run produced no final_output"
    # The entry is terminal (acked) — the worker did not leave it pending.
    assert await queue.pending_count() == 0
