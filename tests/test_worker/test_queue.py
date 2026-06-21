"""Unit tests for the Redis Streams run queue (Phase 2b).

Covers the producer/consumer surface over ``turing:runs``: group creation
idempotency, enqueue/read_new roundtrip fidelity, empty-read non-blocking,
ack removal from the pending list, pending-count tracking, and the
crash-recovery reclaim (XAUTOCLAIM) that underpins at-least-once delivery.
All hermetic against fakeredis (Streams-faithful).
"""

from __future__ import annotations

import pytest

from src.worker.queue import RunsQueue, _entry_to_job
from src.worker.schema import RunJob


class TestRunsQueue:
    async def test_ensure_group_is_idempotent(
        self, fake_redis, worker_settings
    ) -> None:
        """A second ensure_group must not raise BUSYGROUP."""
        q = RunsQueue(fake_redis, worker_settings)
        await q.ensure_group()
        await q.ensure_group()  # second call swallowed, not raised
        assert await q.pending_count() == 0

    async def test_enqueue_then_read_new_roundtrip(
        self, fake_redis, worker_settings
    ) -> None:
        """A serialized RunJob survives the XADD → XREADGROUP roundtrip."""
        q = RunsQueue(fake_redis, worker_settings)
        await q.ensure_group()
        job = RunJob(
            run_id="r1",
            goal="do the thing",
            max_iterations=12,
            no_evolution=True,
            model="glm-4.7-flash",
        )
        await q.enqueue(job)
        entries = await q.read_new()
        assert len(entries) == 1
        _entry_id, decoded = entries[0]
        assert decoded.run_id == "r1"
        assert decoded.goal == "do the thing"
        assert decoded.max_iterations == 12
        assert decoded.no_evolution is True
        assert decoded.model == "glm-4.7-flash"

    async def test_read_new_returns_empty_when_idle(
        self, fake_redis, worker_settings
    ) -> None:
        """No backlog → read_new returns [] (low block_ms, no hang)."""
        q = RunsQueue(fake_redis, worker_settings)
        await q.ensure_group()
        assert await q.read_new() == []

    async def test_read_new_is_once_per_entry(
        self, fake_redis, worker_settings
    ) -> None:
        """``>`` delivers each entry exactly once; a second read is empty."""
        q = RunsQueue(fake_redis, worker_settings)
        await q.ensure_group()
        await q.enqueue(RunJob(run_id="r1", goal="g"))
        first = await q.read_new()
        second = await q.read_new()
        assert len(first) == 1
        assert second == []

    async def test_ack_removes_from_pending(
        self, fake_redis, worker_settings
    ) -> None:
        q = RunsQueue(fake_redis, worker_settings)
        await q.ensure_group()
        await q.enqueue(RunJob(run_id="r1", goal="g"))
        entries = await q.read_new()
        assert await q.pending_count() == 1
        assert await q.ack([entries[0][0]]) == 1
        assert await q.pending_count() == 0

    async def test_ack_empty_or_blank_returns_zero(
        self, fake_redis, worker_settings
    ) -> None:
        q = RunsQueue(fake_redis, worker_settings)
        await q.ensure_group()
        assert await q.ack([]) == 0
        assert await q.ack([""]) == 0  # blank ids are filtered

    async def test_reclaim_stale_redelivers_idle_entry(
        self, fake_redis, worker_settings
    ) -> None:
        """Crash recovery: a claimed-but-unacked entry is reclaimed by another
        worker via XAUTOCLAIM (min_idle=0)."""
        settings_a = worker_settings.model_copy(update={"consumer_name": "wa"})
        settings_b = worker_settings.model_copy(update={"consumer_name": "wb"})
        q_a = RunsQueue(fake_redis, settings_a)
        q_b = RunsQueue(fake_redis, settings_b)
        await q_a.ensure_group()
        await q_a.enqueue(RunJob(run_id="x", goal="g"))

        claimed = await q_a.read_new()  # worker-a claims; never acks (simulated crash)
        assert len(claimed) == 1
        assert await q_a.pending_count() == 1

        reclaimed = await q_b.reclaim_stale()  # worker-b picks up the idle entry
        assert len(reclaimed) == 1
        assert reclaimed[0][1].run_id == "x"
        assert await q_a.pending_count() == 1  # still pending, now under wb
        assert await q_b.ack([reclaimed[0][0]]) == 1
        assert await q_a.pending_count() == 0

    async def test_reclaim_stale_empty_when_nothing_idle(
        self, fake_redis, worker_settings
    ) -> None:
        q = RunsQueue(fake_redis, worker_settings)
        await q.ensure_group()
        assert await q.reclaim_stale() == []


def test_entry_to_job_raises_on_missing_job_field() -> None:
    """A malformed entry (no 'job' field) raises, not silently dropped."""
    with pytest.raises(ValueError):
        _entry_to_job("1-0", {"notjob": "x"})


def test_entry_to_job_decodes_bytes_fields() -> None:
    """Bytes field names/values from a raw client are decoded before parse."""
    entry_id, job = _entry_to_job(
        b"1-0", {b"job": b'{"run_id":"r9","goal":"g"}'}
    )
    assert entry_id == "1-0"
    assert job.run_id == "r9"
