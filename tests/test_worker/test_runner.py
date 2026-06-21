"""Unit tests for the worker consumer (Phase 2b).

The ``RunConsumer`` is the at-least-once drain loop. Its executor is injected
(dependency inversion), so these tests run the real queue + status store over
fakeredis with a fake executor — no agent, no LLM. They lock the core contract:

- success → status COMPLETED + XACK (terminal);
- executor exception → status FAILED + NO ack, entry stays pending for redelivery;
- a crashed worker's job is reclaimed and retried by a second worker (resume).

That last case is the horizontal-scaling-correctness guarantee this seam adds.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.worker.queue import RunsQueue
from src.worker.runner import RunConsumer
from src.worker.schema import JobStatus, RunJob
from src.worker.status import RunStatusStore


def _ok_result() -> dict[str, Any]:
    return {
        "final_output": "answer",
        "is_complete": True,
        "iteration_count": 3,
    }


class TestRunConsumerCore:
    def test_thread_id_for_is_stable(self) -> None:
        """thread_id_for is the resume handle — deterministic per run_id."""
        assert RunConsumer.thread_id_for("r1") == "api-r1"

    async def test_process_success_marks_completed_and_acks(
        self, fake_redis, worker_settings
    ) -> None:
        queue = RunsQueue(fake_redis, worker_settings)
        store = RunStatusStore(fake_redis, worker_settings)
        job = RunJob(run_id="ok", goal="g")

        async def executor(_j: RunJob) -> dict[str, Any]:
            return _ok_result()

        consumer = RunConsumer(queue, store, executor, worker_settings)
        await queue.ensure_group()
        await queue.enqueue(job)
        # claim into the consumer's PEL first — XACK only removes DELIVERED
        # entries, so calling _process on a bare-XADD'd entry would ack 0.
        entries = await queue.read_new()
        acked = await consumer._process(entries[0][0], job)

        assert acked is True
        assert await queue.pending_count() == 0
        record = await store.get("ok")
        assert record is not None
        assert record.status is JobStatus.COMPLETED
        assert record.final_output == "answer"
        assert record.is_complete is True
        assert record.iteration_count == 3

    async def test_process_failure_marks_failed_and_does_not_ack(
        self, fake_redis, worker_settings
    ) -> None:
        """Executor raising → FAILED + the entry is NOT acked (stays pending)."""
        queue = RunsQueue(fake_redis, worker_settings)
        store = RunStatusStore(fake_redis, worker_settings)
        job = RunJob(run_id="boom", goal="g")

        async def executor(_j: RunJob) -> dict[str, Any]:
            raise RuntimeError("worker crashed mid-run")

        consumer = RunConsumer(queue, store, executor, worker_settings)
        await queue.ensure_group()
        entry_id = await queue.enqueue(job)
        await queue.read_new()  # claim into the PEL so pending_count reflects it
        acked = await consumer._process(entry_id, job)

        assert acked is False
        assert await queue.pending_count() == 1  # left for redelivery
        record = await store.get("boom")
        assert record is not None
        assert record.status is JobStatus.FAILED
        assert "crashed mid-run" in record.error

    async def test_run_once_drains_new_entries(
        self, fake_redis, worker_settings
    ) -> None:
        queue = RunsQueue(fake_redis, worker_settings)
        store = RunStatusStore(fake_redis, worker_settings)
        await queue.ensure_group()
        await queue.enqueue(RunJob(run_id="a", goal="g"))
        await queue.enqueue(RunJob(run_id="b", goal="g"))

        async def executor(j: RunJob) -> dict[str, Any]:
            return {"final_output": j.run_id, "is_complete": True}

        consumer = RunConsumer(queue, store, executor, worker_settings)
        acked = await consumer.run_once()
        assert acked == 2
        assert await queue.pending_count() == 0
        assert (await store.get("a")).status is JobStatus.COMPLETED  # type: ignore[union-attr]
        assert (await store.get("b")).status is JobStatus.COMPLETED  # type: ignore[union-attr]


class TestRunConsumerRecovery:
    async def test_crashed_job_is_reclaimed_and_retried_by_peer(
        self, fake_redis, worker_settings
    ) -> None:
        """At-least-once redelivery: worker-a fails (no ack); worker-b reclaims
        via XAUTOCLAIM and completes. The stable ``api-{run_id}`` thread means
        the retry resumes from the last checkpoint, not a fresh start."""
        settings_a = worker_settings.model_copy(update={"consumer_name": "wa"})
        settings_b = worker_settings.model_copy(update={"consumer_name": "wb"})
        queue_a = RunsQueue(fake_redis, settings_a)
        queue_b = RunsQueue(fake_redis, settings_b)
        store = RunStatusStore(fake_redis, worker_settings)
        job = RunJob(run_id="crash", goal="survive the crash")

        async def always_fail(_j: RunJob) -> dict[str, Any]:
            raise RuntimeError("boom")

        async def succeed(_j: RunJob) -> dict[str, Any]:
            return _ok_result()

        await queue_a.ensure_group()
        await queue_a.enqueue(job)

        consumer_a = RunConsumer(queue_a, store, always_fail, settings_a)
        consumer_b = RunConsumer(queue_b, store, succeed, settings_b)

        # 1st pass: worker-a claims, executor raises → FAILED, NOT acked.
        assert await consumer_a.run_once() == 0
        after_fail = await store.get("crash")
        assert after_fail is not None
        assert after_fail.status is JobStatus.FAILED
        assert await queue_a.pending_count() == 1

        # 2nd pass: worker-b reclaims the idle entry and completes it.
        assert await consumer_b.run_once() == 1
        after_ok = await store.get("crash")
        assert after_ok is not None
        assert after_ok.status is JobStatus.COMPLETED
        assert after_ok.final_output == "answer"
        assert await queue_a.pending_count() == 0


class TestRunConsumerServeLoop:
    async def test_serve_forever_drains_then_stops_on_event(
        self, fake_redis, worker_settings
    ) -> None:
        """serve_forever processes enqueued work then exits when stop is set.

        Stop is signalled FROM WITHIN the executor (same coroutine as the loop)
        rather than a sibling task: fakeredis's XREADGROUP returns immediately
        on an empty stream (it doesn't honor ``block`` / yield the event loop),
        so a sibling stop-task would be starved by the tight poll loop. Real
        Redis blocks on ``block_ms`` and yields, so production is unaffected.
        """
        queue = RunsQueue(fake_redis, worker_settings)
        store = RunStatusStore(fake_redis, worker_settings)
        await queue.ensure_group()
        await queue.enqueue(RunJob(run_id="s1", goal="g"))
        stop = asyncio.Event()

        async def executor(_j: RunJob) -> dict[str, Any]:
            stop.set()  # in-loop signal — no competing task to starve
            return _ok_result()

        consumer = RunConsumer(queue, store, executor, worker_settings)
        await consumer.serve_forever(stop)

        record = await store.get("s1")
        assert record is not None
        assert record.status is JobStatus.COMPLETED
        assert await queue.pending_count() == 0

    async def test_serve_forever_handles_cancel(
        self, fake_redis, worker_settings
    ) -> None:
        """Cancellation while a run is in flight propagates as CancelledError;
        the job stays pending for redelivery (not acked)."""
        queue = RunsQueue(fake_redis, worker_settings)
        store = RunStatusStore(fake_redis, worker_settings)
        await queue.ensure_group()
        await queue.enqueue(RunJob(run_id="c1", goal="g"))
        started = asyncio.Event()

        async def slow_executor(_j: RunJob) -> dict[str, Any]:
            started.set()
            # Suspend here so serve_forever is parked when we cancel — cancel is
            # injected at this suspension point (CancelledError is BaseException,
            # so _process's `except Exception` does NOT swallow it).
            await asyncio.sleep(100)
            return {}  # unreachable — cancelled mid-sleep

        consumer = RunConsumer(queue, store, slow_executor, worker_settings)
        task = asyncio.create_task(consumer.serve_forever())
        await started.wait()  # the run is mid-execution (loop suspended in sleep)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await queue.pending_count() == 1  # not acked → left for redelivery


class TestRunConsumerDeadLetter:
    """Bug B: a DETERMINISTIC executor failure must dead-letter after
    ``dead_letter_max_attempts`` retries instead of redelivering forever.

    Without the cap, ``reclaim_stale`` (XAUTOCLAIM, every ``reclaim_min_idle_ms``)
    re-hands the same pending entry to a worker for an identical failure — an
    infinite poison loop (observed live: 15+ identical retries over 7+ min for a
    missing-dep crash). At the cap the entry is acked (removed from the PEL) so it
    can never be redelivered again. Transient failures still retry up to the cap.
    """

    async def test_retries_below_cap_then_dead_letters(
        self, fake_redis, worker_settings
    ) -> None:
        s = worker_settings.model_copy(update={"dead_letter_max_attempts": 2})
        queue = RunsQueue(fake_redis, s)
        store = RunStatusStore(fake_redis, s)
        job = RunJob(run_id="poison", goal="g")

        async def always_raise(_j: RunJob) -> dict[str, Any]:
            raise RuntimeError("deterministic boom")

        consumer = RunConsumer(queue, store, always_raise, s)
        await queue.ensure_group()
        entry_id = await queue.enqueue(job)
        await queue.read_new()  # claim into the PEL so pending_count reflects it

        # attempt 1 (< cap 2): FAILED, NOT acked → left for redelivery.
        assert await consumer._process(entry_id, job) is False
        assert await queue.pending_count() == 1
        rec1 = await store.get("poison")
        assert rec1 is not None
        assert rec1.status is JobStatus.FAILED
        assert "deterministic boom" in (rec1.error or "")
        assert "dead-lettered" not in (rec1.error or "")  # pre-cap: raw error only

        # attempt 2 (>= cap 2): DEAD-LETTERED — acked (terminal), NOT redelivered.
        assert await consumer._process(entry_id, job) is True
        assert await queue.pending_count() == 0
        rec2 = await store.get("poison")
        assert rec2 is not None
        assert rec2.status is JobStatus.FAILED
        assert "dead-lettered after 2 attempts" in (rec2.error or "")


class TestRunConsumerLeaseLock:
    """Bug C — concurrent double-claim. ``reclaim_min_idle_ms`` (XAUTOCLAIM) is
    shorter than a normal run, so a peer worker steals a still-healthy in-flight
    entry and runs the SAME goal a second time. The per-run lease makes the second
    claimant SKIP (return False without acking, without calling the executor). This
    is the regression the lock exists for."""

    async def test_second_consumer_skips_in_flight_run(
        self, fake_redis, worker_settings
    ) -> None:
        settings_a = worker_settings.model_copy(update={"consumer_name": "wa"})
        settings_b = worker_settings.model_copy(update={"consumer_name": "wb"})
        queue_a = RunsQueue(fake_redis, settings_a)
        queue_b = RunsQueue(fake_redis, settings_b)
        store = RunStatusStore(fake_redis, worker_settings)
        job = RunJob(run_id="contended", goal="g")

        started = asyncio.Event()
        release_a = asyncio.Event()
        a_calls = 0

        async def slow_executor_a(_j: RunJob) -> dict[str, Any]:
            nonlocal a_calls
            a_calls += 1
            started.set()
            await release_a.wait()  # hold the lease — run outlasts reclaim_min_idle_ms
            return _ok_result()

        b_calls = 0

        async def executor_b(_j: RunJob) -> dict[str, Any]:
            nonlocal b_calls
            b_calls += 1
            return _ok_result()

        consumer_a = RunConsumer(queue_a, store, slow_executor_a, settings_a)
        consumer_b = RunConsumer(queue_b, store, executor_b, settings_b)

        await queue_a.ensure_group()
        entry_id = await queue_a.enqueue(job)
        await queue_a.read_new()  # claim into the group PEL so A's ack resolves

        # Worker A claims → acquires the lease → runs (parked in slow_executor).
        task_a = asyncio.create_task(consumer_a._process(entry_id, job))
        await started.wait()  # A is mid-run, holding the lease

        # Worker B handed the same entry while A still runs: B must NOT acquire the
        # lease → it skips (False, no executor call, no ack). The bug, fixed.
        assert await consumer_b._process(entry_id, job) is False
        assert b_calls == 0  # B never ran the executor

        # Let A finish; it completes + acks normally.
        release_a.set()
        assert await task_a is True
        assert a_calls == 1  # A ran exactly once

        # A released the lease on completion → a fresh claim can now acquire it.
        assert (
            await queue_b.try_lock(job.run_id, "fresh", worker_settings.lock_ttl_s)
            is True
        )

    async def test_lease_released_on_executor_exception(
        self, fake_redis, worker_settings
    ) -> None:
        """The lease MUST be released even when the executor raises — otherwise a
        legitimate redelivery (reclaim_stale) would skip forever behind a lingering
        lock from the failed attempt."""
        queue = RunsQueue(fake_redis, worker_settings)
        store = RunStatusStore(fake_redis, worker_settings)
        job = RunJob(run_id="fails", goal="g")

        async def boom(_j: RunJob) -> dict[str, Any]:
            raise RuntimeError("boom")

        consumer = RunConsumer(queue, store, boom, worker_settings)
        await queue.ensure_group()
        entry_id = await queue.enqueue(job)
        await queue.read_new()  # claim into the PEL so pending_count reflects it

        assert await consumer._process(entry_id, job) is False  # NOT acked → redelivered

        # Lease released in finally → a new claim can acquire it immediately.
        assert await queue.try_lock(job.run_id, "x", worker_settings.lock_ttl_s) is True

    async def test_no_skip_when_lease_disabled(
        self, fake_redis, worker_settings
    ) -> None:
        """With ``lock_ttl_s=0`` the lease is disabled: a second consumer does NOT
        skip — it proceeds (legacy double-processing), exactly the bug the lease
        (when enabled) prevents. Confirms the skip is solely the lease's doing, not
        incidental state, and the disabled happy path is unaffected."""
        s = worker_settings.model_copy(update={"lock_ttl_s": 0})
        sa = s.model_copy(update={"consumer_name": "wa"})
        sb = s.model_copy(update={"consumer_name": "wb"})
        queue_a = RunsQueue(fake_redis, sa)
        queue_b = RunsQueue(fake_redis, sb)
        store = RunStatusStore(fake_redis, s)
        job = RunJob(run_id="no-lock", goal="g")

        a_calls = 0

        async def exec_a(_j: RunJob) -> dict[str, Any]:
            nonlocal a_calls
            a_calls += 1
            return _ok_result()

        b_calls = 0

        async def exec_b(_j: RunJob) -> dict[str, Any]:
            nonlocal b_calls
            b_calls += 1
            return _ok_result()

        consumer_a = RunConsumer(queue_a, store, exec_a, sa)
        consumer_b = RunConsumer(queue_b, store, exec_b, sb)

        await queue_a.ensure_group()
        entry_id = await queue_a.enqueue(job)
        await queue_a.read_new()  # claim into the group PEL

        # Lease disabled → the guard is OFF: BOTH consumers run their executors
        # (the legacy concurrent double-processing the lease exists to prevent).
        # The decisive assertion is executor call counts, not the return value: the
        # group-wide XACK means only the first acker returns True (acked=1) and the
        # second returns False (acked=0) regardless of the lease.
        await consumer_a._process(entry_id, job)
        await consumer_b._process(entry_id, job)
        assert a_calls == 1
        assert b_calls == 1  # B ran too — the bug, present only when the lease is off

