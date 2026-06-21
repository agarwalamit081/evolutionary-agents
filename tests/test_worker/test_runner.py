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

