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

from src.llm.exceptions import BudgetExhaustedError
from src.runner import RunCancelled
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

        async def executor(_j: RunJob, _on_progress: Any = None) -> dict[str, Any]:
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

    async def test_process_reports_iteration_count_mid_run(
        self, fake_redis, worker_settings
    ) -> None:
        """#255: the executor's on_progress callback mirrors the live
        iteration_count onto the RUNNING status record mid-run, so GET
        /runs/<id> no longer reports iteration_count=0 until completion."""
        queue = RunsQueue(fake_redis, worker_settings)
        store = RunStatusStore(fake_redis, worker_settings)
        job = RunJob(run_id="prog", goal="g")

        mid_run_counts: list[int] = []

        async def executor(_j: RunJob, on_progress: Any = None) -> dict[str, Any]:
            # Simulate execute_run advancing through iterations (as astream
            # would). Each report must land on the RUNNING status record.
            assert on_progress is not None, "consumer must pass a progress callback"
            for ic in (1, 2, 3):
                await on_progress(ic)
                record = await store.get("prog")
                assert record is not None
                assert record.iteration_count == ic
                assert record.status is JobStatus.RUNNING
                mid_run_counts.append(record.iteration_count)
            return {"final_output": "done", "is_complete": True, "iteration_count": 3}

        consumer = RunConsumer(queue, store, executor, worker_settings)
        await queue.ensure_group()
        await queue.enqueue(job)
        entries = await queue.read_new()
        acked = await consumer._process(entries[0][0], job)

        assert acked is True
        # Progress was visible mid-run (not stuck at 0 throughout).
        assert mid_run_counts == [1, 2, 3]
        # The terminal COMPLETED record carries the authoritative count.
        final = await store.get("prog")
        assert final is not None
        assert final.status is JobStatus.COMPLETED
        assert final.iteration_count == 3

    async def test_process_failure_marks_failed_and_does_not_ack(
        self, fake_redis, worker_settings
    ) -> None:
        """Executor raising → FAILED + the entry is NOT acked (stays pending)."""
        queue = RunsQueue(fake_redis, worker_settings)
        store = RunStatusStore(fake_redis, worker_settings)
        job = RunJob(run_id="boom", goal="g")

        async def executor(_j: RunJob, _on_progress: Any = None) -> dict[str, Any]:
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

        async def executor(j: RunJob, _on_progress: Any = None) -> dict[str, Any]:
            return {"final_output": j.run_id, "is_complete": True}

        consumer = RunConsumer(queue, store, executor, worker_settings)
        acked = await consumer.run_once()
        assert acked == 2
        assert await queue.pending_count() == 0
        assert (await store.get("a")).status is JobStatus.COMPLETED  # type: ignore[union-attr]
        assert (await store.get("b")).status is JobStatus.COMPLETED  # type: ignore[union-attr]


class TestRunConsumerTerminalGuard:
    """Fix A1: a DUPLICATE stream entry for a run already in a terminal state is
    acked-and-deleted WITHOUT re-running the goal. This is the q04/q06
    redelivery-forever fix — a completed run that lost its ack used to re-execute
    from checkpoint on every duplicate. Terminal-but-retryable ``FAILED`` is
    intentionally excluded so transient failures stay redeliverable."""

    async def _consumer_with_poison_executor(
        self, fake_redis, worker_settings
    ) -> tuple[RunConsumer, RunsQueue, RunStatusStore, list[int]]:
        """A consumer whose executor MUST NEVER run — it appends to ``calls`` and
        raises. If the terminal guard works, ``calls`` stays empty."""
        queue = RunsQueue(fake_redis, worker_settings)
        store = RunStatusStore(fake_redis, worker_settings)
        calls: list[int] = []

        async def executor(_j: RunJob, _on_progress: Any = None) -> dict[str, Any]:
            calls.append(1)
            raise AssertionError("terminal run must NOT re-execute the executor")

        consumer = RunConsumer(queue, store, executor, worker_settings)
        await queue.ensure_group()
        return consumer, queue, store, calls

    async def test_completed_duplicate_skipped_not_rerun(
        self, fake_redis, worker_settings
    ) -> None:
        """A run that already reached COMPLETED on a prior delivery is acked +
        deleted; the executor is never invoked."""
        run_id = "dup-done"
        consumer, queue, store, calls = await self._consumer_with_poison_executor(
            fake_redis, worker_settings
        )
        job = RunJob(run_id=run_id, goal="g")
        # Simulate the prior completed delivery that lost its XACK.
        await store.mark(
            run_id,
            RunConsumer.thread_id_for(run_id),
            JobStatus.COMPLETED,
            final_output="already done",
            is_complete=True,
        )
        entry_id = await queue.enqueue(job)
        await queue.read_new()  # claim into PEL

        acked = await consumer._process(entry_id, job)

        assert acked is True  # the duplicate entry was removed
        assert await queue.pending_count() == 0
        assert calls == []  # executor never ran
        # The terminal record is unchanged (still COMPLETED, not re-RUNNING).
        rec = await store.get(run_id)
        assert rec is not None
        assert rec.status is JobStatus.COMPLETED
        assert rec.final_output == "already done"

    async def test_cancelled_duplicate_skipped(self, fake_redis, worker_settings) -> None:
        run_id = "dup-can"
        consumer, queue, store, calls = await self._consumer_with_poison_executor(
            fake_redis, worker_settings
        )
        await store.mark(
            run_id, RunConsumer.thread_id_for(run_id), JobStatus.CANCELLED
        )
        entry_id = await queue.enqueue(RunJob(run_id=run_id, goal="g"))
        await queue.read_new()

        assert await consumer._process(entry_id, RunJob(run_id=run_id, goal="g")) is True
        assert calls == []
        assert (await store.get(run_id)).status is JobStatus.CANCELLED  # type: ignore[union-attr]

    async def test_budget_exhausted_duplicate_skipped(
        self, fake_redis, worker_settings
    ) -> None:
        run_id = "dup-budget"
        consumer, queue, store, calls = await self._consumer_with_poison_executor(
            fake_redis, worker_settings
        )
        await store.mark(
            run_id, RunConsumer.thread_id_for(run_id), JobStatus.BUDGET_EXHAUSTED
        )
        entry_id = await queue.enqueue(RunJob(run_id=run_id, goal="g"))
        await queue.read_new()

        assert await consumer._process(entry_id, RunJob(run_id=run_id, goal="g")) is True
        assert calls == []

    async def test_failed_run_not_skipped_stays_retryable(
        self, fake_redis, worker_settings
    ) -> None:
        """``FAILED`` is excluded from the skip set: a prior failure stays
        redeliverable (bounded by the dead-letter cap) — the guard must not trap
        retryable failures."""
        queue = RunsQueue(fake_redis, worker_settings)
        store = RunStatusStore(fake_redis, worker_settings)
        run_id = "dup-fail"
        re_runs: list[int] = []

        async def executor(_j: RunJob, _on_progress: Any = None) -> dict[str, Any]:
            re_runs.append(1)
            return _ok_result()  # succeeds on the redelivery

        consumer = RunConsumer(queue, store, executor, worker_settings)
        await queue.ensure_group()
        await store.mark(run_id, RunConsumer.thread_id_for(run_id), JobStatus.FAILED)
        entry_id = await queue.enqueue(RunJob(run_id=run_id, goal="g"))
        await queue.read_new()

        acked = await consumer._process(entry_id, RunJob(run_id=run_id, goal="g"))

        assert acked is True  # re-processed and completed
        assert re_runs == [1]  # the executor DID run (not skipped)
        assert (await store.get(run_id)).status is JobStatus.COMPLETED  # type: ignore[union-attr]


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

        async def always_fail(_j: RunJob, _on_progress: Any = None) -> dict[str, Any]:
            raise RuntimeError("boom")

        async def succeed(_j: RunJob, _on_progress: Any = None) -> dict[str, Any]:
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

        async def executor(_j: RunJob, _on_progress: Any = None) -> dict[str, Any]:
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

        async def slow_executor(_j: RunJob, _on_progress: Any = None) -> dict[str, Any]:
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

        async def always_raise(_j: RunJob, _on_progress: Any = None) -> dict[str, Any]:
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

        async def slow_executor_a(_j: RunJob, _on_progress: Any = None) -> dict[str, Any]:
            nonlocal a_calls
            a_calls += 1
            started.set()
            await release_a.wait()  # hold the lease — run outlasts reclaim_min_idle_ms
            return _ok_result()

        b_calls = 0

        async def executor_b(_j: RunJob, _on_progress: Any = None) -> dict[str, Any]:
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

        async def boom(_j: RunJob, _on_progress: Any = None) -> dict[str, Any]:
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

        async def exec_a(_j: RunJob, _on_progress: Any = None) -> dict[str, Any]:
            nonlocal a_calls
            a_calls += 1
            return _ok_result()

        b_calls = 0

        async def exec_b(_j: RunJob, _on_progress: Any = None) -> dict[str, Any]:
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


class TestRunConsumerRunControl:
    """Run-control hardening (A/D/E + F glue). Three new terminal statuses —
    TIMEOUT / BUDGET_EXHAUSTED / CANCELLED — are each acked (NOT redelivered) and
    resumable via the per-iteration AsyncPostgresSaver checkpoint. The typed
    exception handlers sit BEFORE the dead-letter ``except Exception`` so a
    timeout/budget/cancel is never mis-redelivered as a poison message; and a
    STRAY downstream ``TimeoutError`` while the wall-clock bound is UNARMED still
    falls through to the dead-letter path (preserving the prior default
    behavior). ``_resolve_timeout`` precedence is pinned so the per-run API
    override wins, an explicit ``0`` disables, and ``None`` defers to the worker
    default."""

    def test_resolve_timeout_precedence(
        self, fake_redis, worker_settings  # noqa: ARG002 — settings needed for ctor
    ) -> None:
        """``_resolve_timeout`` precedence (no executor call — pure resolution):
        per-run override wins; an explicit per-run 0 disables even when the worker
        default is set; None defers to the worker default; a worker-default 0 (the
        shipped default) is disabled; negatives are disabled."""
        queue = RunsQueue(fake_redis, worker_settings)
        store = RunStatusStore(fake_redis, worker_settings)

        async def noop(_j: RunJob, _p: Any = None) -> dict[str, Any]:
            return _ok_result()

        # Worker default is 0.0 (shipped off) — so a None override stays disabled.
        consumer_off = RunConsumer(queue, store, noop, worker_settings)
        assert consumer_off._resolve_timeout(RunJob(run_id="a", goal="g")) == 0.0

        # A set worker default is used when the job has no per-run override.
        s_on = worker_settings.model_copy(update={"run_timeout_s": 1800.0})
        consumer_on = RunConsumer(queue, store, noop, s_on)
        assert consumer_on._resolve_timeout(RunJob(run_id="b", goal="g")) == 1800.0

        # Per-run override wins over the worker default.
        assert (
            consumer_on._resolve_timeout(RunJob(run_id="c", goal="g", run_timeout_s=60.0))
            == 60.0
        )
        # An explicit per-run 0 disables EVEN WHEN the worker default is set.
        assert (
            consumer_on._resolve_timeout(RunJob(run_id="d", goal="g", run_timeout_s=0.0))
            == 0.0
        )
        # Negatives are disabled.
        assert (
            consumer_on._resolve_timeout(RunJob(run_id="e", goal="g", run_timeout_s=-5.0))
            == 0.0
        )

    async def test_run_timeout_marks_terminal_timeout(
        self, fake_redis, worker_settings
    ) -> None:
        """A (armed): an executor that outlasts ``run_timeout_s`` is bounded by
        ``asyncio.timeout`` → terminal TIMEOUT + acked (NOT redelivered, resumable
        via checkpoint). Per-run override is the API surface exercised here."""
        queue = RunsQueue(fake_redis, worker_settings)
        store = RunStatusStore(fake_redis, worker_settings)
        # Tiny per-run bound; the executor sleeps well past it.
        job = RunJob(run_id="slow", goal="g", run_timeout_s=0.05)

        async def slow(_j: RunJob, _p: Any = None) -> dict[str, Any]:
            await asyncio.sleep(5.0)  # far exceeds the 0.05s bound
            return _ok_result()

        consumer = RunConsumer(queue, store, slow, worker_settings)
        await queue.ensure_group()
        entry_id = await queue.enqueue(job)
        await queue.read_new()  # claim into the PEL so ack resolves

        assert await consumer._process(entry_id, job) is True  # terminal — acked
        assert await queue.pending_count() == 0
        rec = await store.get("slow")
        assert rec is not None
        assert rec.status is JobStatus.TIMEOUT
        assert "timeout after 0.05s" in (rec.error or "")

    async def test_worker_default_run_timeout_arms(
        self, fake_redis, worker_settings
    ) -> None:
        """A (worker default): with no per-run override, the worker-default
        ``run_timeout_s`` arms the bound (the common production path when the
        operator sets ``WORKER_RUN_TIMEOUT_S`` and jobs don't override it)."""
        s = worker_settings.model_copy(update={"run_timeout_s": 0.05})
        queue = RunsQueue(fake_redis, s)
        store = RunStatusStore(fake_redis, s)
        job = RunJob(run_id="wd", goal="g")  # no per-run override

        async def slow(_j: RunJob, _p: Any = None) -> dict[str, Any]:
            await asyncio.sleep(5.0)
            return _ok_result()

        consumer = RunConsumer(queue, store, slow, s)
        await queue.ensure_group()
        entry_id = await queue.enqueue(job)
        await queue.read_new()

        assert await consumer._process(entry_id, job) is True
        assert (await store.get("wd")).status is JobStatus.TIMEOUT  # type: ignore[union-attr]

    async def test_run_cancelled_marks_terminal_cancelled(
        self, fake_redis, worker_settings
    ) -> None:
        """E (cancel): execute_run polls the Redis cancel flag at the per-iteration
        progress callback and raises ``RunCancelled`` → terminal CANCELLED + acked
        (NOT redelivered). Here the fake executor raises it directly."""
        queue = RunsQueue(fake_redis, worker_settings)
        store = RunStatusStore(fake_redis, worker_settings)
        job = RunJob(run_id="can", goal="g")

        async def cancelled(_j: RunJob, _p: Any = None) -> dict[str, Any]:
            raise RunCancelled("cancelled via POST /runs/can/cancel")

        consumer = RunConsumer(queue, store, cancelled, worker_settings)
        await queue.ensure_group()
        entry_id = await queue.enqueue(job)
        await queue.read_new()

        assert await consumer._process(entry_id, job) is True  # terminal — acked
        assert await queue.pending_count() == 0
        rec = await store.get("can")
        assert rec is not None
        assert rec.status is JobStatus.CANCELLED
        assert "cancelled" in (rec.error or "")

    async def test_progress_callback_raises_cancel_when_flag_set(
        self, fake_redis, worker_settings
    ) -> None:
        """E (cancel checkpoint): the per-iteration progress callback the worker
        passes to execute_run polls the cancel flag and raises ``RunCancelled``
        → CANCELLED + acked. Unlike the direct-raise test above, this exercises
        the REAL flag-check chain — ``request_cancel`` → ``_report_progress``
        → ``is_cancelled`` → ``raise RunCancelled`` — by invoking the callback
        the way execute_run does (``await on_progress(ic)``)."""
        queue = RunsQueue(fake_redis, worker_settings)
        store = RunStatusStore(fake_redis, worker_settings)
        job = RunJob(run_id="can2", goal="g")

        # Pre-set the cancel flag before the run polls it.
        await store.request_cancel("can2")

        async def poller(_j: RunJob, progress: Any = None) -> dict[str, Any]:
            # Mimic execute_run's worker path: report iteration via the
            # callback (the worker's _report_progress closure) — which sees the
            # flag and raises RunCancelled before we reach the return.
            assert progress is not None
            await progress(1)
            return _ok_result()  # unreachable: progress raises first

        consumer = RunConsumer(queue, store, poller, worker_settings)
        await queue.ensure_group()
        entry_id = await queue.enqueue(job)
        await queue.read_new()

        assert await consumer._process(entry_id, job) is True  # terminal — acked
        assert await queue.pending_count() == 0
        rec = await store.get("can2")
        assert rec is not None
        assert rec.status is JobStatus.CANCELLED
        assert "cancel" in (rec.error or "").lower()

    async def test_progress_callback_continues_when_flag_unset(
        self, fake_redis, worker_settings
    ) -> None:
        """E (no-op when not cancelled): with no flag set, the progress callback
        reports iteration and returns normally — the run is NOT killed. Locks
        that adding the cancel checkpoint did not regress the happy path."""
        queue = RunsQueue(fake_redis, worker_settings)
        store = RunStatusStore(fake_redis, worker_settings)
        job = RunJob(run_id="ok", goal="g")

        seen: list[int] = []

        async def healthy(_j: RunJob, progress: Any = None) -> dict[str, Any]:
            assert progress is not None
            await progress(1)
            await progress(2)
            seen.extend([1, 2])
            return _ok_result()

        consumer = RunConsumer(queue, store, healthy, worker_settings)
        await queue.ensure_group()
        entry_id = await queue.enqueue(job)
        await queue.read_new()

        assert await consumer._process(entry_id, job) is True  # completed — acked
        rec = await store.get("ok")
        assert rec is not None
        assert rec.status is JobStatus.COMPLETED
        assert seen == [1, 2]

    async def test_budget_exhausted_marks_terminal_budget_exhausted(
        self, fake_redis, worker_settings
    ) -> None:
        """D (opt-in hard-stop): when ``BUDGET_HARD_STOP`` is on, the gateway raises
        ``BudgetExhaustedError`` instead of downgrading → terminal BUDGET_EXHAUSTED
        + acked (resumable via checkpoint; caveat: cumulative cap re-trips on
        resume — documented, deferred)."""
        queue = RunsQueue(fake_redis, worker_settings)
        store = RunStatusStore(fake_redis, worker_settings)
        job = RunJob(run_id="over", goal="g")

        async def over_budget(_j: RunJob, _p: Any = None) -> dict[str, Any]:
            raise BudgetExhaustedError("per-task token limit reached (hard_stop on)")

        consumer = RunConsumer(queue, store, over_budget, worker_settings)
        await queue.ensure_group()
        entry_id = await queue.enqueue(job)
        await queue.read_new()

        assert await consumer._process(entry_id, job) is True  # terminal — acked
        assert await queue.pending_count() == 0
        rec = await store.get("over")
        assert rec is not None
        assert rec.status is JobStatus.BUDGET_EXHAUSTED
        assert "token limit" in (rec.error or "")

    async def test_stray_timeout_when_unarmed_dead_letters(
        self, fake_redis, worker_settings
    ) -> None:
        """F (fallthrough guarantee): while the wall-clock bound is UNARMED
        (``run_timeout_s == 0``, the shipped default), a STRAY downstream
        ``TimeoutError`` (e.g. an httpx read timeout) must NOT be mislabeled
        TIMEOUT. ``_process`` re-raises it so it lands in the dead-letter path
        (FAILED + left for redelivery) — preserving the prior default behavior so
        the opt-in timeout can't change outcomes when off."""
        queue = RunsQueue(fake_redis, worker_settings)
        store = RunStatusStore(fake_redis, worker_settings)
        # Default worker_settings → run_timeout_s == 0 (unarmed).
        job = RunJob(run_id="stray", goal="g")

        async def stray_timeout(_j: RunJob, _p: Any = None) -> dict[str, Any]:
            # A downstream I/O read timeout — NOT our wall-clock bound.
            raise TimeoutError("httpx read timeout")

        consumer = RunConsumer(queue, store, stray_timeout, worker_settings)
        await queue.ensure_group()
        entry_id = await queue.enqueue(job)
        await queue.read_new()

        # NOT terminal: dead-letter (attempt 1 < cap 3) → FAILED + left for redelivery.
        assert await consumer._process(entry_id, job) is False
        assert await queue.pending_count() == 1
        rec = await store.get("stray")
        assert rec is not None
        # Decisive: it is FAILED (dead-letter), NOT mislabeled TIMEOUT.
        assert rec.status is JobStatus.FAILED
        assert rec.status is not JobStatus.TIMEOUT
        assert "read timeout" in (rec.error or "")


class TestRunConsumerWatchdog:
    """Phase 1 — out-of-band run-timeout watchdog (the q04 ~3h-stall fix).

    The prior in-band ``asyncio.timeout`` was defeated when a stalled run
    ABSORBED the cancellation somewhere in the execute_run↔gateway stack and
    never returned: the ``await self._executor(...)`` blocked forever, so no
    terminal mark fired and ``_renew_lease`` (created OUTSIDE the timeout ctx)
    kept the lease alive ~1h28m past the ceiling (only the Redis-side TTL ended
    it). The watchdog enforces a terminal TIMEOUT with plain Redis calls
    INDEPENDENT of whether the executor honors cancellation.

    The existing ``asyncio.sleep``-based timeout test (above) passes under BOTH
    the old and new code, and so cannot prove the fix: ``asyncio.sleep`` honors
    cancellation. A "swallow-once-then-return" executor also cannot —
    ``asyncio.timeout.__aexit__`` re-raises ``TimeoutError`` on expiry anyway.
    ONLY an executor that hangs INDEFINITELY distinguishes the out-of-band
    watchdog; that is the regression locked here.
    """

    async def test_watchdog_enforces_timeout_when_executor_absorbs_cancel(
        self, fake_redis, worker_settings, monkeypatch
    ) -> None:
        """A stalled run that swallows every ``CancelledError`` and never
        finishes still terminates at the deadline: the watchdog marks TIMEOUT +
        acks + releases the lease out-of-band, and ``_process`` returns True
        within ``deadline + grace`` — NOT the absorbed executor's hang. The old
        in-band code blocked forever in ``await self._executor(...)`` here."""
        import src.worker.runner as runner_mod

        # Shrink the abandon grace so the truly-stuck executor case is fast.
        monkeypatch.setattr(runner_mod, "_EXEC_CANCEL_GRACE_S", 0.1)

        queue = RunsQueue(fake_redis, worker_settings)
        store = RunStatusStore(fake_redis, worker_settings)
        job = RunJob(run_id="hang", goal="g", run_timeout_s=0.05)

        # The leaked background executor (it absorbs cancellation) is reaped via
        # ``abandon`` after the assertions so the event loop is clean on teardown.
        abandon = asyncio.Event()
        cancels: list[int] = []
        hung: list[asyncio.Task[Any]] = []

        async def absorbing(_j: RunJob, _p: Any = None) -> dict[str, Any]:
            # Absorb cancellation indefinitely — the litellm/gateway stall analog.
            # Never finishes on its own; only the out-of-band watchdog ends the RUN.
            hung.append(asyncio.current_task())  # type: ignore[arg-type]
            while not abandon.is_set():
                try:
                    await asyncio.sleep(0.5)
                except asyncio.CancelledError:
                    cancels.append(1)
                    continue  # swallow → keep hanging
            return {"final_output": "absorbed", "is_complete": False, "iteration_count": 0}

        consumer = RunConsumer(queue, store, absorbing, worker_settings)
        await queue.ensure_group()
        entry_id = await queue.enqueue(job)
        await queue.read_new()  # claim into the PEL so the watchdog's ack resolves

        loop = asyncio.get_running_loop()
        start = loop.time()
        acked = await consumer._process(entry_id, job)
        elapsed = loop.time() - start

        # Terminal + acked even though the executor never finished (it swallowed
        # every cancellation) — the watchdog did the mark/ack/release out-of-band.
        assert acked is True
        assert await queue.pending_count() == 0
        rec = await store.get("hang")
        assert rec is not None
        assert rec.status is JobStatus.TIMEOUT
        assert "timeout after 0.05s" in (rec.error or "")
        # Bounded by the deadline + the shrunk grace, NOT the executor's hang.
        assert elapsed < 2.0
        assert cancels  # exec_task.cancel() was delivered at least once (absorbed)
        # The watchdog cancelled the renewal + released the lease → a fresh claim
        # can acquire the per-run lock (peer takeover is unblocked, not held for
        # ~1h28m past the ceiling by a zombie renewer as in the old code).
        assert await queue.try_lock(job.run_id, "peer", worker_settings.lock_ttl_s) is True

        # Reap the leaked background executor: set the gate so its next loop
        # iteration exits, then wake its sleep with a cancel. ``asyncio.wait``
        # (not ``wait_for``/bare ``await``) bounds the reap so an absorbing
        # executor can never hang the test teardown.
        abandon.set()
        if hung and not hung[0].done():
            hung[0].cancel()
            await asyncio.wait({hung[0]}, timeout=1.0)

    async def test_armed_watchdog_does_not_fire_on_fast_completion(
        self, fake_redis, worker_settings
    ) -> None:
        """An ARMED wall-clock bound must NOT produce a false TIMEOUT on a healthy
        fast run: when ``exec_task`` finishes before the deadline, the watchdog is
        cancelled and the normal COMPLETED path runs. Locks that the watchdog is a
        pure addition — no behavior change when the run completes in time."""
        s = worker_settings.model_copy(update={"run_timeout_s": 5.0})
        queue = RunsQueue(fake_redis, s)
        store = RunStatusStore(fake_redis, s)
        job = RunJob(run_id="fast", goal="g")  # no per-run override → 5s bound

        async def fast(_j: RunJob, _p: Any = None) -> dict[str, Any]:
            return _ok_result()

        consumer = RunConsumer(queue, store, fast, s)
        await queue.ensure_group()
        entry_id = await queue.enqueue(job)
        await queue.read_new()

        assert await consumer._process(entry_id, job) is True  # completed — acked
        assert await queue.pending_count() == 0
        rec = await store.get("fast")
        assert rec is not None
        assert rec.status is JobStatus.COMPLETED  # NOT a false TIMEOUT
        assert rec.final_output == "answer"

