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


class TestAckAndDelete:
    """``ack_and_delete`` — the terminal-completion primitive the worker's
    success/terminal paths call (complex-arxiv-stats-3 regression).

    ``ack`` alone removes an entry from the pending list but LEAVES it in the
    stream body, so every completed run's entry accumulated in XRANGE
    indefinitely (one had to be purged manually). ``ack_and_delete`` XACKs
    then XDELs so a terminal run leaves no lingering entry — while still
    returning the ``XACK`` count (matching ``ack``) so the worker's
    ``acked_total`` counter stays accurate. Distinct from ``delete_entry``
    (cancel-path: single-id, bool, swallows even the XACK failure)."""

    async def test_ack_and_delete_removes_entry_entirely(
        self, fake_redis, worker_settings
    ) -> None:
        """Terminal completion drops the entry from BOTH the pending list
        (``pending_count``) and the stream body (``xlen``) — the regression:
        ``ack``-only leaves the entry lingering at xlen==1."""
        q = RunsQueue(fake_redis, worker_settings)
        await q.ensure_group()
        await q.enqueue(RunJob(run_id="r1", goal="g"))
        entries = await q.read_new()
        entry_id = entries[0][0]
        assert await q.pending_count() == 1

        assert await q.ack_and_delete([entry_id]) == 1
        assert await q.pending_count() == 0
        # XDEL proof: the stream body is empty (ack-only would leave xlen==1).
        assert int(await fake_redis.xlen(worker_settings.runs_stream)) == 0

    async def test_ack_and_delete_empty_or_blank_returns_zero(
        self, fake_redis, worker_settings
    ) -> None:
        """No real ids → 0 (mirrors ack's filter)."""
        q = RunsQueue(fake_redis, worker_settings)
        await q.ensure_group()
        assert await q.ack_and_delete([]) == 0
        assert await q.ack_and_delete([""]) == 0  # blank ids are filtered

    async def test_ack_and_delete_already_acked_returns_zero(
        self, fake_redis, worker_settings
    ) -> None:
        """A second ack_and_delete on an entry already removed from the
        pending list returns 0 (XACK count) and does not raise — terminal
        removal is idempotent (cancel's delete_entry may have raced ahead)."""
        q = RunsQueue(fake_redis, worker_settings)
        await q.ensure_group()
        await q.enqueue(RunJob(run_id="r1", goal="g"))
        entries = await q.read_new()
        entry_id = entries[0][0]
        assert await q.ack_and_delete([entry_id]) == 1
        # Second call: entry already acked+deleted → XACK returns 0, no raise.
        assert await q.ack_and_delete([entry_id]) == 0


class TestDeleteEntry:
    """``delete_entry`` — the terminal-removal primitive cancel calls (P1) so a
    cancelled run's pending entry can never be reclaimed by a peer worker.

    Cancel must make the entry vanish the instant the flag is set, not merely
    when the in-flight worker gets around to acking: ``reclaim_stale``
    (XAUTOCLAIM) re-hands out any entry still in the group's pending-entries
    list, which would resume the run from its checkpoint and burn tokens while
    the owner cooperatively winds down. ``XACK`` drops it from the PEL;
    ``XDEL`` drops it from the stream body so cancelled runs don't accumulate.
    """

    async def test_delete_entry_acks_and_deletes(
        self, fake_redis, worker_settings
    ) -> None:
        """After delete_entry the entry is gone from BOTH the pending list
        (``pending_count``) and the stream body (``xlen``) — distinguishing it
        from a plain ``ack`` (which leaves the entry in the stream)."""
        q = RunsQueue(fake_redis, worker_settings)
        await q.ensure_group()
        await q.enqueue(RunJob(run_id="r1", goal="g"))
        claimed = await q.read_new()
        assert await q.pending_count() == 1
        entry_id = claimed[0][0]

        assert await q.delete_entry(entry_id) is True
        assert await q.pending_count() == 0
        # XDEL proof: the stream length is 0 (ack-only would leave it at 1).
        assert int(await fake_redis.xlen(worker_settings.runs_stream)) == 0

    async def test_delete_entry_defeats_reclaim(
        self, fake_redis, worker_settings
    ) -> None:
        """The regression for the respawn bug: a claimed-but-unacked entry,
        once delete_entry'd, is NOT handed to a peer by ``reclaim_stale``
        (``reclaim_min_idle_ms=0`` here, so a still-present entry WOULD be
        reclaimed). Without the fix this returns the entry → respawn."""
        settings_a = worker_settings.model_copy(update={"consumer_name": "wa"})
        settings_b = worker_settings.model_copy(update={"consumer_name": "wb"})
        q_a = RunsQueue(fake_redis, settings_a)
        q_b = RunsQueue(fake_redis, settings_b)
        await q_a.ensure_group()
        await q_a.enqueue(RunJob(run_id="x", goal="g"))

        claimed = await q_a.read_new()  # worker-a claims; simulates cancel BEFORE ack
        assert len(claimed) == 1
        entry_id = claimed[0][0]

        # Cancel path: delete the entry the worker is still nominally "holding".
        assert await q_a.delete_entry(entry_id) is True

        reclaimed = await q_b.reclaim_stale()  # would-be respawn
        assert reclaimed == []

    async def test_delete_entry_blank_returns_false(
        self, fake_redis, worker_settings
    ) -> None:
        """A blank entry id is a no-op False (cancel when no id was recorded)."""
        q = RunsQueue(fake_redis, worker_settings)
        await q.ensure_group()
        assert await q.delete_entry("") is False

    async def test_delete_entry_unknown_id_is_harmless(
        self, fake_redis, worker_settings
    ) -> None:
        """XACK/XDEL on an id that isn't pending is a 0-return no-op (no raise) —
        cancel racing with the worker's own terminal ack must not error."""
        q = RunsQueue(fake_redis, worker_settings)
        await q.ensure_group()
        assert await q.delete_entry("9999-0") is True


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


class TestRecordAttempt:
    """``record_attempt`` — the per-run delivery counter that drives the dead-letter
    cap (Bug B). Keyed by ``run_id`` (stable across XAUTOCLAIM redelivery)."""

    async def test_increments_and_returns_count(
        self, fake_redis, worker_settings
    ) -> None:
        """Each call increments a per-run counter; the count gates dead-lettering."""
        q = RunsQueue(fake_redis, worker_settings)
        assert await q.record_attempt("r1") == 1
        assert await q.record_attempt("r1") == 2
        assert await q.record_attempt("r1") == 3
        # distinct run_ids keep independent counters
        assert await q.record_attempt("r2") == 1
        assert await q.record_attempt("r1") == 4  # r1 unaffected by r2's calls

    async def test_persists_with_zero_ttl(self, fake_redis, worker_settings) -> None:
        """status_ttl_s=0 means "no TTL" (the test convention): the counter must NOT
        be deleted by an EXPIRE 0, so the dead-letter cap can actually be reached.
        Guards the ``status_ttl_s > 0`` guard in record_attempt (a bare
        ``expire(key, 0)`` would reset the counter to 1 on every call → cap never
        hit → the infinite poison loop would survive the "fix")."""
        assert worker_settings.status_ttl_s == 0
        q = RunsQueue(fake_redis, worker_settings)
        await q.record_attempt("r0")
        assert await q.record_attempt("r0") == 2  # survived — no self-delete


class TestRunsQueueLeaseLock:
    """Bug C guard: the per-run lease lock primitives themselves (SET-NX acquire,
    compare-and-del release, compare-and-expire renew). The consumer behavior
    (skip a run another worker holds) is in test_runner.py::TestRunConsumerLeaseLock.
    All hermetic against fakeredis — the WATCH/MULTI compare-and-set runs against
    the real redis.asyncio Pipeline class fakeredis reuses."""

    async def test_try_lock_is_exclusive(
        self, fake_redis, worker_settings
    ) -> None:
        """Only the first ``SET … NX EX`` wins; a second caller (another worker)
        is refused — the atomic guard against concurrent double-processing."""
        q = RunsQueue(fake_redis, worker_settings)
        assert await q.try_lock("run-1", "token-a", 60) is True
        assert await q.try_lock("run-1", "token-b", 60) is False

    async def test_release_requires_owner_token(
        self, fake_redis, worker_settings
    ) -> None:
        """A release with the WRONG token must NOT free the lock (a stale holder
        whose TTL expired cannot evict a fresh owner); only the owner's token
        frees it for the next claimant."""
        q = RunsQueue(fake_redis, worker_settings)
        await q.try_lock("run-1", "owner", 60)
        assert await q.release_lock("run-1", "wrong") is False  # no-op
        assert await q.try_lock("run-1", "other", 60) is False  # still held
        assert await q.release_lock("run-1", "owner") is True  # owner frees it
        assert await q.try_lock("run-1", "other", 60) is True  # now claimable

    async def test_release_missing_lock_is_false(
        self, fake_redis, worker_settings
    ) -> None:
        """Releasing a lock nobody holds is False (GET → None ≠ token)."""
        q = RunsQueue(fake_redis, worker_settings)
        assert await q.release_lock("never-held", "any-token") is False

    async def test_renew_extends_ttl_for_owner_only(
        self, fake_redis, worker_settings
    ) -> None:
        """Renew returns True (extends TTL) for the owner, False for a non-owner."""
        q = RunsQueue(fake_redis, worker_settings)
        await q.try_lock("run-1", "owner", 2)
        assert await q.renew_lock("run-1", "owner", 60) is True
        assert await q.renew_lock("run-1", "wrong", 60) is False

    async def test_lock_key_is_stream_scoped(
        self, fake_redis, worker_settings
    ) -> None:
        """Distinct run_ids (and the per-stream prefix) never collide — one run's
        lease does not block another."""
        q = RunsQueue(fake_redis, worker_settings)
        assert q.lock_key("a") == f"{worker_settings.runs_stream}:lock:a"
        assert await q.try_lock("run-a", "ta", 60) is True
        assert await q.try_lock("run-b", "tb", 60) is True

    async def test_ttl_zero_disables_lease(
        self, fake_redis, worker_settings
    ) -> None:
        """``lock_ttl_s <= 0`` disables the lease (legacy single-worker mode):
        try_lock/renew_lock short-circuit True and are NOT exclusive."""
        s = worker_settings.model_copy(update={"lock_ttl_s": 0})
        q = RunsQueue(fake_redis, s)
        assert await q.try_lock("run-1", "a", 0) is True
        assert await q.try_lock("run-1", "b", 0) is True  # not exclusive (disabled)
        assert await q.renew_lock("run-1", "a", 0) is True
