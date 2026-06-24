"""Unit tests for the per-run Redis status store (Phase 2b).

Covers serialize/deserialize fidelity (incl. bytes from a non-decode-responses
client), the ``get`` miss path, and the ``mark`` transition timestamping
(started_at on RUNNING, finished_at on COMPLETED/FAILED, started_at preserved
across the queued→running→completed lifecycle).
"""

from __future__ import annotations

from src.worker.schema import JobStatus, RunStatus
from src.worker.status import RunStatusStore


class TestRunStatusHashing:
    def test_to_hash_from_hash_roundtrip(self) -> None:
        """serialize/deserialize is lossless for the typed fields."""
        original = RunStatus(
            run_id="r1",
            thread_id="api-r1",
            status=JobStatus.RUNNING,
            is_complete=False,
            iteration_count=7,
            final_output="done",
        )
        rebuilt = RunStatus.from_hash(original.to_hash())
        assert rebuilt.run_id == "r1"
        assert rebuilt.status is JobStatus.RUNNING
        assert rebuilt.iteration_count == 7
        assert rebuilt.is_complete is False
        assert rebuilt.final_output == "done"

    def test_from_hash_decodes_bytes(self) -> None:
        """Redis without decode_responses returns bytes — from_hash handles it."""
        rebuilt = RunStatus.from_hash(
            {
                b"run_id": b"r1",
                b"thread_id": b"api-r1",
                b"status": b"completed",
                b"is_complete": b"True",
                b"iteration_count": b"3",
            }
        )
        assert rebuilt.run_id == "r1"
        assert rebuilt.status is JobStatus.COMPLETED
        assert rebuilt.is_complete is True
        assert rebuilt.iteration_count == 3

    def test_from_hash_coerces_bad_iteration_count(self) -> None:
        """A non-numeric iteration_count falls back to 0, not a crash."""
        rebuilt = RunStatus.from_hash(
            {"run_id": "r1", "thread_id": "api-r1", "iteration_count": "oops"}
        )
        assert rebuilt.iteration_count == 0

    def test_from_hash_empty_mapping_raises(self) -> None:
        import pytest

        with pytest.raises(KeyError):
            RunStatus.from_hash({})


class TestRunStatusStore:
    async def test_put_then_get_roundtrip(
        self, fake_redis, worker_settings
    ) -> None:
        store = RunStatusStore(fake_redis, worker_settings)
        status = RunStatus(
            run_id="r1", thread_id="api-r1", status=JobStatus.RUNNING
        )
        await store.put(status)
        fetched = await store.get("r1")
        assert fetched is not None
        assert fetched.run_id == "r1"
        assert fetched.status is JobStatus.RUNNING

    async def test_get_returns_none_when_missing(
        self, fake_redis, worker_settings
    ) -> None:
        store = RunStatusStore(fake_redis, worker_settings)
        assert await store.get("nope") is None

    async def test_mark_stamps_started_at_on_running(
        self, fake_redis, worker_settings
    ) -> None:
        store = RunStatusStore(fake_redis, worker_settings)
        rec = await store.mark("r1", "api-r1", JobStatus.RUNNING)
        assert rec.started_at != ""
        assert rec.status is JobStatus.RUNNING

    async def test_mark_stamps_finished_at_on_completed(
        self, fake_redis, worker_settings
    ) -> None:
        store = RunStatusStore(fake_redis, worker_settings)
        await store.mark("r1", "api-r1", JobStatus.RUNNING)
        rec = await store.mark(
            "r1", "api-r1", JobStatus.COMPLETED, final_output="ok"
        )
        assert rec.finished_at != ""
        assert rec.status is JobStatus.COMPLETED
        assert rec.final_output == "ok"

    async def test_mark_failed_sets_error(
        self, fake_redis, worker_settings
    ) -> None:
        store = RunStatusStore(fake_redis, worker_settings)
        rec = await store.mark("r1", "api-r1", JobStatus.FAILED, error="boom")
        assert rec.status is JobStatus.FAILED
        assert rec.error == "boom"
        assert rec.finished_at != ""

    async def test_mark_preserves_started_across_lifecycle(
        self, fake_redis, worker_settings
    ) -> None:
        """started_at stamped at RUNNING survives the COMPLETED transition."""
        store = RunStatusStore(fake_redis, worker_settings)
        running = await store.mark("r1", "api-r1", JobStatus.RUNNING)
        completed = await store.mark("r1", "api-r1", JobStatus.COMPLETED)
        assert completed.started_at == running.started_at

    async def test_mark_merges_extra_fields(
        self, fake_redis, worker_settings
    ) -> None:
        """final_output / is_complete / iteration_count merge on completion."""
        store = RunStatusStore(fake_redis, worker_settings)
        rec = await store.mark(
            "r1",
            "api-r1",
            JobStatus.COMPLETED,
            final_output="answer",
            is_complete=True,
            iteration_count=4,
        )
        assert rec.final_output == "answer"
        assert rec.is_complete is True
        assert rec.iteration_count == 4


class _ExplodingRedis:
    """Stub whose set/exists raise — proves request_cancel/is_cancelled are
    best-effort (never raise / fail-open) on a Redis error."""

    async def set(self, *_a: object, **_kw: object) -> None:
        raise RuntimeError("redis down")

    async def exists(self, *_a: object, **_kw: object) -> int:
        raise RuntimeError("redis down")


class TestRunStatusStoreCancelFlag:
    async def test_request_cancel_then_is_cancelled(
        self, fake_redis, worker_settings
    ) -> None:
        """Setting the flag flips is_cancelled from False to True."""
        store = RunStatusStore(fake_redis, worker_settings)
        assert await store.is_cancelled("r1") is False
        await store.request_cancel("r1")
        assert await store.is_cancelled("r1") is True

    async def test_request_cancel_idempotent(
        self, fake_redis, worker_settings
    ) -> None:
        """A repeat POST is a no-op — the flag's presence IS the signal."""
        store = RunStatusStore(fake_redis, worker_settings)
        await store.request_cancel("r1")
        await store.request_cancel("r1")
        assert await store.is_cancelled("r1") is True

    async def test_is_cancelled_false_for_unrelated_run(
        self, fake_redis, worker_settings
    ) -> None:
        """The flag is per-run_id — cancelling r1 does not cancel r2."""
        store = RunStatusStore(fake_redis, worker_settings)
        await store.request_cancel("r1")
        assert await store.is_cancelled("r2") is False

    async def test_cancel_flag_uses_separate_key_from_status(
        self, fake_redis, worker_settings
    ) -> None:
        """The cancel flag must not collide with the status hash namespace."""
        store = RunStatusStore(fake_redis, worker_settings)
        await store.put(
            RunStatus(run_id="r1", thread_id="api-r1", status=JobStatus.RUNNING)
        )
        await store.request_cancel("r1")
        # A status record WITHOUT a cancel flag still reads as not-cancelled.
        assert await store.is_cancelled("r1") is True
        # And a cancelled run with no status record still reads cancelled.
        await store.request_cancel("solo")
        assert await store.is_cancelled("solo") is True
        assert await store.get("solo") is None

    async def test_is_cancelled_fails_open_on_redis_error(
        self, worker_settings
    ) -> None:
        """On a Redis error is_cancelled returns False (run continues) — cancel
        is cooperative/best-effort; the run-timeout is the hard bound."""
        store = RunStatusStore(_ExplodingRedis(), worker_settings)
        assert await store.is_cancelled("r1") is False

    async def test_request_cancel_never_raises_on_redis_error(
        self, worker_settings
    ) -> None:
        """request_cancel is best-effort — a Redis error must not propagate."""
        store = RunStatusStore(_ExplodingRedis(), worker_settings)
        await store.request_cancel("r1")  # no raise
