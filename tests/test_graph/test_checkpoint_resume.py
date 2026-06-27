"""Checkpoint-resume contract — thread_id derivation + the resume handle.

Companion to ``tests/test_graph/test_checkpoint.py`` (which pins the
``create_checkpointer``/``close_checkpointer`` factory lifecycle: direct
connection, setup-failure-closes, no-leak). This file covers the DISTINCT
resume-contract angles called out by the design, exercising PURE LOGIC against
an in-memory fake checkpointer (no live Postgres):

* ``thread_id = f"api-{run_id}"`` is derived from the run_id, NOT the pid — so a
  redelivered/restarted worker resumes the SAME checkpoint thread;
* a checkpointer round-trips state (put → get returns it);
* ``resume`` replays from the last checkpoint (the config carries the same
  thread_id the put used);
* ``resume`` refuses cleanly (RuntimeError) when no checkpoint exists, and
  raises ValueError when ``--resume`` is given without a ``--run-id``;
* the worker's ``thread_id_for`` and the API route's literal agree on
  ``api-{run_id}`` (the cross-process resume invariant).
"""

from __future__ import annotations

from typing import Any

import pytest

from src.graph.checkpoint import close_checkpointer


# ─── thread_id derivation (run_id-keyed, NOT pid-keyed) ─────────────


class TestThreadIdDerivation:
    """``thread_id = f"api-{run_id}"`` — keyed on the explicit run_id."""

    def test_thread_id_derived_from_run_id(self) -> None:
        from src.worker.runner import RunConsumer

        assert RunConsumer.thread_id_for("q09-20260624") == "api-q09-20260624"

    def test_thread_id_stable_across_redelivery(self) -> None:
        """The same run_id yields the same thread_id on every call — the resume
        invariant a redelivered/restarted worker relies on."""
        from src.worker.runner import RunConsumer

        first = RunConsumer.thread_id_for("abc-123")
        second = RunConsumer.thread_id_for("abc-123")
        assert first == second == "api-abc-123"

    def test_thread_id_not_pid_keyed(self) -> None:
        """Two distinct run_ids produce distinct threads (a pid-keyed scheme would
        collide a re-enqueued run onto a different process's thread)."""
        from src.worker.runner import RunConsumer

        a = RunConsumer.thread_id_for("run-A")
        b = RunConsumer.thread_id_for("run-B")
        assert a != b
        assert a.startswith("api-") and b.startswith("api-")

    def test_api_route_convention_matches_worker(self) -> None:
        """The API route that ENQUEUES the run and the worker that RESUMES it must
        agree on the literal ``api-{run_id}`` (else resume reads the wrong thread)."""
        # The API route constructs it inline:
        run_id = "vector-db-7"
        api_thread_id = f"api-{run_id}"
        from src.worker.runner import RunConsumer

        assert api_thread_id == RunConsumer.thread_id_for(run_id)


# ─── In-memory fake checkpointer (put → get round-trip) ─────────────


class _InMemoryCheckpointer:
    """A minimal fake of ``AsyncPostgresSaver`` keyed on thread_id.

    Models ONLY the resume contract the LangGraph runtime exercises: ``put`` to
    write a checkpoint, ``aget_tuple`` to read the latest one for a thread. No
    Postgres, no setup — pure dict storage.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}
        self.conn: _FakeConn = _FakeConn()

    async def setup(self) -> None:
        return None

    async def put(self, config: dict[str, Any], checkpoint: dict[str, Any], metadata: dict[str, Any]) -> None:
        thread_id = config["configurable"]["thread_id"]
        self._store[thread_id] = {
            "checkpoint": checkpoint,
            "metadata": metadata,
        }

    async def aget_tuple(self, config: dict[str, Any]) -> dict[str, Any] | None:
        thread_id = config["configurable"]["thread_id"]
        return self._store.get(thread_id)


class _FakeConn:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_checkpointer() -> _InMemoryCheckpointer:
    return _InMemoryCheckpointer()


class TestCheckpointerRoundTrip:
    """A checkpointer round-trips state: put → get returns it (same thread)."""

    @pytest.mark.asyncio
    async def test_put_then_get_returns_state(self, fake_checkpointer: _InMemoryCheckpointer) -> None:
        thread_id = "api-roundtrip-1"
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint = {"iteration_count": 4, "current_step_index": 2, "errors": []}
        metadata = {"source": "loop", "step": "reflect"}

        await fake_checkpointer.put(config, checkpoint, metadata)
        result = await fake_checkpointer.aget_tuple(config)

        assert result is not None
        assert result["checkpoint"] == checkpoint
        assert result["metadata"]["step"] == "reflect"

    @pytest.mark.asyncio
    async def test_get_missing_thread_returns_none(self, fake_checkpointer: _InMemoryCheckpointer) -> None:
        """Reading a thread that was never written returns None — the resume
        refusal's trigger condition."""
        config = {"configurable": {"thread_id": "api-never-written"}}
        assert await fake_checkpointer.aget_tuple(config) is None

    @pytest.mark.asyncio
    async def test_threads_are_isolated(self, fake_checkpointer: _InMemoryCheckpointer) -> None:
        """Two runs write to two distinct threads; reading one does not leak the other."""
        cfg_a = {"configurable": {"thread_id": "api-runA"}}
        cfg_b = {"configurable": {"thread_id": "api-runB"}}
        await fake_checkpointer.put(cfg_a, {"v": "A"}, {})
        await fake_checkpointer.put(cfg_b, {"v": "B"}, {})

        got_a = await fake_checkpointer.aget_tuple(cfg_a)
        got_b = await fake_checkpointer.aget_tuple(cfg_b)
        assert got_a["checkpoint"]["v"] == "A"
        assert got_b["checkpoint"]["v"] == "B"


# ─── The resume contract (src/runner.resume_run) ─────────────────────


class TestResumeContract:
    """``_require_resumable_checkpoint`` replays from the last checkpoint OR
    refuses cleanly."""

    @pytest.mark.asyncio
    async def test_resume_replays_from_last_checkpoint(
        self, fake_checkpointer: _InMemoryCheckpointer
    ) -> None:
        """A checkpointed run resumes from its persisted state: the resume lookup
        uses the SAME thread_id the put used, and returns the stored tuple."""
        from src.runner import _require_resumable_checkpoint

        run_id = "resumable-1"
        thread_id = f"api-{run_id}"
        # Seed a checkpoint as if a prior attempt had persisted mid-run.
        await fake_checkpointer.put(
            {"configurable": {"thread_id": thread_id}},
            {"iteration_count": 7, "current_step_index": 3},
            {"source": "loop"},
        )

        existing = await _require_resumable_checkpoint(
            checkpointer=fake_checkpointer, thread_id=thread_id, run_id=run_id
        )
        assert existing is not None
        assert existing["checkpoint"]["iteration_count"] == 7

    @pytest.mark.asyncio
    async def test_resume_refuses_when_no_checkpoint(
        self, fake_checkpointer: _InMemoryCheckpointer
    ) -> None:
        """Resuming a run that never persisted raises RuntimeError (clean refuse),
        not a silent no-op."""
        from src.runner import _require_resumable_checkpoint

        run_id = "never-started"
        thread_id = f"api-{run_id}"
        with pytest.raises(RuntimeError, match="no checkpoint found"):
            await _require_resumable_checkpoint(
                checkpointer=fake_checkpointer, thread_id=thread_id, run_id=run_id
            )

    @pytest.mark.asyncio
    async def test_resume_refuses_without_run_id(
        self, fake_checkpointer: _InMemoryCheckpointer
    ) -> None:
        """``--resume`` without ``--run-id`` raises ValueError."""
        from src.runner import _require_resumable_checkpoint

        with pytest.raises(ValueError, match="--resume requires --run-id"):
            await _require_resumable_checkpoint(
                checkpointer=fake_checkpointer, thread_id="api-something", run_id=None
            )

    @pytest.mark.asyncio
    async def test_resume_refuses_without_checkpointer(self) -> None:
        """Resuming with no checkpointer (PostgreSQL unreachable) raises RuntimeError."""
        from src.runner import _require_resumable_checkpoint

        with pytest.raises(RuntimeError, match="no checkpointer available"):
            await _require_resumable_checkpoint(
                checkpointer=None, thread_id="api-x", run_id="x"
            )

    @pytest.mark.asyncio
    async def test_resume_uses_run_id_keyed_thread_not_pid(
        self, fake_checkpointer: _InMemoryCheckpointer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even if the process pid differs between the original run and the resume,
        the lookup finds the checkpoint because the thread is keyed on run_id."""
        from src.runner import _require_resumable_checkpoint

        run_id = "pid-stable"
        thread_id = f"api-{run_id}"
        await fake_checkpointer.put(
            {"configurable": {"thread_id": thread_id}},
            {"stamped": True},
            {},
        )
        # Simulate a different process: pid changes, but run_id (hence thread_id) is the same.
        monkeypatch.setattr("os.getpid", lambda: 99999)

        existing = await _require_resumable_checkpoint(
            checkpointer=fake_checkpointer, thread_id=thread_id, run_id=run_id
        )
        assert existing is not None
        assert existing["checkpoint"]["stamped"] is True


# ─── close_checkpointer releases the connection (resume cleanup) ─────


class TestResumeCheckpointerLifecycle:
    """The resume path's checkpointer is released via close_checkpointer."""

    @pytest.mark.asyncio
    async def test_close_releases_connection(self, fake_checkpointer: _InMemoryCheckpointer) -> None:
        """Closing a fake checkpointer awaits conn.close() (no leak across resumes)."""
        await close_checkpointer(fake_checkpointer)
        assert fake_checkpointer.conn.closed is True

    @pytest.mark.asyncio
    async def test_close_none_is_noop(self) -> None:
        """A resume that never built a checkpointer closes cleanly."""
        await close_checkpointer(None)  # must not raise
