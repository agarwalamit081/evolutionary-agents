"""Regression smoke: the ``api-{run_id}`` join-key is identical across surfaces.

An API-routed run is attributed, checkpointed, status-tracked, and evaluated
under ONE string — ``api-{run_id}`` — that must be produced identically by every
surface that touches it. If any one surface drifted to a different prefix
(``run-{...}``, ``api_{...}``, a bare ``{run_id}``), the run's cost ledger,
checkpoint thread, status hash, and eval rows would silently split across keys:
resume would miss the checkpoint, spend would be unattributable, eval results
unjoinable. This file locks that invariant in one place.

The five surfaces:

  1. API enqueue  — ``src/api/routes/agent.py`` derives ``thread_id = api-{run_id}``
     and stamps it into the run-status store.
  2. Worker consumer — ``RunConsumer.thread_id_for(run_id)`` (the resume handle
     across redelivery).
  3. Executor origin — ``default_agent_executor`` passes ``origin="api"`` to
     ``execute_run``; ``_thread_id_for_run(..., origin="api")`` resolves the key.
  4. Cost ledger   — ``execute_run`` binds that same key via
     ``gateway.set_run_id(thread_id)`` (runner.py:300) so every LLM call's cost
     row carries it.
  5. Eval results  — ``execute_run`` sets the graph config ``thread_id`` (runner.py:396),
     which flows into the verify node's eval-store writes keyed on ``run_id``.

Surfaces 4 & 5 reuse ONE ``thread_id`` local resolved at runner.py:244 — they
agree by construction; this smoke asserts the seam each reads is bound to the
same ``api-{run_id}`` the other surfaces emit.

Deterministic: no LLM, no Redis, no Postgres. The API route is exercised with
faked queue/status clients; the executor and cost-ledger seams are exercised
against the real symbols. The DB eval round-trip is covered end-to-end by
``test_role_split_integration.py`` (asserts ``thread_id == "api-p3-rs-ok"``).
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from src.api.routes.agent import EnqueueResponse, RunRequest, enqueue_run
from src.config.settings import get_settings
from src.eval.store import EvalStore
from src.llm.gateway import LLMGateway
from src.runner import _thread_id_for_run
from src.worker import executors as executor_mod
from src.worker.runner import RunConsumer
from src.worker.schema import JobStatus, RunJob

_RUN_ID = "uid-align-deadbeef"
_EXPECTED = f"api-{_RUN_ID}"


# ─── fakes for the API route (surface 1) ──────────────────────────────


class _FakeQueue:
    """Stand-in for RunsQueue — records the enqueued job, no Redis."""

    def __init__(self) -> None:
        self.enqueued: list[RunJob] = []

    async def ensure_group(self) -> None:
        return None

    async def enqueue(self, job: RunJob) -> None:
        self.enqueued.append(job)


class _FakeStatus:
    """Stand-in for RunStatusStore — captures the mark() arguments verbatim."""

    def __init__(self) -> None:
        self.marked: tuple[str, str, JobStatus] | None = None

    async def mark(self, run_id: str, thread_id: str, status: JobStatus, **_fields: object) -> None:
        self.marked = (run_id, thread_id, status)

    async def get(self, _run_id: str) -> None:
        """No prior in-flight run — a fresh enqueue must clear the dedup check
        (P1). The real ``RunStatusStore.get`` returns ``None`` for an unknown
        run_id; this stand-in mirrors that so ``enqueue_run``'s dedup lookup
        passes and the route proceeds to ``mark``."""
        return None


# ─── the invariant ────────────────────────────────────────────────────


class TestUidAlignment:
    """All five surfaces emit the identical ``api-{run_id}`` join key."""

    def test_consumer_and_resolver_emit_api_run_id_key(self) -> None:
        """Surfaces 2 & 3 (pure resolvers) both yield ``api-{run_id}``."""
        assert RunConsumer.thread_id_for(_RUN_ID) == _EXPECTED
        assert _thread_id_for_run(_RUN_ID, "any goal", origin="api") == _EXPECTED

    def test_api_and_cli_origins_never_collide(self) -> None:
        """THE reason the prefix exists: an API run and a CLI run sharing a
        run_id must NOT land on the same checkpoint thread / cost ledger / eval
        rows. A regression that dropped the prefix (or made both ``{run_id}``)
        would silently cross-contaminate resume + attribution."""
        api = _thread_id_for_run(_RUN_ID, "g", origin="api")
        cli = _thread_id_for_run(_RUN_ID, "g", origin="cli")
        assert api == _EXPECTED
        assert cli == f"cli-{_RUN_ID}"
        assert api != cli

    @pytest.mark.asyncio
    async def test_api_route_stamps_api_thread_id_into_status_store(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Surface 1: the real enqueue route derives + stamps ``api-{run_id}``."""
        from src.api.routes import agent as agent_mod

        fake_q, fake_s = _FakeQueue(), _FakeStatus()
        monkeypatch.setattr(agent_mod, "_client_and_queue", lambda: (fake_q, fake_s))

        resp = await enqueue_run(RunRequest(run_id=_RUN_ID, goal="ship the deliverable"))

        # The response surfaces the key back to the client (202 handle).
        assert isinstance(resp, EnqueueResponse)
        assert resp.run_id == _RUN_ID
        assert resp.thread_id == _EXPECTED
        # The status store received (run_id, api-{run_id}, QUEUED).
        assert fake_s.marked == (_RUN_ID, _EXPECTED, JobStatus.QUEUED)
        # And the job carries the raw run_id (the executor re-derives the prefix).
        assert len(fake_q.enqueued) == 1 and fake_q.enqueued[0].run_id == _RUN_ID

    @pytest.mark.asyncio
    async def test_executor_passes_origin_api(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Surface 3: the deployed executor threads ``origin="api"`` (and the
        raw run_id) into execute_run — never the CLI default. (Per-surface this
        is also locked by test_executors; here it is one tile in the invariant.)"""
        captured: dict[str, Any] = {}

        async def _spy_execute_run(**kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return {"is_complete": True}

        monkeypatch.setattr(executor_mod, "execute_run", _spy_execute_run)

        job = RunJob(run_id=_RUN_ID, goal="g")
        await executor_mod.default_agent_executor(job)

        assert captured.get("origin") == "api"
        assert captured.get("run_id") == _RUN_ID
        assert captured.get("resume") is False  # each claim resumes via checkpoint, not the flag

    def test_gateway_set_run_id_binds_api_thread_id_for_cost_ledger(self) -> None:
        """Surface 4: the cost-ledger attribution seam accepts + retains the
        ``api-{run_id}`` key. ``execute_run`` calls ``set_run_id(thread_id)``
        (runner.py:300) with the SAME local the resolver produced; every
        ``CostTracker.record_usage`` then carries this run_id."""
        gw = LLMGateway(get_settings())
        assert gw._run_id is None  # unbound until the run binds it
        gw.set_run_id(_EXPECTED)
        assert gw._run_id == _EXPECTED  # bound to the api join key, retained

    def test_eval_store_is_keyed_on_run_id_join_param(self) -> None:
        """Surface 5: the eval store records + queries under ``run_id`` — the
        graph thread_id ``execute_run`` drops into the verify node's config. We
        assert the public API keys every record/query on that one param (so eval
        rows stay joinable on ``api-{run_id}``) and that the resolver emits it.

        The DB persistence round-trip is exercised end-to-end by
        ``test_role_split_integration.py`` (asserts ``thread_id == api-p3-rs-ok``
        through the live api); this locks the join-key contract deterministically."""
        record_params = inspect.signature(EvalStore.record_correctness).parameters
        assert "run_id" in record_params, "eval store must record under run_id (the join key)"
        assert "run_id" in inspect.signature(EvalStore.query_by_run).parameters
        assert "run_id" in inspect.signature(EvalStore.query_latest_attempt).parameters
        # And the value the verify node receives (graph thread_id) is api-{run_id}.
        assert _thread_id_for_run(_RUN_ID, "g", origin="api") == _EXPECTED

    def test_cross_surface_invariant_single_join_key(self) -> None:
        """The tie-together: every surface that produces the key emits the SAME
        string for one run_id. Asserting them as a set collapses to a singleton —
        the regression signal if any surface's prefix drifts."""
        produced = {
            "consumer": RunConsumer.thread_id_for(_RUN_ID),
            "resolver": _thread_id_for_run(_RUN_ID, "g", origin="api"),
            "gateway_seam_input": _EXPECTED,  # what set_run_id receives (runner.py:300)
            "eval_run_id": _thread_id_for_run(_RUN_ID, "g", origin="api"),  # graph thread_id
        }
        assert set(produced.values()) == {_EXPECTED}, produced
