"""API route + sub-agent runner/registry/persister gap tests.

Focused on error paths and DI wiring NOT already covered:
  - src/api/routes/health.py — /health liveness, /ready degraded when DB down
    (DI overridden to fakes; NEVER a real DB/Redis).
  - src/api/routes/agent.py — submit returns run_id + enqueues; status maps
    run_id -> state; duplicate-in-flight is 409; cancel unknown is 404;
    Redis-unreachable is 503. All via patched ``_client_and_queue`` fakes.
  - src/agents/registry.py — the production <=60 governance cap on the real
    ``load_active_agents`` path (existing cap tests use tiny numbers; this
    exercises ``AgentSettings.max_active_sub_agents`` end-to-end through the
    persister loader).
  - src/agents/runner.py — execute-subtask returns its result; depth-limit
    short-circuit; build-failure is non-fatal.
  - src/agents/persister.py — DB failure is non-fatal (returns None / []).
  - src/agents/subgraph.py — _ModelOverrideProxy forces model; scope_tools
    inherit_subset skips missing tools.

External I/O is mocked everywhere (FastAPI DI / monkeypatched module funcs /
AsyncMock). No real DB / Redis / LLM / LangGraph execution. Fixed data.
asyncio_mode=auto. No @pytest.mark.e2e.

NOTE: existing coverage we deliberately do NOT duplicate:
  - tests/test_agents/test_selection_ranking.py — AGENT_SELECTION ranking.
  - tests/test_scheduler/test_agent_cron_registration.py — cron registration.
  - tests/test_api/test_app.py — route registration / dual-mount.
  - tests/test_agents/test_registry_caps.py — enforce_caps with tiny numbers.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.graph.enums import TaskComplexity
from src.graph.models import SubAgentSpec


# ─── Helpers ──────────────────────────────────────────────────────────────


def _spec(
    name: str,
    *,
    is_active: bool = True,
    success_rate: float = 0.9,
    total_runs: int = 100,
    quality_score: float = 0.8,
    last_used_at: datetime | None = None,
) -> SubAgentSpec:
    """Deterministic SubAgentSpec fixture."""
    return SubAgentSpec(
        id=str(uuid.uuid4()),
        goal="",
        name=name,
        description=f"{name} description",
        model_tier=TaskComplexity.SIMPLE,
        parent_thread_id="",
        is_active=is_active,
        success_rate=success_rate,
        total_runs=total_runs,
        quality_score=quality_score,
        last_used_at=last_used_at,
    )


class _FakeStatusStore:
    """In-memory RunStatusStore double — dict keyed by run_id."""

    def __init__(self) -> None:
        self.records: dict[str, Any] = {}
        self.cancel_calls: list[str] = []

    async def get(self, run_id: str) -> Any:
        return self.records.get(run_id)

    async def mark(
        self, run_id: str, thread_id: str, status: Any, **fields: Any
    ) -> Any:
        from src.worker.schema import RunStatus

        rec = self.records.get(run_id) or RunStatus(
            run_id=run_id, thread_id=thread_id
        )
        rec.thread_id = thread_id
        rec.status = status
        for k, v in fields.items():
            if v is not None and hasattr(rec, k):
                setattr(rec, k, v)
        self.records[run_id] = rec
        return rec

    async def request_cancel(self, run_id: str, ttl: int | None = None) -> None:
        self.cancel_calls.append(run_id)


class _FakeQueue:
    """RunsQueue double — records enqueue/delete calls."""

    def __init__(self) -> None:
        self.enqueued: list[Any] = []
        self.deleted: list[str] = []

    async def ensure_group(self) -> None:
        return None

    async def enqueue(self, job: Any) -> str:
        self.enqueued.append(job)
        return "1-0"

    async def delete_entry(self, entry_id: str) -> bool:
        self.deleted.append(entry_id)
        return True


@pytest.fixture
def app_client() -> Iterator[TestClient]:
    """A TestClient over create_app() — route handlers are patched per-test."""
    with TestClient(create_app()) as client:
        yield client


def _patch_queue(
    monkeypatch: pytest.MonkeyPatch,
    queue: _FakeQueue | None = None,
    store: _FakeStatusStore | None = None,
) -> tuple[_FakeQueue, _FakeStatusStore]:
    """Patch agent.py's ``_client_and_queue`` with fakes (no real Redis)."""
    queue = queue or _FakeQueue()
    store = store or _FakeStatusStore()
    monkeypatch.setattr(
        "src.api.routes.agent._client_and_queue",
        lambda: (queue, store),
    )
    return queue, store


# ═══════════════════════════════════════════════════════════════════════════
# /health + /ready
# ═══════════════════════════════════════════════════════════════════════════


class TestHealthRoutes:
    def test_health_liveness_returns_200(self, app_client: TestClient) -> None:
        """Liveness probe is always alive (200) — no dependency checks."""
        resp = app_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "alive"
        assert body["service"] == "turing-agent"

    def test_ready_degraded_when_db_down(
        self,
        app_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``/ready`` reports degraded when the DB ``SELECT 1`` fails.

        DI override: a fake engine whose ``connect()`` raises, and a fake Redis
        client whose ``ping`` also raises — so the route's bare ``except`` sets
        both checks False and the status is "degraded". NEVER a real DB/Redis.
        """

        class _BadConn:
            async def __aenter__(self) -> "_BadConn":
                raise RuntimeError("DB down")

            async def __aexit__(self, *exc: object) -> None:
                return None

        bad_engine = MagicMock()
        bad_engine.connect = MagicMock(return_value=_BadConn())
        monkeypatch.setattr("src.db.session.get_engine", lambda: bad_engine)

        bad_redis = MagicMock()
        bad_redis.ping = AsyncMock(side_effect=RuntimeError("redis down"))
        bad_redis.aclose = AsyncMock(return_value=None)
        monkeypatch.setattr(
            "redis.asyncio.from_url", lambda url: bad_redis
        )

        resp = app_client.get("/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["checks"]["postgresql"] is False
        assert body["checks"]["redis"] is False

    def test_ready_reports_db_down_redis_up(
        self,
        app_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Degraded is also correct for a mixed state (DB down, Redis up)."""

        class _BadConn:
            async def __aenter__(self) -> "_BadConn":
                raise RuntimeError("DB down")

            async def __aexit__(self, *exc: object) -> None:
                return None

        bad_engine = MagicMock()
        bad_engine.connect = MagicMock(return_value=_BadConn())
        monkeypatch.setattr("src.db.session.get_engine", lambda: bad_engine)

        good_redis = MagicMock()
        good_redis.ping = AsyncMock(return_value=True)
        good_redis.aclose = AsyncMock(return_value=None)
        monkeypatch.setattr(
            "redis.asyncio.from_url", lambda url: good_redis
        )

        body = app_client.get("/ready").json()
        assert body["status"] == "degraded"
        assert body["checks"] == {"postgresql": False, "redis": True}


# ═══════════════════════════════════════════════════════════════════════════
# POST /run + GET /runs/{run_id} + POST /runs/{run_id}/cancel
# ═══════════════════════════════════════════════════════════════════════════


class TestAgentRunRoutes:
    def test_submit_enqueues_and_returns_run_id(
        self,
        app_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A POST /run with a valid goal returns 202 + a fresh run_id and the
        status_url, and the RunJob is enqueued on the queue."""
        queue, store = _patch_queue(monkeypatch)

        resp = app_client.post("/run", json={"goal": "summarize quicksort"})

        assert resp.status_code == 202, resp.text
        body = resp.json()
        assert "run_id" in body and body["run_id"]
        assert body["thread_id"] == f"api-{body['run_id']}"
        assert body["status"] == "queued"
        assert body["status_url"].endswith(body["run_id"])
        assert len(queue.enqueued) == 1
        enqueued = queue.enqueued[0]
        assert enqueued.run_id == body["run_id"]
        assert enqueued.goal == "summarize quicksort"
        # The status store recorded the QUEUED mark for this run_id.
        assert store.records[body["run_id"]].status.value == "queued"

    def test_submit_honors_explicit_run_id(
        self,
        app_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An explicit run_id is threaded through (thread_id = api-{run_id})."""
        queue, _ = _patch_queue(monkeypatch)

        resp = app_client.post(
            "/run", json={"goal": "do thing", "run_id": "my-run-123"}
        )
        assert resp.status_code == 202
        body = resp.json()
        assert body["run_id"] == "my-run-123"
        assert body["thread_id"] == "api-my-run-123"
        assert queue.enqueued[0].run_id == "my-run-123"

    def test_submit_rejects_empty_goal_with_422(
        self,
        app_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Pydantic min_length=1 validation rejects an empty goal (422)."""
        _patch_queue(monkeypatch)
        resp = app_client.post("/run", json={"goal": ""})
        assert resp.status_code == 422

    def test_submit_duplicate_in_flight_is_409(
        self,
        app_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A second POST /run for an already-QUEUED run is refused (409) — the
        P1 respawn/double-spend guard."""
        _, store = _patch_queue(monkeypatch)
        from src.worker.schema import JobStatus, RunStatus

        store.records["dupe-1"] = RunStatus(
            run_id="dupe-1", thread_id="api-dupe-1", status=JobStatus.RUNNING
        )

        resp = app_client.post(
            "/run", json={"goal": "x", "run_id": "dupe-1"}
        )
        assert resp.status_code == 409
        assert "dupe-1" in resp.json()["detail"]

    def test_submit_redis_build_failure_is_server_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If ``_client_and_queue`` raises (Redis unreachable), the route does
        NOT catch the bare build error (its try/except wraps enqueue/mark, not
        the client build) → a 5xx server error, never a 2xx success. Starlette's
        TestClient re-raises unhandled server errors by default, so we disable
        that to observe the real status code."""
        monkeypatch.setattr(
            "src.api.routes.agent._client_and_queue",
            MagicMock(side_effect=RuntimeError("connection refused")),
        )
        with TestClient(
            create_app(), raise_server_exceptions=False
        ) as client:
            resp = client.post("/run", json={"goal": "x"})
        assert resp.status_code >= 500
        # No run_id is minted on a failed client build (body is the generic
        # 500 error text, not the EnqueueResponse JSON).
        assert "run_id" not in resp.text

    def test_status_unknown_run_is_404(
        self,
        app_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """GET /runs/{run_id} on an unknown/expired run is 404."""
        _patch_queue(monkeypatch)
        resp = app_client.get("/runs/never-seen")
        assert resp.status_code == 404
        assert "never-seen" in resp.json()["detail"]

    def test_status_maps_run_id_to_state(
        self,
        app_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """GET /runs/{run_id} reflects the recorded status, final_output, and
        iteration_count."""
        _, store = _patch_queue(monkeypatch)
        from src.worker.schema import JobStatus, RunStatus

        store.records["r1"] = RunStatus(
            run_id="r1",
            thread_id="api-r1",
            status=JobStatus.COMPLETED,
            final_output="done",
            is_complete=True,
            iteration_count=4,
        )

        resp = app_client.get("/runs/r1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["run_id"] == "r1"
        assert body["status"] == "completed"
        assert body["is_complete"] is True
        assert body["final_output"] == "done"
        assert body["iteration_count"] == 4

    def test_cancel_unknown_run_is_404(
        self,
        app_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """POST /runs/{id}/cancel on an unknown run is 404 (not a silent no-op)."""
        _patch_queue(monkeypatch)
        resp = app_client.post("/runs/ghost/cancel")
        assert resp.status_code == 404

    def test_cancel_known_run_sets_flag_and_deletes_entry(
        self,
        app_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cancel of a known run with a recorded entry_id sets the cancel flag
        AND deletes the pending stream entry (P1 — no peer respawn)."""
        queue, store = _patch_queue(monkeypatch)
        from src.worker.schema import JobStatus, RunStatus

        store.records["r2"] = RunStatus(
            run_id="r2",
            thread_id="api-r2",
            status=JobStatus.RUNNING,
            entry_id="9-9",
        )

        resp = app_client.post("/runs/r2/cancel")
        assert resp.status_code == 202
        assert resp.json()["status"] == "cancel_requested"
        assert store.cancel_calls == ["r2"]
        assert queue.deleted == ["9-9"]

    def test_cancel_known_run_no_entry_id_is_flag_only(
        self,
        app_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cancel of a run with no recorded entry_id still sets the flag
        (flag-only path; worker ack + run timeout remain the backstop)."""
        queue, store = _patch_queue(monkeypatch)
        from src.worker.schema import JobStatus, RunStatus

        store.records["r3"] = RunStatus(
            run_id="r3", thread_id="api-r3", status=JobStatus.RUNNING,
        )
        resp = app_client.post("/runs/r3/cancel")
        assert resp.status_code == 202
        assert store.cancel_calls == ["r3"]
        assert queue.deleted == []  # nothing to delete


# ═══════════════════════════════════════════════════════════════════════════
# Sub-agent registry — production <=60 governance cap
# ═══════════════════════════════════════════════════════════════════════════


class TestSubAgentRegistryCap:
    def test_enforce_caps_at_production_limit_60(self) -> None:
        """The real AgentSettings cap (60) is honored: 65 healthy active agents
        are trimmed to exactly 60, with the lowest-scoring 5 retired."""
        from src.agents.registry import SubAgentRegistry
        from src.config.settings import AgentSettings

        cap = AgentSettings().max_active_sub_agents
        assert cap == 60  # the documented production cap

        reg = SubAgentRegistry()
        # 65 distinct, HEALTHY (success_rate above the deprecation floor so
        # check_deprecation leaves them alone), recently-used agents. Scores
        # ascending so the cap path retires the lowest 5 — this isolates the
        # cumulative-cap retirement from the chronic-low-performer path.
        now = datetime.now(timezone.utc)
        for i in range(65):
            score = 0.60 + (i / 200.0)  # 0.60..0.92 -> ascending, all >= floor
            reg.register(
                _spec(
                    f"agent_{i:03d}",
                    success_rate=score,
                    total_runs=200,
                    quality_score=score,
                    last_used_at=now,
                )
            )
        assert reg.active_count == 65

        retired = reg.enforce_caps(
            max_active=cap,
            min_runs=20,
            success_floor=0.5,
            recency_days=30,
            now=now,
        )

        assert reg.active_count == cap  # trimmed to 60
        assert len(retired) == 5
        # Lowest (success_rate, total_runs, quality_score) tuples retired first
        # → agent_000..agent_004 (the 5 lowest scores).
        assert retired == [f"agent_{i:03d}" for i in range(5)]
        # Survivors include the highest scorer; the lowest is gone.
        assert reg.has("agent_064")
        lowest = reg.get("agent_000")
        assert lowest is not None
        assert not lowest.is_active

    def test_enforce_caps_under_limit_retires_nothing(self) -> None:
        """When active_count <= cap, no healthy agent is retired."""
        from src.agents.registry import SubAgentRegistry

        reg = SubAgentRegistry()
        now = datetime.now(timezone.utc)
        for i in range(10):
            reg.register(
                _spec(f"a{i}", success_rate=0.9, last_used_at=now)
            )
        retired = reg.enforce_caps(max_active=60, now=now)
        assert retired == []
        assert reg.active_count == 10

    async def test_load_active_agents_enforces_cap_via_persister(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """load_active_agents threads AgentSettings.max_active_sub_agents into
        registry.enforce_caps and persists the retirements — the runtime
        governance entry point (Bug E). 65 active rows → 60 loaded."""
        from src.agents.persister import SubAgentPersister
        from src.agents.registry import SubAgentRegistry
        from src.config.settings import AgentSettings

        settings = AgentSettings()
        assert settings.max_active_sub_agents == 60

        now = datetime.now(timezone.utc)

        class _Row:
            """A plain stub for a SubAgentModel row.

            MagicMock's ``name=`` ctor arg sets the mock's repr, NOT the
            ``.name`` attribute, so ``_model_to_spec`` would read a child-mock
            and fail SubAgentSpec's str validation. A real object sidesteps
            that and keeps every attribute properly typed.
            """

            def __init__(self, i: int) -> None:
                score = 0.60 + (i / 200.0)  # healthy, ascending
                self.id = uuid.uuid4()
                self.name = f"agent_{i:03d}"
                self.description = f"d{i}"
                self.template_type = "fixed"
                self.tool_scope = "inherit_all"
                self.tool_subset = []
                self.budget_mode = "shared"
                self.budget_limit = 0.0
                self.model_tier = TaskComplexity.SIMPLE
                self.max_iterations = 10
                self.depth_limit = 0
                self.node_config = {}
                self.system_prompt_override = None
                self.is_active = True
                self.version = 1
                self.total_runs = 200
                self.success_rate = score
                self.avg_cost = 0.0
                self.avg_latency_ms = 0
                self.quality_score = score
                self.updated_at = now

        mock_models = [_Row(i) for i in range(65)]

        # Patch the DB-touching pieces inside load_active_agents. The execute()
        # return value supports the ``.scalars().all()`` chain used by the loader.
        scalars_all = MagicMock(return_value=mock_models)
        scalars_obj = MagicMock()
        scalars_obj.all = scalars_all
        exec_result = MagicMock()
        exec_result.scalars.return_value = scalars_obj
        fake_session = MagicMock()
        fake_session.execute = AsyncMock(return_value=exec_result)

        class _FakeSessionCM:
            async def __aenter__(self) -> Any:
                return fake_session

            async def __aexit__(self, *exc: object) -> None:
                return None

        monkeypatch.setattr(
            "src.db.session.get_session", lambda: _FakeSessionCM()
        )
        # retire_redundant + retire are DB-touching; stub them to no-ops so the
        # test isolates the in-memory cap path.
        monkeypatch.setattr(
            SubAgentPersister, "retire_redundant", AsyncMock(return_value=[])
        )
        retire_mock = AsyncMock(return_value=5)
        monkeypatch.setattr(SubAgentPersister, "retire", retire_mock)

        reg = SubAgentRegistry()
        persister = SubAgentPersister()
        loaded = await persister.load_active_agents(reg, settings=settings)

        # Exactly cap survivors survive; the 5 lowest were retired + persisted.
        assert reg.active_count == 60
        assert len(loaded) == 60
        assert "agent_064" in loaded  # top scorer survives
        assert "agent_000" not in loaded  # lowest retired
        retire_mock.assert_awaited_once()
        call_args = retire_mock.await_args
        assert call_args is not None
        retired_names = call_args.args[0]
        assert len(retired_names) == 5
        assert retired_names[0] == "agent_000"


# ═══════════════════════════════════════════════════════════════════════════
# Sub-agent runner — execute-subtask + error paths
# ═══════════════════════════════════════════════════════════════════════════


class TestSubAgentRunner:
    async def test_run_returns_subtask_result(
        self, mock_gateway: MagicMock, mock_tools: MagicMock
    ) -> None:
        """run() executes the subgraph and returns a structured result dict
        keyed by success/result/goal/sub_agent_name."""
        from src.agents.runner import SubAgentRunner

        spec = _spec("summarizer")
        runner = SubAgentRunner(
            definition=spec, gateway=mock_gateway, tools=mock_tools
        )

        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(
            return_value={
                "is_complete": True,
                "final_output": "summary of quicksort",
                "iteration_count": 2,
                "errors": [],
                "cost_records": [],
                "total_tokens_used": 120,
            }
        )
        graph = MagicMock()
        graph.compile.return_value = compiled

        with patch("src.agents.runner.build_subgraph", return_value=graph):
            result = await runner.run(
                goal="summarize quicksort",
                parent_thread_id="parent-1",
            )

        assert result["success"] is True
        assert result["result"] == "summary of quicksort"
        assert result["goal"] == "summarize quicksort"
        assert result["sub_agent_name"] == "summarizer"
        assert result["sub_agent_id"] == spec.id
        assert result["iterations"] == 2
        assert result["tokens_used"] == 120
        assert result["latency_ms"] >= 0

    async def test_run_depth_limit_short_circuits(
        self, mock_gateway: MagicMock, mock_tools: MagicMock
    ) -> None:
        """At depth >= depth_limit, run() returns a failed result WITHOUT
        building/executing the subgraph."""
        from src.agents.runner import SubAgentRunner

        spec = _spec("deep")
        spec.depth_limit = 2
        runner = SubAgentRunner(
            definition=spec, gateway=mock_gateway, tools=mock_tools
        )

        with patch("src.agents.runner.build_subgraph") as mock_build:
            result = await runner.run("g", "t", depth=2)

        assert result["success"] is False
        assert result["result"] == ""
        assert any("Depth limit" in e for e in result["errors"])
        assert result["sub_agent_name"] == "deep"
        # The subgraph was never built (the short-circuit fires first).
        mock_build.assert_not_called()

    async def test_run_build_failure_is_non_fatal(
        self, mock_gateway: MagicMock, mock_tools: MagicMock
    ) -> None:
        """If build_subgraph raises, run() catches it and returns a structured
        failed result (never re-raises into the parent graph)."""
        from src.agents.runner import SubAgentRunner

        spec = _spec("broken")
        runner = SubAgentRunner(
            definition=spec, gateway=mock_gateway, tools=mock_tools
        )

        with patch(
            "src.agents.runner.build_subgraph",
            side_effect=RuntimeError("graph explode"),
        ):
            result = await runner.run("g", "t")

        assert result["success"] is False
        assert any("execution error" in e.lower() for e in result["errors"])
        assert "graph explode" in result["errors"][0]
        assert result["sub_agent_name"] == "broken"

    async def test_model_affinity_forces_model_via_proxy(
        self, mock_gateway: MagicMock, mock_tools: MagicMock
    ) -> None:
        """A model_affinity set on the runner routes build_subgraph through the
        _ModelOverrideProxy, which forces the model on every acompletion call.

        The proxy is constructed INSIDE build_subgraph and handed to
        ``_build_fixed_subgraph`` as the effective gateway, so we patch the
        latter to capture the wrapped gateway (NOT the raw runner gateway)."""
        from src.agents.runner import SubAgentRunner
        from src.agents.subgraph import _ModelOverrideProxy

        spec = _spec("routed")
        runner = SubAgentRunner(
            definition=spec, gateway=mock_gateway, tools=mock_tools
        )
        runner._model_affinity = "glm-4.7"

        captured: dict[str, Any] = {}

        def _capture_fixed(s, gateway, tools, memory):
            captured["gateway"] = gateway
            compiled = MagicMock()
            compiled.ainvoke = AsyncMock(return_value={"is_complete": True})
            graph = MagicMock()
            graph.compile.return_value = compiled
            return graph

        with patch(
            "src.agents.subgraph._build_fixed_subgraph",
            side_effect=_capture_fixed,
        ):
            await runner.run("g", "t")

        proxy = captured["gateway"]
        assert isinstance(proxy, _ModelOverrideProxy)
        assert proxy._model == "glm-4.7"

        # The proxy forces the model on acompletion (forwards to the real gw).
        await proxy.acompletion(messages=[{"role": "user", "content": "hi"}])
        mock_gateway.acompletion.assert_awaited_once()
        gw_call = mock_gateway.acompletion.await_args
        assert gw_call is not None
        assert gw_call.kwargs["model"] == "glm-4.7"


# ═══════════════════════════════════════════════════════════════════════════
# Sub-agent persister — DB failure is non-fatal
# ═══════════════════════════════════════════════════════════════════════════


class TestSubAgentPersisterFailure:
    async def test_persist_db_failure_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A DB failure in persist() returns None — never re-raises (the
        CostTracker-resilience pattern: persistence is best-effort)."""
        from src.agents.persister import SubAgentPersister

        class _ExplodingCM:
            async def __aenter__(self) -> "_ExplodingCM":
                raise RuntimeError("DB connection lost")

            async def __aexit__(self, *exc: object) -> None:
                return None

        monkeypatch.setattr(
            "src.db.session.get_session", lambda: _ExplodingCM()
        )

        persister = SubAgentPersister()
        result = await persister.persist(_spec("doomed"))
        assert result is None

    async def test_find_similar_db_failure_returns_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """find_similar degrades to [] on DB error so dedup falls back to spawn."""
        from src.agents.persister import SubAgentPersister

        class _ExplodingCM:
            async def __aenter__(self) -> "_ExplodingCM":
                raise RuntimeError("DB down")

            async def __aexit__(self, *exc: object) -> None:
                return None

        monkeypatch.setattr(
            "src.db.session.get_session", lambda: _ExplodingCM()
        )

        persister = SubAgentPersister()
        matches = await persister.find_similar([0.1] * 768)
        assert matches == []

    async def test_load_active_agents_db_failure_returns_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A total DB failure during load yields no loaded agents (and does not
        raise) — the registry stays empty, not crash."""
        from src.agents.persister import SubAgentPersister
        from src.agents.registry import SubAgentRegistry

        class _ExplodingCM:
            async def __aenter__(self) -> "_ExplodingCM":
                raise RuntimeError("DB unreachable")

            async def __aexit__(self, *exc: object) -> None:
                return None

        monkeypatch.setattr(
            "src.db.session.get_session", lambda: _ExplodingCM()
        )

        reg = SubAgentRegistry()
        persister = SubAgentPersister()
        loaded = await persister.load_active_agents(reg)  # settings=None
        assert loaded == []
        assert reg.active_count == 0

    async def test_record_run_db_failure_is_non_fatal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """record_run_and_update_metrics returns None on DB failure (run metrics
        are observability-only; a hiccup can never abort the parent run)."""
        from src.agents.persister import SubAgentPersister

        class _ExplodingCM:
            async def __aenter__(self) -> "_ExplodingCM":
                raise RuntimeError("DB gone")

            async def __aexit__(self, *exc: object) -> None:
                return None

        monkeypatch.setattr(
            "src.db.session.get_session", lambda: _ExplodingCM()
        )

        persister = SubAgentPersister()
        result = await persister.record_run_and_update_metrics(
            uuid.uuid4(),
            {"success": True, "result": "ok", "tokens_used": 5, "cost_usd": 0.0},
        )
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# Sub-agent subgraph — tool scoping + proxy
# ═══════════════════════════════════════════════════════════════════════════


class TestSubgraphHelpers:
    def test_scope_tools_inherit_subset_skips_missing(self) -> None:
        """inherit_subset copies only the named tools that exist in the parent
        registry and silently skips the rest."""
        from src.agents.subgraph import scope_tools
        from src.tools.registry import ToolRegistry

        parent = ToolRegistry()
        parent.register(
            "alpha", AsyncMock(return_value="a"), "alpha tool", {"type": "object"}
        )
        parent.register(
            "beta", AsyncMock(return_value="b"), "beta tool", {"type": "object"}
        )

        spec = _spec("scoped")
        spec.tool_scope = "inherit_subset"
        spec.tool_subset = ["alpha", "missing", "beta"]

        scoped = scope_tools(spec, parent)
        assert sorted(scoped.list_names()) == ["alpha", "beta"]

    def test_model_override_proxy_delegates_attributes(self) -> None:
        """The proxy forwards unknown attributes (cost_tracker, cache, …) to the
        underlying gateway via __getattr__."""
        from src.agents.subgraph import _ModelOverrideProxy

        gateway = MagicMock()
        gateway.cost_tracker = "TRACKER"
        proxy = _ModelOverrideProxy(gateway, "glm-4.7")
        assert proxy.cost_tracker == "TRACKER"
        assert proxy._model == "glm-4.7"
