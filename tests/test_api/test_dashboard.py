"""Tests for the operator dashboard (Phase 5) — ``src/api/routes/dashboard.py``
and its read-only data layer ``src/api/routes/dashboard_data.py``.

Two layers, both deterministic and infra-free (no Redis/DB/LLM):

1. **Data-layer unit tests** — exercise the real Python logic (cost join, run
   listing/sort, mutation timeline shaping, the curve matrix math + suffix
   filter + limit cap, best-effort degradation on a DB/Redis error) against
   mocked SQLAlchemy sessions and a fake Redis SCAN.
2. **Route contract tests** — ``TestClient(create_app())`` with the data
   functions + ``_open_store`` + ``get_session`` patched, asserting each
   endpoint renders 200 with the expected fragment, the ``?partial=1``
   fragment path, and 404 for an unknown run.

Mirrors the pattern in ``tests/test_api/test_tool_routes.py``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.routes import dashboard_data as data
from src.worker.schema import JobStatus, RunStatus
from src.worker.status import RunStatusStore


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def _row(**kw: Any) -> SimpleNamespace:
    """A row-like object exposing attributes (what SQLAlchemy ``.all()`` yields)."""
    return SimpleNamespace(**kw)


def _run(run_id: str, *, thread_id: str | None = None, **kw: Any) -> RunStatus:
    """Build a ``RunStatus`` with sensible defaults (thread_id defaults to api-{id})."""
    return RunStatus(run_id=run_id, thread_id=thread_id or f"api-{run_id}", **kw)


class _FakeRedis:
    """Minimal async Redis for ``list_runs``: SCAN + HGETALL over a key→hash map."""

    def __init__(self, hashes: dict[str, dict[str, str]]) -> None:
        self._hashes = hashes

    async def scan_iter(self, *, match: str | None = None, count: int | None = None) -> Any:  # noqa: ARG002
        prefix = (match or "").replace("*", "")
        for key in self._hashes:
            if match is None or key.startswith(prefix):
                yield key

    async def hgetall(self, key: str) -> dict[str, str]:
        return self._hashes.get(key, {})


def _store(hashes: dict[str, dict[str, str]]) -> RunStatusStore:
    """A ``RunStatusStore`` over a fake Redis (status_ttl_s=0 → no expiry calls)."""
    return RunStatusStore(_FakeRedis(hashes), SimpleNamespace(status_ttl_s=0))  # type: ignore[arg-type]


def _session_returning(rows: list[Any]) -> MagicMock:
    """An AsyncSession whose ``execute`` returns a result with ``.all()`` → rows."""
    session = MagicMock()
    session.execute = AsyncMock(return_value=MagicMock(all=lambda: rows))
    return session


def _session_raising(exc: BaseException) -> MagicMock:
    """An AsyncSession whose ``execute`` always raises (best-effort path)."""
    session = MagicMock()
    session.execute = AsyncMock(side_effect=exc)
    return session


class _FakeSessionCtx:
    """An async context manager standing in for ``get_session()`` in route tests."""

    async def __aenter__(self) -> MagicMock:
        return MagicMock()

    async def __aexit__(self, *_args: object) -> bool:
        return False


# ---------------------------------------------------------------------------
# Data layer — pure helpers
# ---------------------------------------------------------------------------


class TestRunToView:
    def test_flattens_status_to_view_dict(self) -> None:
        record = _run("r1", status=JobStatus.RUNNING, iteration_count=3, is_complete=True)
        view = data._run_to_view(record)
        assert view["run_id"] == "r1"
        assert view["thread_id"] == "api-r1"
        assert view["status"] == "running"  # enum → its string value
        assert view["iteration_count"] == 3
        assert view["is_complete"] is True

    def test_flattens_terminal_status(self) -> None:
        record = _run("r2", status=JobStatus.BUDGET_EXHAUSTED, error="cap hit")
        assert data._run_to_view(record)["status"] == "budget_exhausted"


class TestRunCostKeys:
    def test_thread_id_then_bare_run_id(self) -> None:
        keys = data._run_cost_keys({"thread_id": "api-r1", "run_id": "r1"})
        assert keys == ["api-r1", "r1"]

    def test_dedupes_when_thread_equals_run(self) -> None:
        # A status hash whose thread_id == run_id should not duplicate.
        keys = data._run_cost_keys({"thread_id": "r1", "run_id": "r1"})
        assert keys == ["r1"]

    def test_empty_when_neither_present(self) -> None:
        assert data._run_cost_keys({}) == []


class TestCostForRun:
    def test_matches_on_thread_id(self) -> None:
        index = {"api-r1": {"cost_usd": 0.5, "calls": 7, "total_tokens": 100}}
        run = {"thread_id": "api-r1", "run_id": "r1"}
        assert data._cost_for_run(run, index)["cost_usd"] == 0.5

    def test_matches_on_bare_run_id_when_thread_absent(self) -> None:
        index = {"r1": {"cost_usd": 0.2, "calls": 1, "total_tokens": 10}}
        run = {"thread_id": "api-r1", "run_id": "r1"}
        assert data._cost_for_run(run, index)["calls"] == 1

    def test_zeros_when_unattributed(self) -> None:
        run = {"thread_id": "api-x", "run_id": "x"}
        cost = data._cost_for_run(run, {})
        assert cost == {"cost_usd": 0.0, "calls": 0, "total_tokens": 0}


@pytest.mark.asyncio
class TestSummaryCards:
    async def test_counts_in_flight_completed_and_spend(self) -> None:
        runs = [
            {"status": "running"},
            {"status": "queued"},
            {"status": "completed", "is_complete": True},
            {"status": "completed", "is_complete": False},
        ]
        index = {"api-a": {"cost_usd": 1.25, "calls": 3, "total_tokens": 9}}
        summary = await data.summary_cards(runs, index)
        assert summary["runs_total"] == 4
        assert summary["runs_in_flight"] == 2
        assert summary["runs_completed"] == 1
        assert summary["total_cost_usd"] == 1.25


# ---------------------------------------------------------------------------
# Data layer — async DB helpers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRunsCostIndex:
    async def test_groups_by_run_id(self) -> None:
        rows = [
            _row(run_id="api-r1", cost_usd=0.5, calls=3, total_tokens=100),
            _row(run_id="api-r2", cost_usd=0.0, calls=1, total_tokens=10),
        ]
        index = await data.runs_cost_index(_session_returning(rows))
        assert set(index) == {"api-r1", "api-r2"}
        assert index["api-r1"] == {"cost_usd": 0.5, "calls": 3, "total_tokens": 100}

    async def test_degrades_to_empty_on_db_error(self) -> None:
        assert await data.runs_cost_index(_session_raising(RuntimeError("db down"))) == {}


@pytest.mark.asyncio
class TestListRuns:
    @staticmethod
    def _hashes() -> dict[str, dict[str, str]]:
        older = _run("old", status=JobStatus.COMPLETED, started_at="2026-07-01T00:00:00+00:00")
        newer = _run("new", status=JobStatus.RUNNING, started_at="2026-07-15T00:00:00+00:00")
        return {
            f"turing:run:{older.run_id}": older.to_hash(),
            f"turing:run:{newer.run_id}": newer.to_hash(),
        }

    async def test_lists_newest_started_first(self) -> None:
        runs = await data.list_runs(_store(self._hashes()))
        assert [r["run_id"] for r in runs] == ["new", "old"]
        assert runs[0]["status"] == "running"

    async def test_malformed_hash_is_skipped_not_fatal(self) -> None:
        hashes = self._hashes()
        hashes["turing:run:garbage"] = {}  # empty hash → from_hash raises → skipped
        runs = await data.list_runs(_store(hashes))
        assert {r["run_id"] for r in runs} == {"new", "old"}

    async def test_respects_limit(self) -> None:
        runs = await data.list_runs(_store(self._hashes()), limit=1)
        assert len(runs) == 1 and runs[0]["run_id"] == "new"

    async def test_degrades_to_empty_on_redis_error(self) -> None:
        store = MagicMock()
        store._redis = _FakeRedis({})  # scan_iter is fine but force an error path
        # Simulate the scan itself raising.
        store._redis.scan_iter = MagicMock(side_effect=RuntimeError("redis down"))
        assert await data.list_runs(store) == []


@pytest.mark.asyncio
class TestRunWithCost:
    async def test_joins_spend_onto_each_run(self) -> None:
        a = _run("a", status=JobStatus.COMPLETED, started_at="2026-07-10T00:00:00+00:00")
        b = _run("b", status=JobStatus.RUNNING, started_at="2026-07-12T00:00:00+00:00")
        hashes = {
            f"turing:run:{a.run_id}": a.to_hash(),
            f"turing:run:{b.run_id}": b.to_hash(),
        }
        cost_rows = [
            _row(run_id="api-a", cost_usd=0.42, calls=5, total_tokens=222),
        ]
        runs, summary = await data.runs_with_cost(_store(hashes), _session_returning(cost_rows))
        # b sorts first (newer started_at).
        assert [r["run_id"] for r in runs] == ["b", "a"]
        a_view = next(r for r in runs if r["run_id"] == "a")
        assert a_view["cost"]["cost_usd"] == 0.42
        assert next(r for r in runs if r["run_id"] == "b")["cost"]["calls"] == 0
        assert summary["runs_total"] == 2
        assert summary["runs_in_flight"] == 1
        assert summary["total_cost_usd"] == 0.42


@pytest.mark.asyncio
class TestRunCostBreakdown:
    async def test_groups_by_model_most_expensive_first(self) -> None:
        rows = [
            _row(model="glm-5.2", cost_usd=0.9, calls=9, total_tokens=900),
            _row(model="haiku", cost_usd=0.1, calls=1, total_tokens=100),
        ]
        run = {"thread_id": "api-r1", "run_id": "r1"}
        bd = await data.run_cost_breakdown(_session_returning(rows), run)
        assert [c["model"] for c in bd] == ["glm-5.2", "haiku"]
        assert bd[0]["cost_usd"] == 0.9

    async def test_empty_when_no_keys(self) -> None:
        assert await data.run_cost_breakdown(MagicMock(), {}) == []

    async def test_degrades_to_empty_on_db_error(self) -> None:
        run = {"thread_id": "api-r1", "run_id": "r1"}
        assert await data.run_cost_breakdown(_session_raising(RuntimeError("db")), run) == []


@pytest.mark.asyncio
class TestMutationTimeline:
    @staticmethod
    def _rows() -> list[SimpleNamespace]:
        return [
            _row(
                id="mut-1",
                mutation_type="PROMPT",
                target_path="prompts/system_prompt.md",
                description="tighten execute",
                status="deployed",
                diff_content="- old\n+ new",
                model_used="glm-5.2",
                created_at=datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc),
                p_value=0.04,
                is_significant=True,
                confidence=0.95,
                sample_size=3,
                control_value=0.71,
                treatment_value=0.88,
            ),
            _row(
                id="mut-2",
                mutation_type="TOOL",
                target_path=None,
                description="add normalizer",
                status="rejected",
                diff_content=None,
                model_used=None,
                created_at=datetime(2026, 7, 14, 9, 0, tzinfo=timezone.utc),
                p_value=None,
                is_significant=None,
                confidence=None,
                sample_size=None,
                control_value=None,
                treatment_value=None,
            ),
        ]

    async def test_shapes_rows_with_diff_and_ab_stats(self) -> None:
        rows = await data.mutation_timeline(_session_returning(self._rows()))
        assert len(rows) == 2
        first = rows[0]  # newest first
        assert first["id"] == "mut-1"
        assert first["has_diff"] is True
        assert first["diff_content"] == "- old\n+ new"
        assert first["p_value"] == pytest.approx(0.04)
        assert first["is_significant"] is True
        assert first["sample_size"] == 3
        assert first["created_at"].startswith("2026-07-15T12:00")
        second = rows[1]
        assert second["has_diff"] is False
        assert second["p_value"] is None
        assert second["target_path"] is None  # falls back to description in template

    async def test_degrades_to_empty_on_db_error(self) -> None:
        assert await data.mutation_timeline(_session_raising(RuntimeError("db"))) == []


@pytest.mark.asyncio
class TestGenerationCurve:
    @staticmethod
    def _rows() -> list[SimpleNamespace]:
        # Two runs × two goals, terminal-state means precomputed by the SQL.
        return [
            _row(run_id="r1-20260713", goal_id="q01", mean_score=1.0, n_checks=2),
            _row(run_id="r1-20260713", goal_id="q02", mean_score=0.5, n_checks=2),
            _row(run_id="r2-20260714", goal_id="q01", mean_score=0.8, n_checks=2),
            _row(run_id="r2-20260714", goal_id="q02", mean_score=0.6, n_checks=2),
        ]

    async def test_matrix_and_means_without_filter(self) -> None:
        curve = await data.generation_curve(_session_returning(self._rows()))
        # runs sorted desc → r2 first.
        assert curve["runs"] == ["r2-20260714", "r1-20260713"]
        assert curve["goals"] == ["q01", "q02"]
        assert curve["matrix"]["r2-20260714"]["q01"]["mean"] == pytest.approx(0.8)
        # run mean = average of its goal means.
        assert curve["run_means"]["r1-20260713"] == pytest.approx(0.75)
        assert curve["run_means"]["r2-20260714"] == pytest.approx(0.7)
        # goal mean across the visible runs.
        assert curve["goal_means"]["q01"] == pytest.approx(0.9)

    async def test_suffix_filter_keeps_matching_runs_only(self) -> None:
        curve = await data.generation_curve(_session_returning(self._rows()), suffix="20260713")
        assert curve["runs"] == ["r1-20260713"]
        assert curve["run_means"]["r1-20260713"] == pytest.approx(0.75)
        # goal mean recomputed over the filtered set.
        assert curve["goal_means"]["q02"] == pytest.approx(0.5)

    async def test_suffix_matching_nothing_returns_empty(self) -> None:
        curve = await data.generation_curve(_session_returning(self._rows()), suffix="zzz")
        assert curve == {"runs": [], "goals": [], "matrix": {}, "run_means": {}, "goal_means": {}}

    async def test_limit_runs_caps_columns_to_most_recent(self) -> None:
        curve = await data.generation_curve(_session_returning(self._rows()), limit_runs=1)
        assert curve["runs"] == ["r2-20260714"]  # most-recent kept
        # goal means now reflect only r2.
        assert curve["goal_means"]["q01"] == pytest.approx(0.8)

    async def test_degrades_to_empty_on_db_error(self) -> None:
        empty = {"runs": [], "goals": [], "matrix": {}, "run_means": {}, "goal_means": {}}
        assert await data.generation_curve(_session_raising(RuntimeError("db"))) == empty


# ---------------------------------------------------------------------------
# Routes — HTTP contract (patched data layer + infra)
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture
def patch_infra() -> Any:
    """Patch ``_open_store`` + ``get_session`` so no Redis/DB is touched.

    Yields ``(store_mock, redis_mock)`` so a test can set ``store.get`` for the
    run-detail path. The data functions are patched per-test as needed.
    """
    store = MagicMock()
    store.get = AsyncMock(return_value=None)
    redis_mock = MagicMock()
    redis_mock.aclose = AsyncMock()
    with patch(
        "src.api.routes.dashboard._open_store",
        new=AsyncMock(return_value=(store, redis_mock)),
    ), patch(
        "src.api.routes.dashboard.get_session",
        return_value=_FakeSessionCtx(),
    ):
        yield store, redis_mock


class TestIndexRoute:
    def test_renders_summary_and_nav(
        self, client: TestClient, patch_infra: Any
    ) -> None:
        runs = [{"run_id": "r1", "thread_id": "api-r1", "status": "completed",
                 "is_complete": True, "iteration_count": 2, "started_at": "2026-07-15",
                 "finished_at": "2026-07-15", "error": "", "final_output": "",
                 "results_dir": "", "cost": {"cost_usd": 0.0, "calls": 0, "total_tokens": 0}}]
        summary = {"runs_total": 1, "runs_in_flight": 0, "runs_completed": 1, "total_cost_usd": 0.0}
        with patch.object(data, "runs_with_cost", new=AsyncMock(return_value=(runs, summary))), \
             patch.object(data, "mutation_timeline", new=AsyncMock(return_value=[])):
            resp = client.get("/dashboard")
        assert resp.status_code == 200, resp.text
        body = resp.text
        assert "Turing Dashboard" in body
        assert "Overview" in body
        assert "Recent runs" in body
        assert "/dashboard/curve" in body  # nav link present

    def test_index_degrades_to_empty_on_data_error(
        self, client: TestClient, patch_infra: Any
    ) -> None:
        # The route's own try/except returns a zeroed summary + empty lists.
        with patch.object(data,"runs_with_cost", new=AsyncMock(side_effect=RuntimeError("x"))), \
             patch.object(data,"mutation_timeline", new=AsyncMock(side_effect=RuntimeError("x"))):
            resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert "Overview" in resp.text  # page still renders


class TestRunsRoute:
    def test_full_page_renders_table(self, client: TestClient, patch_infra: Any) -> None:
        runs: list[dict[str, Any]] = []
        summary = {"runs_total": 0, "runs_in_flight": 0, "runs_completed": 0, "total_cost_usd": 0.0}
        with patch.object(data,"runs_with_cost", new=AsyncMock(return_value=(runs, summary))):
            resp = client.get("/dashboard/runs")
        assert resp.status_code == 200, resp.text
        assert "auto-refresh" in resp.text  # the runs-page header
        assert 'data-poll="/dashboard/runs?partial=1"' in resp.text

    def test_partial_renders_only_rows_fragment(
        self, client: TestClient, patch_infra: Any
    ) -> None:
        run = {"run_id": "r1", "status": "running", "iteration_count": 1,
               "started_at": "2026-07-15T00:00:00", "finished_at": "",
               "cost": {"cost_usd": 0.0, "calls": 0, "total_tokens": 0},
               "error": "", "is_complete": False}
        with patch.object(data,"runs_with_cost",
                          new=AsyncMock(return_value=([run], {"runs_total": 1, "runs_in_flight": 1, "runs_completed": 0, "total_cost_usd": 0.0}))):
            resp = client.get("/dashboard/runs?partial=1")
        assert resp.status_code == 200, resp.text
        body = resp.text
        # Fragment is just the <tr> rows — NOT a full document.
        assert "<html" not in body
        assert "/dashboard/runs/r1" in body  # run-detail link in the row
        assert "running" in body


class TestRunDetailRoute:
    def test_renders_detail_review_and_cost(
        self, client: TestClient, patch_infra: Any
    ) -> None:
        store, _ = patch_infra
        record = _run("r1", status=JobStatus.COMPLETED, is_complete=True,
                      iteration_count=4, final_output="the answer is 42", error="")
        store.get = AsyncMock(return_value=record)
        breakdown = [{"model": "glm-5.2", "cost_usd": 0.3, "calls": 2, "total_tokens": 200}]
        with patch.object(data,"run_cost_breakdown", new=AsyncMock(return_value=breakdown)):
            resp = client.get("/dashboard/runs/r1")
        assert resp.status_code == 200, resp.text
        body = resp.text
        assert "Run <code>r1</code>" in body
        assert "the answer is 42" in body  # the review card surfaces the full output
        assert "Cost by model" in body
        assert "glm-5.2" in body

    def test_unknown_run_returns_404(
        self, client: TestClient, patch_infra: Any
    ) -> None:
        store, _ = patch_infra
        store.get = AsyncMock(return_value=None)  # unknown / expired run
        resp = client.get("/dashboard/runs/does-not-exist")
        assert resp.status_code == 404


class TestCurveRoute:
    def test_renders_chart_and_matrix(self, client: TestClient, patch_infra: Any) -> None:
        curve = {
            "runs": ["r2", "r1"],
            "goals": ["q01"],
            "matrix": {"r2": {"q01": {"mean": 0.9, "n": 2}}, "r1": {"q01": {"mean": 0.5, "n": 2}}},
            "run_means": {"r2": 0.9, "r1": 0.5},
            "goal_means": {"q01": 0.7},
        }
        with patch.object(data,"generation_curve", new=AsyncMock(return_value=curve)):
            resp = client.get("/dashboard/curve")
        assert resp.status_code == 200, resp.text
        body = resp.text
        assert "Generation curve" in body
        assert "Run means" in body
        assert "<svg" in body  # server-side SVG chart rendered

    def test_suffix_passed_through(self, client: TestClient, patch_infra: Any) -> None:
        with patch.object(
            data, "generation_curve",
            new=AsyncMock(return_value={"runs": [], "goals": [], "matrix": {}, "run_means": {}, "goal_means": {}}),
        ) as gen:
            resp = client.get("/dashboard/curve?suffix=G2")
        assert resp.status_code == 200
        gen.assert_awaited_once()
        assert gen.await_args is not None
        assert gen.await_args.kwargs.get("suffix") == "G2"

    def test_empty_curve_renders_placeholder(self, client: TestClient, patch_infra: Any) -> None:
        empty = {"runs": [], "goals": [], "matrix": {}, "run_means": {}, "goal_means": {}}
        with patch.object(data,"generation_curve", new=AsyncMock(return_value=empty)):
            resp = client.get("/dashboard/curve")
        assert resp.status_code == 200
        assert "No eval rows match" in resp.text


class TestMutationsRoute:
    def test_renders_timeline_with_diff(self, client: TestClient, patch_infra: Any) -> None:
        mutations = [{
            "id": "mut-1", "mutation_type": "PROMPT", "target_path": "prompts/x.md",
            "description": "d", "status": "deployed", "has_diff": True, "diff_content": "- a\n+ b",
            "model_used": "glm-5.2", "created_at": "2026-07-15T12:00:00+00:00",
            "p_value": 0.04, "is_significant": True, "confidence": 0.95, "sample_size": 3,
            "control_value": 0.7, "treatment_value": 0.9,
        }]
        with patch.object(data,"mutation_timeline", new=AsyncMock(return_value=mutations)):
            resp = client.get("/dashboard/mutations")
        assert resp.status_code == 200, resp.text
        body = resp.text
        assert "Mutation timeline" in body
        assert "PROMPT" in body  # type badge
        assert "- a" in body  # diff surfaced in the <details>

    def test_empty_renders_placeholder(self, client: TestClient, patch_infra: Any) -> None:
        with patch.object(data,"mutation_timeline", new=AsyncMock(return_value=[])):
            resp = client.get("/dashboard/mutations")
        assert resp.status_code == 200
        assert "No mutations yet" in resp.text
