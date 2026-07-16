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

import uuid
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Generator
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
    session.rollback = AsyncMock()
    return session


def _session_raising(exc: BaseException) -> MagicMock:
    """An AsyncSession whose ``execute`` always raises (best-effort path)."""
    session = MagicMock()
    session.execute = AsyncMock(side_effect=exc)
    session.rollback = AsyncMock()
    return session


def _session_one_returning(row: Any) -> MagicMock:
    """An AsyncSession whose ``execute`` returns a result with ``.one()`` → row.

    ``run_token_split`` aggregates to a single row via ``.one()``, not ``.all()``,
    so it needs a result whose ``.one()`` yields the row.
    """
    session = MagicMock()
    session.execute = AsyncMock(return_value=MagicMock(one=lambda: row))
    session.rollback = AsyncMock()
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
class TestExecutionSteps:
    @staticmethod
    def _rows() -> list[SimpleNamespace]:
        # Three node invocations, chronological by created_at.
        return [
            _row(phase="classify_node", status="completed", duration_ms=1500,
                  created_at=datetime(2026, 7, 15, 12, 0, 0, tzinfo=timezone.utc)),
            _row(phase="execute_node", status="completed", duration_ms=22500,
                  created_at=datetime(2026, 7, 15, 12, 0, 5, tzinfo=timezone.utc)),
            _row(phase="verify_node", status="failed", duration_ms=3000,
                  created_at=datetime(2026, 7, 15, 12, 0, 10, tzinfo=timezone.utc)),
        ]

    async def test_chronological_seq_phase_status_duration(self) -> None:
        run = {"thread_id": "api-r1", "run_id": "r1"}
        steps = await data.execution_steps(_session_returning(self._rows()), run)
        assert [s["seq"] for s in steps] == [1, 2, 3]  # synthetic seq (step_number is always 0)
        assert [s["phase"] for s in steps] == ["classify_node", "execute_node", "verify_node"]
        assert steps[0]["status"] == "completed"
        assert steps[2]["status"] == "failed"
        # duration_ms preserved + duration_s derived (22500ms → 22.5s).
        assert steps[1]["duration_ms"] == 22500
        assert steps[1]["duration_s"] == 22.5
        assert steps[0]["created_at"].startswith("2026-07-15T12:00")

    async def test_empty_when_no_keys(self) -> None:
        # A run view with neither thread_id nor run_id yields no candidate keys.
        assert await data.execution_steps(MagicMock(), {}) == []

    async def test_degrades_to_empty_on_db_error(self) -> None:
        run = {"thread_id": "api-r1", "run_id": "r1"}
        assert await data.execution_steps(_session_raising(RuntimeError("db")), run) == []


@pytest.mark.asyncio
class TestRunTokenSplit:
    _ZEROS = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0,
              "total_tokens": 0, "input_pct": 0.0, "cache_hit_pct": 0.0}

    async def test_split_with_input_pct_and_cache_hit(self) -> None:
        # 800 in, 200 out, 80 cached (80/800 = 10% of input was a cache hit).
        row = _row(input_tokens=800, output_tokens=200, cached_tokens=80, total_tokens=1000)
        run = {"thread_id": "api-r1", "run_id": "r1"}
        split = await data.run_token_split(_session_one_returning(row), run)
        assert split["input_tokens"] == 800
        assert split["output_tokens"] == 200
        assert split["cached_tokens"] == 80
        assert split["total_tokens"] == 1000
        # input_pct = 800 / (800+200) = 80.0 ; cache_hit_pct = 80 / 800 = 10.0
        assert split["input_pct"] == 80.0
        assert split["cache_hit_pct"] == 10.0

    async def test_empty_when_no_keys(self) -> None:
        assert await data.run_token_split(MagicMock(), {}) == self._ZEROS

    async def test_degrades_to_zeros_on_db_error(self) -> None:
        run = {"thread_id": "api-r1", "run_id": "r1"}
        assert await data.run_token_split(_session_raising(RuntimeError("db")), run) == self._ZEROS

    async def test_zero_input_does_not_divide_by_zero(self) -> None:
        # A run with no attributed input tokens → both ratios stay 0.0.
        row = _row(input_tokens=0, output_tokens=0, cached_tokens=0, total_tokens=0)
        run = {"thread_id": "api-r1", "run_id": "r1"}
        split = await data.run_token_split(_session_one_returning(row), run)
        assert split["cache_hit_pct"] == 0.0
        assert split["input_pct"] == 0.0


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
# Data layer — sub-agents / counts / evolution / web-search (mutations page)
# ---------------------------------------------------------------------------


class _Result:
    """Fake ``execute()`` result supporting ``.all()`` / ``.scalar()`` / ``.one()``
    / ``.scalars().all()`` so one helper fakes any access pattern."""

    def __init__(
        self,
        *,
        rows: Any = None,
        scalar: Any = None,
        one: Any = None,
        scalars: Any = None,
    ) -> None:
        self._rows = rows if rows is not None else []
        self._scalar = scalar
        self._one = one
        self._scalars = scalars if scalars is not None else []

    def all(self) -> Any:
        return self._rows

    def scalar(self) -> Any:
        return self._scalar

    def one(self) -> Any:
        return self._one

    def scalars(self) -> Any:
        return SimpleNamespace(
            all=lambda: self._scalars,
            first=lambda: (self._scalars[0] if self._scalars else None),
        )


def _seq_session(results: list[Any]) -> MagicMock:
    """An AsyncSession whose successive ``execute()`` calls return ``results`` in
    order (for fns that issue several queries — e.g. ``mutation_counts``)."""
    session = MagicMock()
    session.execute = AsyncMock(side_effect=list(results))
    session.rollback = AsyncMock()
    return session


@pytest.mark.asyncio
class TestSubAgentTimeline:
    @staticmethod
    def _rows() -> list[SimpleNamespace]:
        return [
            _row(
                id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
                name="researcher", description="web research specialist",
                model_tier="complex", is_active=True, source_mutation_id=None,
                owner_run_id="run-web", total_runs=5, success_rate=0.8,
                quality_score=0.7, created_at=datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc),
            ),
            _row(
                id=uuid.UUID("22222222-2222-2222-2222-222222222222"),
                name="coder", description="code writer", model_tier="simple",
                is_active=False, source_mutation_id=None, owner_run_id=None,
                total_runs=0, success_rate=0.0, quality_score=0.5,
                created_at=datetime(2026, 7, 14, 9, 0, tzinfo=timezone.utc),
            ),
        ]

    async def test_shapes_rows_with_active_flag_and_owner(self) -> None:
        rows = await data.sub_agent_timeline(_session_returning(self._rows()))
        assert len(rows) == 2
        first = rows[0]
        assert first["name"] == "researcher"
        assert first["is_active"] is True
        assert first["owner_run_id"] == "run-web"
        assert first["success_rate"] == 0.8
        assert first["id"] == "11111111-1111-1111-1111-111111111111"
        assert rows[1]["is_active"] is False
        assert rows[1]["owner_run_id"] is None

    async def test_degrades_to_empty_on_db_error(self) -> None:
        assert await data.sub_agent_timeline(_session_raising(RuntimeError("db"))) == []


@pytest.mark.asyncio
class TestMutationCounts:
    async def test_groups_by_type_and_rolls_up_prompt_tool(self) -> None:
        # 2 prompt + 1 tool + 1 sub_agent_tools; 6 deployed tools, 0 subagents,
        # 1 active config. Order = group-by, deployed, subagents, configs.
        seq = _seq_session([
            _Result(rows=[
                _row(mtype="prompt", n=2),
                _row(mtype="tool", n=1),
                _row(mtype="sub_agent_tools", n=1),
            ]),
            _Result(scalar=6),
            _Result(scalar=0),
            _Result(scalar=1),
        ])
        counts = await data.mutation_counts(seq)
        assert counts["by_type"] == {"prompt": 2, "tool": 1, "sub_agent_tools": 1}
        assert counts["prompts_mutated"] == 2  # prompt only (sub_agent_prompt absent)
        assert counts["tools_mutated"] == 2     # tool + sub_agent_tools
        assert counts["total_mutations"] == 4
        assert counts["deployed_tools"] == 6
        assert counts["active_subagents"] == 0
        assert counts["active_configs"] == 1

    async def test_sub_agent_prompt_counts_as_prompt(self) -> None:
        seq = _seq_session([
            _Result(rows=[_row(mtype="sub_agent_prompt", n=3)]),
            _Result(scalar=0), _Result(scalar=0), _Result(scalar=0),
        ])
        counts = await data.mutation_counts(seq)
        assert counts["prompts_mutated"] == 3

    async def test_a_failed_groupby_does_not_zero_the_scalars(self) -> None:
        # The group-by raises, but the three scalar counts still resolve.
        session = MagicMock()
        session.execute = AsyncMock(side_effect=[
            RuntimeError("group-by down"),
            _Result(scalar=4), _Result(scalar=2), _Result(scalar=1),
        ])
        counts = await data.mutation_counts(session)
        assert counts["by_type"] == {}
        assert counts["deployed_tools"] == 4
        assert counts["active_subagents"] == 2


@pytest.mark.asyncio
class TestEvolutionSummary:
    @staticmethod
    def _counts(**kw: Any) -> dict[str, Any]:
        base: dict[str, Any] = {"deployed_tools": 0, "active_subagents": 0, "active_configs": 0}
        base.update(kw)
        return base

    async def test_live_promotion_marks_evolved(self) -> None:
        fake_gate = MagicMock()
        fake_gate.active_promotions = MagicMock(return_value=[
            {"node": "execute", "active": "execute.abc123.txt",
             "canary_score": 1.0, "promoted_at": "2026-07-15T10:00:00+00:00"},
        ])
        with patch("src.evolution.promote.PromotionGate", return_value=fake_gate):
            ev = await data.evolution_summary(MagicMock(), self._counts())
        assert ev["total_live_promotions"] == 1
        assert ev["any_evolved"] is True
        assert ev["live_prompt_promotions"][0]["node"] == "execute"

    async def test_channel_a_tools_mark_evolved_without_promotions(self) -> None:
        fake_gate = MagicMock()
        fake_gate.active_promotions = MagicMock(return_value=[])
        with patch("src.evolution.promote.PromotionGate", return_value=fake_gate):
            ev = await data.evolution_summary(MagicMock(), self._counts(deployed_tools=6))
        assert ev["total_live_promotions"] == 0
        assert ev["any_evolved"] is True  # channel-A deployed tools

    async def test_degrades_when_promotion_read_raises(self) -> None:
        with patch("src.evolution.promote.PromotionGate", side_effect=RuntimeError("fs")):
            ev = await data.evolution_summary(MagicMock(), self._counts())
        assert ev["live_prompt_promotions"] == []
        assert ev["total_live_promotions"] == 0
        assert ev["any_evolved"] is False


@pytest.mark.asyncio
class TestWebSearchSummary:
    async def test_total_calls_and_distinct_runs(self) -> None:
        row = _row(calls=10, runs=3)
        summary = await data.web_search_summary(_session_one_returning(row))
        assert summary == {"total_calls": 10, "runs_using_search": 3}

    async def test_degrades_to_zeros_on_db_error(self) -> None:
        zeros = {"total_calls": 0, "runs_using_search": 0}
        assert await data.web_search_summary(_session_raising(RuntimeError("db"))) == zeros


@pytest.mark.asyncio
class TestWebSearchRuns:
    async def test_returns_set_of_runs_that_used_web_search(self) -> None:
        res = _Result(scalars=["run-web", "run-other"])
        out = await data.web_search_runs(_seq_session([res]), ["run-web", "run-silent"])
        assert out == {"run-web", "run-other"}

    async def test_empty_input_short_circuits_without_query(self) -> None:
        session = MagicMock()
        session.execute = AsyncMock()
        assert await data.web_search_runs(session, []) == set()
        assert await data.web_search_runs(session, [None, ""]) == set()  # filtered
        session.execute.assert_not_awaited()

    async def test_degrades_to_empty_on_db_error(self) -> None:
        assert await data.web_search_runs(_session_raising(RuntimeError("db")), ["r1"]) == set()


@pytest.mark.asyncio
class TestMutationsWebSearch:
    _TOOL_MUT = "11111111-1111-1111-1111-111111111111"
    _PROMPT_MUT = "22222222-2222-2222-2222-222222222222"

    async def test_tool_mutation_run_using_web_search(self) -> None:
        # tool_registrations maps the TOOL mutation → run "run-web";
        # sub_agent_definitions empty; web_search used by "run-web". The
        # attribution SELECT yields 2-column Row tuples (smid, orid).
        seq = _seq_session([
            _Result(rows=[(uuid.UUID(self._TOOL_MUT), "run-web")]),
            _Result(rows=[]),
            _Result(scalars=["run-web"]),
        ])
        out = await data.mutations_web_search(seq, [self._TOOL_MUT, self._PROMPT_MUT])
        assert out[self._TOOL_MUT] is True
        assert self._PROMPT_MUT not in out  # PROMPT mutation has no run link

    async def test_tool_mutation_run_without_web_search_is_false(self) -> None:
        seq = _seq_session([
            _Result(rows=[(uuid.UUID(self._TOOL_MUT), "run-silent")]),
            _Result(rows=[]),
            _Result(scalars=[]),
        ])
        out = await data.mutations_web_search(seq, [self._TOOL_MUT])
        assert out[self._TOOL_MUT] is False

    async def test_no_n_plus_1_constant_queries_regardless_of_count(self) -> None:
        # Two mutations still cost 3 queries (tool + subagent + web-search),
        # never one-per-mutation.
        seq = _seq_session([
            _Result(rows=[(uuid.UUID(self._TOOL_MUT), "run-a")]),
            _Result(rows=[]),
            _Result(scalars=["run-a"]),
        ])
        await data.mutations_web_search(seq, [self._TOOL_MUT, self._PROMPT_MUT])
        assert seq.execute.await_count == 3

    async def test_empty_input_short_circuits(self) -> None:
        session = MagicMock()
        session.execute = AsyncMock()
        assert await data.mutations_web_search(session, []) == {}
        session.execute.assert_not_awaited()


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


@pytest.mark.asyncio
class TestCascadeResilience:
    """Regression for the silent ``web_search = 0`` bug.

    A best-effort dashboard query that hits a real DB error (e.g. a column that
    drifted out of sync with the ORM — the live ``agent_config_versions.is_active``
    unrun-migration case) aborts the shared PostgreSQL transaction. Pre-fix the
    per-fn ``try/except`` swallowed the error WITHOUT rolling back, so every
    LATER query on the same session cascaded to ``InFailedSQLTransactionError``
    and degraded to 0/empty — which is exactly how ``web_search`` rendered 0
    despite 132 calls existing. Each best-effort fn must now ROLLBACK so a
    sibling query afterward still gets its real data.
    """

    @staticmethod
    def _session_fail_then_ok(fail_exc: BaseException, ok_row: Any) -> MagicMock:
        """``execute()`` raises once (the drifted-column query) then returns
        ``ok_row`` via ``.one()``; ``rollback`` is awaitable (the fns call it)."""
        session = MagicMock()
        session.execute = AsyncMock(
            side_effect=[fail_exc, MagicMock(one=lambda: ok_row)]
        )
        session.rollback = AsyncMock()
        return session

    async def test_scalar_count_rollbacks_and_returns_zero(self) -> None:
        session = self._session_fail_then_ok(
            RuntimeError("column agent_config_versions.is_active does not exist"),
            SimpleNamespace(calls=9, runs=1),
        )
        assert await data._scalar_count(session, object(), "active-config count") == 0
        session.rollback.assert_awaited_once()

    async def test_sibling_query_not_poisoned_after_failure(self) -> None:
        # The original symptom: a DIFFERENT best-effort fn ran AFTER the failing
        # query and got cascade-zeroed. With rollback it now gets real data.
        session = self._session_fail_then_ok(
            RuntimeError("column agent_config_versions.is_active does not exist"),
            SimpleNamespace(calls=132, runs=1),
        )
        # call #1 — fails (would abort the txn in real Postgres); rolled back
        assert await data._scalar_count(session, object(), "active-config count") == 0
        # call #2 — a sibling fn — must still see its REAL result, not 0
        assert await data.web_search_summary(session) == {
            "total_calls": 132,
            "runs_using_search": 1,
        }

    async def test_safe_rollback_never_propagates(self) -> None:
        session = MagicMock()
        session.rollback = AsyncMock(side_effect=RuntimeError("rollback itself failed"))
        await data._safe_rollback(session)  # must not raise
        session.rollback.assert_awaited_once()


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
        with patch.object(data,"run_cost_breakdown", new=AsyncMock(return_value=breakdown)), \
             patch.object(data, "run_token_split",
                          new=AsyncMock(return_value=TestRunTokenSplit._ZEROS)):
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
        # 404 now requires BOTH sources to miss: a Redis miss alone falls through
        # to the archived reconstruction from cost_ledger (see test_dashboard_historical_runs).
        with patch.object(data, "_archived_run_view_for", new=AsyncMock(return_value=None)):
            resp = client.get("/dashboard/runs/does-not-exist")
        assert resp.status_code == 404

    def test_detail_renders_live_steps_section(
        self, client: TestClient, patch_infra: Any
    ) -> None:
        store, _ = patch_infra
        store.get = AsyncMock(return_value=_run("r1", status=JobStatus.RUNNING))
        steps = [{"seq": 1, "phase": "execute_node", "status": "completed",
                  "duration_ms": 22500, "duration_s": 22.5,
                  "created_at": "2026-07-15T12:00:05+00:00"}]
        with patch.object(data, "run_cost_breakdown", new=AsyncMock(return_value=[])), \
             patch.object(data, "execution_steps", new=AsyncMock(return_value=steps)), \
             patch.object(data, "run_token_split",
                          new=AsyncMock(return_value=TestRunTokenSplit._ZEROS)):
            resp = client.get("/dashboard/runs/r1")
        assert resp.status_code == 200, resp.text
        body = resp.text
        assert "Execution steps" in body  # the section header
        assert "execute_node" in body  # a step row rendered server-side (initial render)
        # The polled-swap wiring: the tbody auto-refreshes the steps partial.
        assert 'data-poll="/dashboard/runs/r1/steps?partial=1"' in body

    def test_detail_renders_token_mix(
        self, client: TestClient, patch_infra: Any
    ) -> None:
        store, _ = patch_infra
        store.get = AsyncMock(return_value=_run("r1", status=JobStatus.COMPLETED))
        # 80% of tokens input, 10% of input a cache hit (the autonomous-agent
        # overhead signature the Q5 token-mix line surfaces).
        split = {"input_tokens": 800, "output_tokens": 200, "cached_tokens": 80,
                 "total_tokens": 1000, "input_pct": 80.0, "cache_hit_pct": 10.0}
        with patch.object(data, "run_cost_breakdown", new=AsyncMock(return_value=[])), \
             patch.object(data, "execution_steps", new=AsyncMock(return_value=[])), \
             patch.object(data, "run_token_split", new=AsyncMock(return_value=split)):
            resp = client.get("/dashboard/runs/r1")
        assert resp.status_code == 200, resp.text
        body = resp.text
        assert "Token mix" in body
        assert "800" in body and "200" in body  # input · output counts
        assert "80.0% of tokens are input" in body  # input_pct
        assert "10.0% cache-hit" in body  # cache_hit_pct


class TestRunStepsRoute:
    def test_partial_renders_step_rows_fragment(
        self, client: TestClient, patch_infra: Any
    ) -> None:
        store, _ = patch_infra
        store.get = AsyncMock(return_value=_run("r1", status=JobStatus.RUNNING))
        steps = [{"seq": 1, "phase": "verify_node", "status": "failed",
                  "duration_ms": 3000, "duration_s": 3.0,
                  "created_at": "2026-07-15T12:00:10+00:00"}]
        with patch.object(data, "execution_steps", new=AsyncMock(return_value=steps)):
            resp = client.get("/dashboard/runs/r1/steps?partial=1")
        assert resp.status_code == 200, resp.text
        body = resp.text
        assert "<html" not in body  # fragment only — not a full document
        assert "verify_node" in body
        assert "failed" in body
        assert "3.0s" in body  # the formatted duration cell

    def test_partial_renders_empty_placeholder(
        self, client: TestClient, patch_infra: Any
    ) -> None:
        store, _ = patch_infra
        store.get = AsyncMock(return_value=_run("r1", status=JobStatus.QUEUED))
        with patch.object(data, "execution_steps", new=AsyncMock(return_value=[])):
            resp = client.get("/dashboard/runs/r1/steps?partial=1")
        assert resp.status_code == 200
        assert "No execution steps recorded" in resp.text

    def test_unknown_run_degrades_to_empty_partial_not_404(
        self, client: TestClient, patch_infra: Any
    ) -> None:
        # A polled partial must never 500/404 mid-page: a run unknown to BOTH
        # Redis and Postgres renders the empty placeholder so the run-detail page
        # keeps polling cleanly.
        store, _ = patch_infra
        store.get = AsyncMock(return_value=None)
        with patch.object(data, "_archived_run_view_for", new=AsyncMock(return_value=None)), \
             patch.object(data, "execution_steps", new=AsyncMock(return_value=[])) as es:
            resp = client.get("/dashboard/runs/expired/steps?partial=1")
        assert resp.status_code == 200
        assert "No execution steps recorded" in resp.text
        es.assert_not_awaited()  # no run view (Redis + Postgres miss) → no DB hit

    def test_archived_run_steps_partial_returns_rows_not_empty(
        self, client: TestClient, patch_infra: Any
    ) -> None:
        # Regression: the polled steps partial used to derive the run view from
        # Redis only, so an archived run's steps vanished ~3s after load (poll.js
        # cleared the tbody). It must now reconstruct the archived view — mirroring
        # the detail route — so an archived run's steps survive the first poll.
        store, _ = patch_infra
        store.get = AsyncMock(return_value=None)  # Redis forgot this run
        archived = {
            "run_id": "old1", "thread_id": "api-old1", "status": "archived",
            "final_output": None, "is_complete": None, "iteration_count": None,
            "error": None, "started_at": "2026-07-01T00:00:00",
            "finished_at": "2026-07-01T01:00:00", "results_dir": None,
        }
        steps = [{"seq": 1, "phase": "execute_node", "status": "completed",
                  "duration_ms": 22500, "duration_s": 22.5,
                  "created_at": "2026-07-01T00:05:00+00:00"}]
        with patch.object(data, "_archived_run_view_for", new=AsyncMock(return_value=archived)), \
             patch.object(data, "execution_steps", new=AsyncMock(return_value=steps)):
            resp = client.get("/dashboard/runs/old1/steps?partial=1")
        assert resp.status_code == 200, resp.text
        body = resp.text
        assert "execute_node" in body               # rows present (not the empty placeholder)
        assert "22.5s" in body                       # the formatted duration cell
        assert "No execution steps recorded" not in body


class TestToolsRoute:
    def test_renders_tool_health_table(
        self, client: TestClient, patch_infra: Any
    ) -> None:
        # Regression (the /dashboard/tools 500): tool_health returns a RAW
        # datetime for created_at (not an iso string like the run views) and None
        # rates for never-called tools. Both must render, not crash the template.
        tools = [
            {
                "tool_name": "web_search", "tool_type": "builtin", "is_active": True,
                "calls": 10, "success_rate": 0.9, "empty_output_rate": 0.1,
                "avg_latency_ms": 120.0,
                "created_at": datetime(2026, 7, 1, 9, 0, tzinfo=timezone.utc),
            },
            {
                "tool_name": "unused_tool", "tool_type": "dynamic", "is_active": False,
                "calls": 0, "success_rate": None, "empty_output_rate": None,
                "avg_latency_ms": None, "created_at": None,
            },
        ]
        with patch.object(data, "tool_health", new=AsyncMock(return_value=tools)):
            resp = client.get("/dashboard/tools")
        assert resp.status_code == 200, resp.text
        body = resp.text
        assert "web_search" in body
        assert "90%" in body                  # success rate formatted (not the 500)
        assert "2026-07-01 09:00" in body     # created_at is a datetime → strftime
        assert "unused_tool" in body
        assert "—" in body                    # None rate renders the em dash
        assert "retired" in body             # is_active=False badge

    def test_empty_renders_placeholder(
        self, client: TestClient, patch_infra: Any
    ) -> None:
        with patch.object(data, "tool_health", new=AsyncMock(return_value=[])):
            resp = client.get("/dashboard/tools")
        assert resp.status_code == 200
        assert "No tools registered" in resp.text


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


_DEFAULT_COUNTS: dict[str, Any] = {
    "by_type": {"prompt": 0, "tool": 0}, "prompts_mutated": 0, "tools_mutated": 0,
    "total_mutations": 0, "deployed_tools": 0, "active_subagents": 0, "active_configs": 0,
}
_DEFAULT_EVOLUTION: dict[str, Any] = {
    "live_prompt_promotions": [], "total_live_promotions": 0,
    "deployed_tools": 0, "active_subagents": 0, "active_configs": 0, "any_evolved": False,
}
_DEFAULT_WEB: dict[str, Any] = {"total_calls": 0, "runs_using_search": 0}


@contextmanager
def _patch_mutations_data(**overrides: Any) -> Generator[None, None, None]:
    """Patch all 7 mutations-page data fns at once.

    The route isolates at the data boundary (call fns → render their dicts), so
    a route test must stub every fn the route awaits — otherwise an unpatched
    real fn runs against the fake session, raises, and the route's try/except
    resets ``mutations=[]`` (the silent-failure trap the old test fell into).
    Per-test ``overrides`` replace specific fns; the rest get sane empty
    defaults matching the dict shapes the template reads.
    """
    mocks: dict[str, Any] = {
        "mutation_timeline": AsyncMock(return_value=[]),
        "sub_agent_timeline": AsyncMock(return_value=[]),
        "mutation_counts": AsyncMock(return_value=dict(_DEFAULT_COUNTS)),
        "evolution_summary": AsyncMock(return_value=dict(_DEFAULT_EVOLUTION)),
        "web_search_summary": AsyncMock(return_value=dict(_DEFAULT_WEB)),
        "mutations_web_search": AsyncMock(return_value={}),
        "web_search_runs": AsyncMock(return_value=set()),
    }
    mocks.update(overrides)
    with ExitStack() as stack:
        for name, mock in mocks.items():
            stack.enter_context(patch.object(data, name, new=mock))
        yield None


class TestMutationsRoute:
    def test_renders_timeline_with_diff(self, client: TestClient, patch_infra: Any) -> None:
        mutations = [{
            "id": "mut-1", "mutation_type": "PROMPT", "target_path": "prompts/x.md",
            "description": "d", "status": "deployed", "has_diff": True, "diff_content": "- a\n+ b",
            "model_used": "glm-5.2", "created_at": "2026-07-15T12:00:00+00:00",
            "p_value": 0.04, "is_significant": True, "confidence": 0.95, "sample_size": 3,
            "control_value": 0.7, "treatment_value": 0.9,
        }]
        with _patch_mutations_data(mutation_timeline=AsyncMock(return_value=mutations)):
            resp = client.get("/dashboard/mutations")
        assert resp.status_code == 200, resp.text
        body = resp.text
        assert "Mutation timeline" in body
        assert "PROMPT" in body  # type badge
        assert "- a" in body  # diff surfaced in the <details>

    def test_empty_renders_placeholder(self, client: TestClient, patch_infra: Any) -> None:
        with _patch_mutations_data():
            resp = client.get("/dashboard/mutations")
        assert resp.status_code == 200
        body = resp.text
        assert "No mutations yet" in body  # empty mutations table
        assert "prompts mutated" in body  # summary cards still render
        assert "No live PROMPT promotions" in body  # evolution empty-state

    def test_renders_summary_cards_and_evolution_status(
        self, client: TestClient, patch_infra: Any
    ) -> None:
        counts = {**_DEFAULT_COUNTS, "prompts_mutated": 2, "tools_mutated": 3, "deployed_tools": 6}
        evolution = {
            **_DEFAULT_EVOLUTION, "any_evolved": True, "total_live_promotions": 1,
            "live_prompt_promotions": [{
                "node": "execute", "active": "execute.abc123.txt",
                "canary_score": 1.0, "promoted_at": "2026-07-15T10:00:00+00:00",
            }],
        }
        with _patch_mutations_data(
            mutation_counts=AsyncMock(return_value=counts),
            evolution_summary=AsyncMock(return_value=evolution),
        ):
            resp = client.get("/dashboard/mutations")
        assert resp.status_code == 200, resp.text
        body = resp.text
        assert "prompts mutated" in body
        assert "tools mutated" in body
        assert "app evolved" in body  # any_evolved chip
        assert "execute.abc123.txt" in body  # live promotion row rendered
        assert "No live PROMPT promotions" not in body  # section NOT in empty-state

    def test_renders_subagents_section(self, client: TestClient, patch_infra: Any) -> None:
        subagents = [{
            "id": "sa-1", "name": "researcher", "description": "web research specialist",
            "model_tier": "complex", "is_active": True, "source_mutation_id": None,
            "owner_run_id": "run-web", "total_runs": 5, "success_rate": 0.8,
            "quality_score": 0.7, "created_at": "2026-07-15T12:00:00+00:00",
        }]
        with _patch_mutations_data(
            sub_agent_timeline=AsyncMock(return_value=subagents),
            web_search_runs=AsyncMock(return_value={"run-web"}),  # → researcher web badge ✓
        ):
            resp = client.get("/dashboard/mutations")
        assert resp.status_code == 200, resp.text
        body = resp.text
        assert "Sub-agents spawned" in body
        assert "researcher" in body
        assert "complex" in body  # model-tier badge
        assert "run-web" in body  # owner run column
