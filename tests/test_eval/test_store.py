"""EvalStore (Phase 3): write/query round-trip + the non-fatal guarantee.

Uses an in-memory fake session (the store is observability-only and must never
raise), mirroring the CostTracker-resilience pattern — a poisoned write logs and
returns 0 rather than aborting the caller.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from sqlalchemy.dialects import postgresql

from src.eval.models import CheckResult, CorrectnessResult
from src.eval.store import EvalStore


class _FakeResult:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def scalars(self) -> _FakeResult:
        return self

    def all(self) -> list[Any]:
        return self._rows


class _FakeSession:
    """In-memory AsyncSession: records adds, replays them on execute()."""

    def __init__(self) -> None:
        self.added: list[Any] = []

    def add(self, obj: Any) -> None:
        self.added.append(obj)

    async def execute(self, _stmt: Any) -> _FakeResult:
        return _FakeResult(list(self.added))

    async def commit(self) -> None:
        return None

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *_args: Any) -> bool:
        return False


class _RaisingCM:
    """Async context manager whose __aenter__ raises (poisoned session)."""

    async def __aenter__(self) -> Any:
        raise RuntimeError("connection refused")

    async def __aexit__(self, *_args: Any) -> bool:
        return False


class _StatementCapturingSession:
    """Async session that records each executed statement and yields no rows.

    Used to assert on the compiled SQL a store method emits (the JSONB columns
    block a real aiosqlite round-trip), mirroring test_warm_facts.py.
    """

    def __init__(self) -> None:
        self.executed: list[Any] = []

    async def execute(self, stmt: Any) -> _FakeResult:
        self.executed.append(stmt)
        return _FakeResult([])

    async def __aenter__(self) -> _StatementCapturingSession:
        return self

    async def __aexit__(self, *_args: Any) -> bool:
        return False


def _enabled_settings(store: bool = True) -> object:
    return SimpleNamespace(eval=SimpleNamespace(eval_store_enabled=store))


def _correctness(n: int = 2) -> CorrectnessResult:
    return CorrectnessResult(
        spec_id="battery04_q01",
        overall_score=0.75,
        passed=False,
        checks=[
            CheckResult(
                check_name=f"check_{i}",
                check_type="structural",
                passed=(i == 0),
                score=1.0 if i == 0 else 0.0,
                evidence={"i": i},
            )
            for i in range(n)
        ],
    )


class TestEvalStoreRecord:
    @pytest.mark.asyncio
    async def test_writes_one_row_per_check(self) -> None:
        session = _FakeSession()
        with patch("src.config.settings.get_settings", lambda: _enabled_settings(True)), patch(
            "src.db.session.get_session", lambda: session
        ):
            written = await EvalStore().record_correctness(
                _correctness(2), goal_id="battery04_q01", run_id="run-1"
            )
        assert written == 2
        assert len(session.added) == 2
        row = session.added[0]
        assert row.goal_id == "battery04_q01"
        assert row.run_id == "run-1"
        assert row.check_name == "check_0"
        assert row.passed is True
        assert float(row.score) == 1.0

    @pytest.mark.asyncio
    async def test_disabled_returns_zero_and_writes_nothing(self) -> None:
        session = _FakeSession()
        with patch("src.config.settings.get_settings", lambda: _enabled_settings(False)), patch(
            "src.db.session.get_session", lambda: session
        ):
            written = await EvalStore().record_correctness(
                _correctness(2), goal_id="g", run_id="r"
            )
        assert written == 0
        assert session.added == []

    @pytest.mark.asyncio
    async def test_no_checks_returns_zero(self) -> None:
        empty = CorrectnessResult(spec_id="x", overall_score=1.0, passed=True, checks=[])
        with patch("src.config.settings.get_settings", lambda: _enabled_settings(True)):
            written = await EvalStore().record_correctness(empty, goal_id="g", run_id="r")
        assert written == 0

    @pytest.mark.asyncio
    async def test_db_failure_is_swallowed(self) -> None:
        with patch("src.config.settings.get_settings", lambda: _enabled_settings(True)), patch(
            "src.db.session.get_session", lambda: _RaisingCM()
        ):
            written = await EvalStore().record_correctness(
                _correctness(1), goal_id="g", run_id="r"
            )
        assert written == 0  # non-fatal

    @pytest.mark.asyncio
    async def test_producer_model_is_stamped_on_rows(self) -> None:
        # Phase-2 attribution: producer_model threads record_correctness ->
        # _store_check -> EvalResult so the curve can slice per-model.
        session = _FakeSession()
        with patch("src.config.settings.get_settings", lambda: _enabled_settings(True)), patch(
            "src.db.session.get_session", lambda: session
        ):
            written = await EvalStore().record_correctness(
                _correctness(2),
                goal_id="battery04_q01",
                run_id="run-1",
                producer_model="glm-4.7",
            )
        assert written == 2
        assert all(r.producer_model == "glm-4.7" for r in session.added)

    @pytest.mark.asyncio
    async def test_producer_model_defaults_to_none(self) -> None:
        # Nullable/additive: an unattributed write leaves the column NULL so
        # legacy callers (and the migration's backfill-free old rows) stay valid.
        session = _FakeSession()
        with patch("src.config.settings.get_settings", lambda: _enabled_settings(True)), patch(
            "src.db.session.get_session", lambda: session
        ):
            await EvalStore().record_correctness(_correctness(1), goal_id="g", run_id="r")
        assert session.added[0].producer_model is None


class TestEvalStoreQuery:
    @pytest.mark.asyncio
    async def test_query_by_run_returns_dicts(self) -> None:
        session = _FakeSession()
        with patch("src.config.settings.get_settings", lambda: _enabled_settings(True)), patch(
            "src.db.session.get_session", lambda: session
        ):
            store = EvalStore()
            await store.record_correctness(_correctness(2), goal_id="g", run_id="run-9")
            rows = await store.query_by_run("run-9")
        assert len(rows) == 2
        assert all(r["run_id"] == "run-9" for r in rows)
        assert {"passed", "score", "check_name"} <= rows[0].keys()

    @pytest.mark.asyncio
    async def test_query_failure_returns_empty(self) -> None:
        with patch("src.db.session.get_session", lambda: _RaisingCM()):
            rows = await EvalStore().query_by_run("anything")
        assert rows == []


class _ScalarRowsResult:
    """execute() result exposing both a scalar and a rows view.

    ``query_latest_attempt`` issues two executes — a scalar (newest attempt_id)
    then a rows fetch — so this result carries both a configured scalar and rows
    so the control flow is exercised against the in-memory fake.
    """

    def __init__(self, scalar: Any, rows: list[Any]) -> None:
        self._scalar = scalar
        self._rows = rows

    def scalar_one_or_none(self) -> Any:
        return self._scalar

    def scalars(self) -> _ScalarRowsResult:
        return self

    def all(self) -> list[Any]:
        return self._rows


class _LatestAttemptSession:
    """Replays a configured sequence of execute() results in order."""

    def __init__(self, results: list[Any]) -> None:
        self._results = results
        self._i = 0

    async def execute(self, _stmt: Any) -> Any:
        result = self._results[self._i]
        self._i += 1
        return result

    async def __aenter__(self) -> _LatestAttemptSession:
        return self

    async def __aexit__(self, *_args: Any) -> bool:
        return False


def _attempt_row(attempt_id: str, check_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        goal_id="g",
        run_id="run-9",
        attempt_id=attempt_id,
        spec_id=None,
        check_name=check_name,
        check_type="structural",
        passed=True,
        score=1.0,
        skipped=False,
        evidence=None,
        cost_usd=0.0,
        producer_model=None,
        created_at=None,
    )


class TestEvalStoreAttemptId:
    """Per-run-attempt scoring: attempt_id is written + isolates the newest run."""

    @pytest.mark.asyncio
    async def test_writes_attempt_id_on_each_row(self) -> None:
        session = _FakeSession()
        with patch("src.config.settings.get_settings", lambda: _enabled_settings(True)), patch(
            "src.db.session.get_session", lambda: session
        ):
            written = await EvalStore().record_correctness(
                _correctness(2),
                goal_id="g",
                run_id="run-1",
                attempt_id="20260619120000-abc12345",
            )
        assert written == 2
        assert all(r.attempt_id == "20260619120000-abc12345" for r in session.added)

    @pytest.mark.asyncio
    async def test_attempt_id_defaults_none(self) -> None:
        # Back-compat: omitting attempt_id writes NULL (legacy rows stay valid).
        session = _FakeSession()
        with patch("src.config.settings.get_settings", lambda: _enabled_settings(True)), patch(
            "src.db.session.get_session", lambda: session
        ):
            await EvalStore().record_correctness(_correctness(1), goal_id="g", run_id="r")
        assert session.added[0].attempt_id is None

    @pytest.mark.asyncio
    async def test_query_latest_attempt_returns_only_newest(self) -> None:
        # run-9 has two attempts; the scalar query resolves the newest, then the
        # rows fetch returns ONLY that attempt's rows (not a blend of both).
        newest = "20260619120000-new1"
        row_old = _attempt_row("20260619110000-old1", "c_old")
        row_new = _attempt_row(newest, "c_new")
        session = _LatestAttemptSession(
            [_ScalarRowsResult(scalar=newest, rows=[row_old, row_new]),
             _ScalarRowsResult(scalar=newest, rows=[row_new])]
        )
        with patch("src.db.session.get_session", lambda: session):
            rows = await EvalStore().query_latest_attempt("run-9")
        assert len(rows) == 1
        assert rows[0]["attempt_id"] == newest
        assert rows[0]["check_name"] == "c_new"

    @pytest.mark.asyncio
    async def test_query_latest_attempt_empty_when_no_attempts(self) -> None:
        # Legacy rows (attempt_id NULL) → the scalar query returns None → no rows.
        session = _LatestAttemptSession([_ScalarRowsResult(scalar=None, rows=[])])
        with patch("src.db.session.get_session", lambda: session):
            rows = await EvalStore().query_latest_attempt("run-9")
        assert rows == []

    @pytest.mark.asyncio
    async def test_query_latest_attempt_failure_returns_empty(self) -> None:
        with patch("src.db.session.get_session", lambda: _RaisingCM()):
            rows = await EvalStore().query_latest_attempt("anything")
        assert rows == []


class TestEvalStoreFetchRows:
    """``fetch_rows`` emits the right goal/window/order/limit SELECT.

    The eval_results ``evidence`` column is JSONB, which blocks a real aiosqlite
    round-trip (the SQLite compiler can't render JSONB). Following the
    test_warm_facts.py convention, the contract is asserted on the compiled
    statement that actually runs against Postgres — goal_id IN (...), the
    created_at window, ORDER BY created_at DESC, and LIMIT. Row-shape/resilience
    is covered by the fake-session tests above; the curve's data shaping is
    covered by test_curve.py.
    """

    @pytest.mark.asyncio
    async def test_filters_by_goal_orders_newest_first_and_limits(self) -> None:
        session = _StatementCapturingSession()
        with patch("src.db.session.get_session", lambda: session):
            await EvalStore().fetch_rows(["battery04_q01", "battery04_q02"], limit=5)

        assert session.executed, "fetch_rows must execute a SELECT"
        sql = str(
            session.executed[0].compile(
                dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
            )
        )
        assert "FROM eval_results" in sql
        assert "goal_id IN ('battery04_q01', 'battery04_q02')" in sql  # goal filter
        assert "ORDER BY eval_results.created_at DESC" in sql  # newest-first
        assert "LIMIT 5" in sql  # row cap honored

    @pytest.mark.asyncio
    async def test_window_until_emits_created_at_bound(self) -> None:
        from datetime import datetime, timezone

        session = _StatementCapturingSession()
        with patch("src.db.session.get_session", lambda: session):
            await EvalStore().fetch_rows(
                ["battery04_q01"],
                until=datetime(2026, 6, 2, tzinfo=timezone.utc),
            )
        sql = str(session.executed[0].compile(dialect=postgresql.dialect()))
        assert "eval_results.created_at <=" in sql  # window upper bound applied

    @pytest.mark.asyncio
    async def test_window_since_emits_created_at_bound(self) -> None:
        from datetime import datetime, timezone

        session = _StatementCapturingSession()
        with patch("src.db.session.get_session", lambda: session):
            await EvalStore().fetch_rows(
                ["battery04_q01"],
                since=datetime(2026, 6, 1, tzinfo=timezone.utc),
            )
        sql = str(session.executed[0].compile(dialect=postgresql.dialect()))
        assert "eval_results.created_at >=" in sql

    @pytest.mark.asyncio
    async def test_empty_goal_list_returns_empty_without_query(self) -> None:
        session = _StatementCapturingSession()
        with patch("src.db.session.get_session", lambda: session):
            rows = await EvalStore().fetch_rows([])
        assert rows == []
        assert not session.executed  # short-circuit: no SELECT for an empty goal set

    @pytest.mark.asyncio
    async def test_failure_returns_empty(self) -> None:
        with patch("src.db.session.get_session", lambda: _RaisingCM()):
            rows = await EvalStore().fetch_rows(["battery04_q01"])
        assert rows == []

