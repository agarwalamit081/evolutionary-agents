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
