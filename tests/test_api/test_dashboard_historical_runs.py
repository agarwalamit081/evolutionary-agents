"""Regression tests for the dashboard historical-runs fix (battery-05, Fix A).

The operator dashboard used to show only the runs Redis still remembers (their
status hashes TTL-expire after ~24h), so history disappeared even though every
run's cost / steps / eval data persist in Postgres. ``runs_with_cost`` now merges
live Redis runs with archived runs reconstructed from ``cost_ledger`` (de-duplicated
by their cost key), ``summary`` exposes an all-time run count, and the run-detail
route reconstructs an archived run from Postgres on a Redis miss (200, not 404).

Deterministic + infra-free: mocked SQLAlchemy sessions + a fake Redis SCAN, the
same pattern as ``test_dashboard.py``.
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
# Local fakes (kept hermetic — mirrors the helpers in test_dashboard.py)
# ---------------------------------------------------------------------------


def _row(**kw: Any) -> SimpleNamespace:
    return SimpleNamespace(**kw)


def _run(run_id: str, *, thread_id: str | None = None, **kw: Any) -> RunStatus:
    return RunStatus(run_id=run_id, thread_id=thread_id or f"api-{run_id}", **kw)


class _FakeRedis:
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
    return RunStatusStore(_FakeRedis(hashes), SimpleNamespace(status_ttl_s=0))  # type: ignore[arg-type]


_UNSET = object()  # sentinel: distinguishes "not provided" from the value None


def _result(rows: Any = _UNSET, first: Any = _UNSET) -> MagicMock:
    """A fake ``execute()`` result exposing ``.all()`` and/or ``.first()``.

    Uses a sentinel so ``_result(first=None)`` genuinely makes ``.first()``
    return ``None`` (None-as-default would collide with the absent-arg case)."""
    res = MagicMock()
    if rows is not _UNSET:
        res.all = lambda: rows
    if first is not _UNSET:
        res.first = lambda: first
    return res


def _seq_session(results: list[Any]) -> MagicMock:
    """An AsyncSession whose successive ``execute()`` calls return ``results``."""
    session = MagicMock()
    session.execute = AsyncMock(side_effect=list(results))
    session.rollback = AsyncMock()
    return session


class _FakeSessionCtx:
    async def __aenter__(self) -> MagicMock:
        return MagicMock()

    async def __aexit__(self, *_args: object) -> bool:
        return False


_DT = datetime(2026, 7, 10, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Data layer — archived view helpers
# ---------------------------------------------------------------------------


class TestArchivedRunView:
    def test_shape_strips_prefix_and_marks_archived(self) -> None:
        view = data._archived_run_view(  # noqa: SLF001 — view-shape helper under test
            {"run_id": "api-old1", "first_at": _DT, "last_at": _DT}
        )
        assert view["run_id"] == "old1"        # api- prefix stripped
        assert view["thread_id"] == "api-old1"  # the cost key (kept whole)
        assert view["status"] == "archived"
        assert view["is_complete"] is None
        assert view["iteration_count"] is None
        assert view["started_at"].startswith("2026-07-10T12:00")  # first_at iso

    def test_bare_run_id_strips_api_and_cli_prefixes(self) -> None:
        assert data._bare_run_id("api-r1") == "r1"  # noqa: SLF001
        assert data._bare_run_id("cli-r2") == "r2"
        assert data._bare_run_id("r3") == "r3"  # no prefix → unchanged


@pytest.mark.asyncio
class TestArchivedRunViewFor:
    async def test_returns_none_when_postgres_has_no_row(self) -> None:
        # .first() → None ⇒ unknown to Postgres ⇒ None (route will 404).
        session = MagicMock()
        session.execute = AsyncMock(return_value=_result(first=None))
        session.rollback = AsyncMock()
        assert await data._archived_run_view_for(session, "ghost") is None  # noqa: SLF001

    async def test_reconstructs_from_any_candidate_key(self) -> None:
        # The bare id "r1" matches the cost_ledger thread_id "api-r1".
        row = _row(run_id="api-r1", first_at=_DT, last_at=_DT)
        session = MagicMock()
        session.execute = AsyncMock(return_value=_result(first=row))
        session.rollback = AsyncMock()
        view = await data._archived_run_view_for(session, "r1")  # noqa: SLF001
        assert view is not None
        assert view["thread_id"] == "api-r1"
        assert view["status"] == "archived"

    async def test_degrades_to_none_on_db_error(self) -> None:
        session = MagicMock()
        session.execute = AsyncMock(side_effect=RuntimeError("db down"))
        session.rollback = AsyncMock()
        assert await data._archived_run_view_for(session, "r1") is None  # noqa: SLF001


# ---------------------------------------------------------------------------
# Data layer — runs_with_cost merge + de-dup + summary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRunWithCostMergesHistorical:
    @staticmethod
    def _store_one_live() -> RunStatusStore:
        live = _run("live", status=JobStatus.RUNNING, started_at="2026-07-15T00:00:00+00:00")
        return _store({f"turing:run:{live.run_id}": live.to_hash()})

    async def test_appends_archived_runs_not_in_redis(self) -> None:
        cost_rows = [
            _row(run_id="api-live", cost_usd=0.5, calls=3, total_tokens=100),
            _row(run_id="api-old2", cost_usd=0.1, calls=2, total_tokens=20),
            _row(run_id="api-old1", cost_usd=0.2, calls=1, total_tokens=10),
            _row(run_id="cli-cli1", cost_usd=0.05, calls=1, total_tokens=5),
        ]
        # Redis forgot everything but "live"; historical lists all four (newest
        # first), including the live run's thread_id — which MUST de-dup out.
        historical_rows = [
            _row(run_id="api-old2", first_at=_DT, last_at=_DT),
            _row(run_id="api-live", first_at=_DT, last_at=_DT),
            _row(run_id="api-old1", first_at=_DT, last_at=_DT),
            _row(run_id="cli-cli1", first_at=_DT, last_at=_DT),
        ]
        session = _seq_session([_result(cost_rows), _result(historical_rows)])
        runs, summary = await data.runs_with_cost(self._store_one_live(), session, limit=100)
        # Live run first; archived rows appended in historical order; the live
        # run's thread_id ("api-live") is NOT re-added as an archived row.
        assert [r["run_id"] for r in runs] == ["live", "old2", "old1", "cli1"]
        assert sum(1 for r in runs if r["thread_id"] == "api-live") == 1  # no dup
        assert all(r["status"] == "archived" for r in runs[1:])
        # Each row — live and archived — carries a cost block joined from the index.
        assert next(r for r in runs if r["run_id"] == "live")["cost"]["cost_usd"] == 0.5
        assert next(r for r in runs if r["run_id"] == "old2")["cost"]["calls"] == 2
        # All-time count comes from the unbounded cost index, not the display list.
        assert summary["runs_total_all"] == 4

    async def test_respects_limit_after_live_runs(self) -> None:
        # limit=2 ⇒ live + one archived; the rest of history is dropped.
        cost_rows = [
            _row(run_id="api-live", cost_usd=0.5, calls=3, total_tokens=100),
            _row(run_id="api-old1", cost_usd=0.2, calls=1, total_tokens=10),
            _row(run_id="api-old2", cost_usd=0.1, calls=2, total_tokens=20),
        ]
        historical_rows = [
            _row(run_id="api-old1", first_at=_DT, last_at=_DT),
            _row(run_id="api-old2", first_at=_DT, last_at=_DT),
        ]
        session = _seq_session([_result(cost_rows), _result(historical_rows)])
        runs, _ = await data.runs_with_cost(self._store_one_live(), session, limit=2)
        assert [r["run_id"] for r in runs] == ["live", "old1"]


@pytest.mark.asyncio
class TestSummaryHistoricalTotals:
    async def test_all_time_total_exceeds_display_window(self) -> None:
        # One live run in Redis, but the cost index knows about 3 historical runs.
        live = _run("only", status=JobStatus.COMPLETED, is_complete=True,
                    started_at="2026-07-15T00:00:00+00:00")
        store = _store({f"turing:run:{live.run_id}": live.to_hash()})
        cost_rows = [
            _row(run_id="api-only", cost_usd=0.4, calls=2, total_tokens=50),
            _row(run_id="api-old1", cost_usd=0.3, calls=1, total_tokens=30),
            _row(run_id="api-old2", cost_usd=0.2, calls=1, total_tokens=20),
        ]
        # No archived rows merge in (historical empty) ⇒ display window = 1.
        session = _seq_session([_result(cost_rows), _result([])])
        _runs, summary = await data.runs_with_cost(store, session)
        assert summary["runs_total"] == 1                 # display window (live only)
        assert summary["runs_total_all"] == 3             # all-time from cost index
        assert summary["total_cost_usd"] == pytest.approx(0.9)  # Σ over ALL index entries


# ---------------------------------------------------------------------------
# Route — archived run-detail reconstructs from Postgres (200, not 404)
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture
def patch_infra() -> Any:
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


_TOKEN_ZEROS = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0,
                "total_tokens": 0, "input_pct": 0.0, "cache_hit_pct": 0.0}


class TestArchivedRunDetailRoute:
    def test_redis_miss_with_postgres_hit_returns_200_archived(
        self, client: TestClient, patch_infra: Any
    ) -> None:
        store, _ = patch_infra
        store.get = AsyncMock(return_value=None)  # Redis forgot this run
        archived = {
            "run_id": "old1", "thread_id": "api-old1", "status": "archived",
            "final_output": None, "is_complete": None, "iteration_count": None,
            "error": None, "started_at": "2026-07-01T00:00:00",
            "finished_at": "2026-07-01T01:00:00", "results_dir": None,
        }
        breakdown = [{"model": "glm-5.2", "cost_usd": 0.3, "calls": 2, "total_tokens": 200}]
        with patch.object(data, "_archived_run_view_for", new=AsyncMock(return_value=archived)), \
             patch.object(data, "run_cost_breakdown", new=AsyncMock(return_value=breakdown)), \
             patch.object(data, "execution_steps", new=AsyncMock(return_value=[])), \
             patch.object(data, "run_token_split", new=AsyncMock(return_value=_TOKEN_ZEROS)):
            resp = client.get("/dashboard/runs/old1")
        assert resp.status_code == 200, resp.text
        body = resp.text
        assert "badge-archived" in body          # the archived status badge
        assert "Archived run." in body           # the honest "live status expired" callout
        assert "glm-5.2" in body                 # cost breakdown reconstructed from Postgres

    def test_redis_and_postgres_miss_returns_404(
        self, client: TestClient, patch_infra: Any
    ) -> None:
        store, _ = patch_infra
        store.get = AsyncMock(return_value=None)
        with patch.object(data, "_archived_run_view_for", new=AsyncMock(return_value=None)):
            resp = client.get("/dashboard/runs/ghost")
        assert resp.status_code == 404
