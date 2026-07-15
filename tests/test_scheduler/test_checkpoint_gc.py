"""Tests for checkpoint GC (Phase 6 q101): UUIDv6 age parse + stale decision + run().

The age signal is a langgraph UUIDv6 ``checkpoint_id`` (no wall-clock column
exists). These tests pin the parse against a validated live sample, exercise
the pure stale-decision logic, and cover the run() I/O (dry-run vs live) with a
fake session — plus the never-raise contract and the scheduler job wiring.
"""

from __future__ import annotations

import datetime as dt
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from src.scheduler.checkpoint_gc import (
    CheckpointGc,
    _stale_threads,
    _uuid6_to_unix,
    add_checkpoint_gc_job,
)

_UUID_EPOCH = 0x01B21DD213814000


def _unix_to_uuid6(unix: float) -> str:
    """Reverse of ``_uuid6_to_unix`` — generate a UUIDv6 string for a Unix time."""
    intervals = (int(unix * 1e7) + _UUID_EPOCH) & ((1 << 60) - 1)
    time_low = (intervals >> 28) & 0xFFFFFFFF
    time_mid = (intervals >> 12) & 0xFFFF
    time_hi = intervals & 0xFFF
    return str(UUID(fields=(time_low, time_mid, 0x6000 | time_hi, 0x80, 0x00, 0)))


# ── UUIDv6 parse ─────────────────────────────────────────────────────────────


def test_uuid6_to_unix_matches_validated_live_sample() -> None:
    # Live sample from turing_agent.checkpoints; its thread_id suffix is -20260714.
    cid = "1f17fb23-6127-6cfb-8099-2baed48a1e28"
    day = dt.datetime.fromtimestamp(_uuid6_to_unix(cid), tz=dt.timezone.utc).date()
    assert day == dt.date(2026, 7, 14)


def test_uuid6_to_unix_round_trip_within_one_second() -> None:
    for unix in (1_700_000_000.0, 1_750_000_000.0, 1_800_000_000.0):
        assert abs(_uuid6_to_unix(_unix_to_uuid6(unix)) - unix) < 1.0


def test_uuid6_to_unix_rejects_non_uuid() -> None:
    with pytest.raises(ValueError):
        _uuid6_to_unix("not-a-uuid")


# ── stale decision (pure) ────────────────────────────────────────────────────


def test_stale_threads_keeps_recent_drops_old() -> None:
    now = 1_800_000_000.0
    newest = {
        "keep-recent": _unix_to_uuid6(now - 86_400),  # 1 day ago → keep (ttl=7)
        "drop-old": _unix_to_uuid6(now - 8 * 86_400),  # 8 days ago → stale
    }
    assert _stale_threads(newest, ttl_days=7, now=lambda: now) == ["drop-old"]


def test_stale_threads_skips_unparseable_keeps_them() -> None:
    # "ok" (1970) is far stale; "bad" cannot be aged → skipped (kept, not stale).
    stale = _stale_threads(
        {"bad": "not-a-uuid", "ok": _unix_to_uuid6(0)}, ttl_days=7, now=lambda: 1e12
    )
    assert stale == ["ok"]


# ── run() I/O ─────────────────────────────────────────────────────────────────


class _FakeResult:
    def __init__(self, rowcount: int, rows: list[tuple[str, str]] | None = None) -> None:
        self.rowcount = rowcount
        self._rows = rows or []

    def fetchall(self) -> list[tuple[str, str]]:
        return self._rows


class _FakeSession:
    """Async context-manager session: serves the param-less SELECT then DELETEs."""

    def __init__(self, newest_rows: dict[str, str], del_rowcount: int) -> None:
        self._newest = newest_rows
        self._del = del_rowcount
        self.execute_calls: list[Any] = []
        self.committed = False

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False

    async def execute(self, stmt: Any, params: Any = None) -> _FakeResult:
        self.execute_calls.append(params)
        if params is None:  # the SELECT (no bind params)
            return _FakeResult(0, rows=list(self._newest.items()))
        return _FakeResult(self._del)  # a DELETE

    async def commit(self) -> None:
        self.committed = True


def _factory(session: _FakeSession):  # type: ignore[no-untyped-def]
    return lambda: session


@pytest.mark.asyncio
async def test_run_dry_run_reports_but_does_not_delete() -> None:
    now = 1_800_000_000.0
    newest = {"drop-old": _unix_to_uuid6(now - 8 * 86_400)}
    session = _FakeSession(newest, del_rowcount=5)
    gc = CheckpointGc(
        MagicMock(ttl_days=7, dry_run=True),
        session_factory=_factory(session),
        now=lambda: now,
    )
    report = await gc.run()
    assert report == {  # dry-run returns no delete counts
        "gc": True,
        "scanned": 1,
        "stale": 1,
        "dry_run": True,
        "sample": ["drop-old"],
    }
    assert session.committed is False  # nothing deleted


@pytest.mark.asyncio
async def test_run_live_deletes_all_three_tables_and_commits() -> None:
    now = 1_800_000_000.0
    newest = {"drop-old": _unix_to_uuid6(now - 8 * 86_400)}
    session = _FakeSession(newest, del_rowcount=5)
    gc = CheckpointGc(
        MagicMock(ttl_days=7, dry_run=False),
        session_factory=_factory(session),
        now=lambda: now,
    )
    report = await gc.run()
    assert report["dry_run"] is False
    assert report["stale"] == 1
    assert report["checkpoints"] == 5
    assert report["writes"] == 5
    assert report["blobs"] == 5
    assert session.committed is True
    # exactly 3 DELETEs (one per table) after the SELECT
    delete_calls = [c for c in session.execute_calls if c is not None]
    assert len(delete_calls) == 3
    for call in delete_calls:
        assert call == {"threads": ["drop-old"]}


@pytest.mark.asyncio
async def test_run_no_stale_threads_is_a_clean_noop() -> None:
    now = 1_800_000_000.0
    newest = {"keep": _unix_to_uuid6(now - 86_400)}  # 1 day old → not stale
    session = _FakeSession(newest, del_rowcount=5)
    gc = CheckpointGc(
        MagicMock(ttl_days=7, dry_run=False),
        session_factory=_factory(session),
        now=lambda: now,
    )
    report = await gc.run()
    assert report == {"gc": True, "scanned": 1, "stale": 0, "dry_run": False}
    assert session.committed is False


@pytest.mark.asyncio
async def test_run_never_raises_on_db_error() -> None:
    class _Boom:
        async def __aenter__(self) -> _Boom:
            return self

        async def __aexit__(self, *_e: object) -> bool:
            return False

        async def execute(self, *_a: object, **_k: object) -> Any:
            raise RuntimeError("db down")

    gc = CheckpointGc(
        MagicMock(ttl_days=7, dry_run=True),
        session_factory=lambda: _Boom(),
        now=lambda: 1e12,
    )
    report = await gc.run()
    assert report["gc"] is False
    assert "db down" in report["error"]


def test_add_checkpoint_gc_job_registers_with_safe_policy() -> None:
    scheduler = MagicMock()
    settings = MagicMock(cron="0 6 * * *", timezone="UTC")
    add_checkpoint_gc_job(scheduler, CheckpointGc(settings), settings)
    assert scheduler.add_job.called
    _args, kwargs = scheduler.add_job.call_args
    assert kwargs["id"] == "turing-checkpoint-gc"
    assert kwargs["max_instances"] == 1
    assert kwargs["coalesce"] is True
