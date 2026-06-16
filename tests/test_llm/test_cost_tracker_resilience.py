"""End-to-end regression for Bug D (cost-ledger resilience) in ``record_usage``.

Unlike ``tests/test_llm/test_cost_tracker.py`` (which uses ``AsyncMock``
sessions), this file exercises the real commit/rollback lifecycle against a
live **aiosqlite** async session. The point is to prove the shared session
itself recovers: a failed flush poisons an ``AsyncSession`` into a
"pending-rollback" state where *every* subsequent operation re-raises the
original error until ``rollback()`` clears it. ``record_usage`` must call that
rollback internally so the next call on the SAME session still commits.

We build a standalone ``cost_ledger`` table (mirroring ``src/db/models.py``'s
``CostLedger`` columns, without the FK to ``task_executions`` so the schema is
self-contained on SQLite) and a deterministic poison: a ``CHECK`` constraint
on ``input_tokens`` that a deliberately-bad insert violates, triggering a real
``IntegrityError`` out of ``commit()``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.llm.cost_tracker import CostTracker

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _make_settings() -> MagicMock:
    """A Settings stub exposing the ``budget.max_cost_usd`` CostTracker reads."""
    settings = MagicMock()
    settings.budget.max_cost_usd = 10.0
    return settings


# ``cost_ledger`` mirrors the columns ``record_usage`` writes. The
# ``CHECK (input_tokens >= 0)`` is the deterministic poison trigger: a row with
# a negative token count violates it at flush time. The FK to
# ``task_executions`` is dropped so the table is standalone on SQLite.
_COST_LEDGER = sa.Table(
    "cost_ledger",
    sa.MetaData(),
    sa.Column("id", sa.Text, primary_key=True),
    sa.Column("task_id", sa.Text, nullable=True),
    sa.Column("provider", sa.Text, nullable=False),
    sa.Column("model", sa.Text, nullable=False),
    sa.Column("input_tokens", sa.Integer, nullable=False),
    sa.Column("output_tokens", sa.Integer, nullable=False),
    sa.Column("total_tokens", sa.Integer, nullable=False, default=0),
    # cached_tokens is emitted on every INSERT (ORM default=0) even though
    # record_usage never sets it; mirror it so the INSERT column list matches.
    sa.Column("cached_tokens", sa.Integer, nullable=False, default=0),
    sa.Column("cost_usd", sa.Numeric(10, 6), nullable=False),
    sa.Column("latency_ms", sa.Integer, nullable=True),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint("input_tokens >= 0", name="ck_input_nonneg"),
)


# A minimal ORM mapping onto the standalone Core table so the bad row can be
# enqueued in the session's pending set (via ``session.add``) rather than
# executed immediately. Core ``execute()`` runs the INSERT at once, which would
# raise before the session is poisoned; the ORM unit-of-work defers the flush
# to the next ``commit()`` — reproducing Bug D's real trigger: a pending row
# that fails on flush and leaves the AsyncSession in "pending-rollback" state.
class _PendingLedgerRow:
    """Mapped stand-in for a cost_ledger row pending flush."""

    id: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: int
    cost_usd: float
    created_at: datetime


sa.orm.registry().map_imperatively(_PendingLedgerRow, _COST_LEDGER)


@pytest.fixture
async def shared_session() -> AsyncSession:
    """A single shared AsyncSession over an in-memory aiosqlite engine.

    ``expire_on_commit=False`` keeps committed instances usable; the engine
    stays alive for the session's lifetime because both share one connection
    scope (in-memory SQLite is per-connection, so we pin the same connection
    via ``creator``-free ``:memory:`` + a single session).
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(lambda sync_conn: _COST_LEDGER.create(sync_conn))
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    session = factory()
    try:
        yield session
    finally:
        await session.close()
        await engine.dispose()


async def _count_rows(session: AsyncSession) -> int:
    """Count committed cost_ledger rows on the shared session."""
    result = await session.execute(sa.select(sa.func.count()).select_from(_COST_LEDGER))
    return int(result.scalar_one())


async def _poison(session: AsyncSession) -> None:
    """Enqueue a pending row violating the ``input_tokens >= 0`` CHECK.

    Added via the ORM unit-of-work (``session.add``), so the INSERT is NOT
    executed immediately — it stays pending until the next ``commit()`` flushes
    it alongside whatever else is queued. That flush then raises
    ``IntegrityError`` (CHECK failed) and leaves the AsyncSession in the
    "pending-rollback" state: the real Bug D condition, where every later op
    re-raises until ``rollback()`` clears it.
    """
    session.add(_PendingLedgerRow(
        id="poison",
        provider="p",
        model="m",
        input_tokens=-1,
        output_tokens=0,
        total_tokens=-1,
        cached_tokens=0,
        cost_usd=0.0,
        created_at=_EPOCH,
    ))


class TestRecordUsageResilience:
    """Bug D: a poisoned shared session must not break later record_usage calls."""

    @pytest.mark.asyncio
    async def test_normal_usage_records_row(self, shared_session: AsyncSession) -> None:
        """Happy path: one record_usage lands exactly one committed row."""
        tracker = CostTracker(session=shared_session, settings=_make_settings())

        cost = await tracker.record_usage(
            model="test-model",
            provider="openai",
            input_tokens=100,
            output_tokens=40,
        )

        assert cost > 0
        assert await _count_rows(shared_session) == 1
        row = (
            await shared_session.execute(
                sa.select(
                    _COST_LEDGER.c.model,
                    _COST_LEDGER.c.input_tokens,
                    _COST_LEDGER.c.output_tokens,
                    _COST_LEDGER.c.total_tokens,
                )
            )
        ).one()
        assert row.model == "test-model"
        assert row.input_tokens == 100
        assert row.output_tokens == 40
        assert row.total_tokens == 140

    @pytest.mark.asyncio
    async def test_subsequent_call_succeeds_after_poisoned_commit(
        self, shared_session: AsyncSession
    ) -> None:
        """A failed commit must not poison the session for the next call.

        The most adversarial Bug D condition: commit a row violating the
        ``input_tokens >= 0`` CHECK so ``commit()`` itself raises and leaves the
        AsyncSession in the "pending-rollback" state. record_usage can't write
        through a session in that state — so the FIRST call's write is rolled
        back with the poison. But it must NOT re-raise (observability never
        crashes the run), and its internal ``rollback()`` must clear the poison
        so the SECOND call lands cleanly. That two-call recovery is the real
        resilience guarantee: the session is usable again after record_usage.
        """
        # Pre-poison: a bad INSERT on the shared session that fails on commit.
        # After this commit() raises, the session is in a failed transaction.
        await _poison(shared_session)
        with pytest.raises(sa.exc.IntegrityError):
            await shared_session.commit()

        tracker = CostTracker(session=shared_session, settings=_make_settings())

        # First call inherits the poisoned session — its own write cannot land
        # (the pending-rollback tx discards it). It must return the cost and
        # MUST NOT re-raise: the run keeps going.
        cost1 = await tracker.record_usage(
            model="recovered-model",
            provider="openai",
            input_tokens=250,
            output_tokens=50,
        )
        assert cost1 == CostTracker.calculate_cost("recovered-model", 250, 50)

        # Second call: record_usage's first-call rollback cleared the poison,
        # so the session is healthy again and this write commits.
        cost2 = await tracker.record_usage(
            model="second-model",
            provider="openai",
            input_tokens=10,
            output_tokens=5,
        )
        assert cost2 == CostTracker.calculate_cost("second-model", 10, 5)

        # Exactly the second (recovered) row is committed; the poison + first
        # write were rolled back together.
        assert await _count_rows(shared_session) == 1
        landed = (await shared_session.execute(sa.select(_COST_LEDGER.c.model))).scalar_one()
        assert landed == "second-model"

    @pytest.mark.asyncio
    async def test_rollback_prevents_session_poison(
        self, shared_session: AsyncSession
    ) -> None:
        """After a record_usage whose commit fails, the session stays usable.

        Force record_usage's own commit to fail by pre-poisoning the session
        (bad pending INSERT), then verify record_usage swallows the failure and
        a SECOND record_usage commits cleanly — proving the internal rollback
        restored a usable transaction state.
        """
        # Pre-poison the shared session but do NOT commit yet — leave a bad
        # pending row so record_usage's commit() flush fails.
        await _poison(shared_session)

        tracker = CostTracker(session=shared_session, settings=_make_settings())

        # First record_usage: commit fails on the pending poison row. record_usage
        # must NOT re-raise (observability-only) and must return the cost.
        cost1 = await tracker.record_usage(
            model="first", provider="openai", input_tokens=10, output_tokens=5
        )
        assert cost1 == CostTracker.calculate_cost("first", 10, 5)

        # Second record_usage on the SAME (now-recovered) session commits cleanly.
        cost2 = await tracker.record_usage(
            model="second", provider="openai", input_tokens=20, output_tokens=10
        )
        assert cost2 == CostTracker.calculate_cost("second", 20, 10)

        # The poison row was rolled back; only the second (recovered) call landed.
        assert await _count_rows(shared_session) == 1
        landed = (await shared_session.execute(sa.select(_COST_LEDGER.c.model))).scalar_one()
        assert landed == "second"

    @pytest.mark.asyncio
    async def test_failed_record_usage_does_not_raise(
        self, shared_session: AsyncSession
    ) -> None:
        """record_usage must swallow a commit failure (return cost, never raise)."""
        await _poison(shared_session)

        tracker = CostTracker(session=shared_session, settings=_make_settings())

        # Must not raise even though the underlying commit hits IntegrityError.
        cost = await tracker.record_usage(
            model="swallowed", provider="openai", input_tokens=7, output_tokens=3
        )
        assert cost == CostTracker.calculate_cost("swallowed", 7, 3)
