"""ToolPersister performance retirement (M4): underperformer scan + retire.

The SQL filter (``calls >= min_runs AND success_rate < floor``) lives
server-side, so these tests assert the wiring with a capturing fake session
(the returned rows flow through sorted) plus the delegation to ``retire`` and
the best-effort empty/error paths. The filter itself is validated end-to-end
against real Postgres in the Phase-10 live run.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.tools.dynamic.persister import ToolPersister


class _FakeResult:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def all(self) -> list[Any]:
        return self._rows


class _FakeSession:
    def __init__(self, rows: list[Any] | None = None) -> None:
        self._rows = rows or []

    async def execute(self, _stmt: Any) -> _FakeResult:
        return _FakeResult(self._rows)

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *_args: Any) -> bool:
        return False


class _RaisingSession:
    async def execute(self, _stmt: Any) -> Any:
        raise RuntimeError("connection refused")

    async def __aenter__(self) -> _RaisingSession:
        return self

    async def __aexit__(self, *_args: Any) -> bool:
        return False


class TestUnderperformingTools:
    @pytest.mark.asyncio
    async def test_returns_sorted_names_from_session(self) -> None:
        session = _FakeSession(rows=[("z_tool",), ("a_tool",)])
        with patch("src.db.session.get_session", lambda: session):
            names = await ToolPersister().underperforming_tools(20, 0.25)
        assert names == ["a_tool", "z_tool"]

    @pytest.mark.asyncio
    async def test_empty_when_no_qualifiers(self) -> None:
        with patch("src.db.session.get_session", lambda: _FakeSession(rows=[])):
            assert await ToolPersister().underperforming_tools(20, 0.25) == []

    @pytest.mark.asyncio
    async def test_db_error_degrades_to_empty(self) -> None:
        with patch("src.db.session.get_session", lambda: _RaisingSession()):
            assert await ToolPersister().underperforming_tools(20, 0.25) == []


class TestRetireUnderperforming:
    @pytest.mark.asyncio
    async def test_no_qualifiers_returns_zero(self) -> None:
        p = ToolPersister()
        p.underperforming_tools = AsyncMock(return_value=[])  # type: ignore[method-assign]
        p.retire = AsyncMock(return_value=99)  # type: ignore[method-assign]
        assert await p.retire_underperforming(20, 0.25) == 0
        p.retire.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_delegates_names_to_retire(self) -> None:
        p = ToolPersister()
        p.underperforming_tools = AsyncMock(return_value=["bad_a", "bad_b"])  # type: ignore[method-assign]
        p.retire = AsyncMock(return_value=2)  # type: ignore[method-assign]
        count = await p.retire_underperforming(20, 0.25)
        assert count == 2
        p.retire.assert_awaited_once_with(["bad_a", "bad_b"])

    @pytest.mark.asyncio
    async def test_empty_output_floor_threaded(self) -> None:
        """Phase 4 G: retire_underperforming forwards empty_output_floor to the scan."""
        p = ToolPersister()
        p.underperforming_tools = AsyncMock(return_value=["blank"])  # type: ignore[method-assign]
        p.retire = AsyncMock(return_value=1)  # type: ignore[method-assign]
        await p.retire_underperforming(20, 0.5, empty_output_floor=0.8)
        p.underperforming_tools.assert_awaited_once_with(20, 0.5, 0.8)


class _CapturingSession(_FakeSession):
    """Fake session that records the compiled statement for SQL-wiring asserts."""

    def __init__(self) -> None:
        super().__init__(rows=[("captured",)])
        self.statements: list[Any] = []

    async def execute(self, stmt: Any) -> _FakeResult:
        self.statements.append(stmt)
        return _FakeResult(self._rows)


class TestEmptyOutputPredicate:
    """Phase 4 G — the empty-output OR clause is emitted only when requested."""

    @pytest.mark.asyncio
    async def test_empty_output_floor_adds_or_clause(self) -> None:
        session = _CapturingSession()
        with patch("src.db.session.get_session", lambda: session):
            await ToolPersister().underperforming_tools(20, 0.5, empty_output_floor=0.8)
        sql = str(session.statements[0])
        assert "success_rate" in sql
        assert "empty_output_rate" in sql

    @pytest.mark.asyncio
    async def test_no_empty_output_floor_omits_clause(self) -> None:
        """Default (None) reproduces the legacy success_rate-only query."""
        session = _CapturingSession()
        with patch("src.db.session.get_session", lambda: session):
            await ToolPersister().underperforming_tools(20, 0.5)
        sql = str(session.statements[0])
        assert "success_rate" in sql
        assert "empty_output_rate" not in sql


# ─── Phase-4 dead-weight pass: 0-call tools aged past the gate ──────────────


class TestUnusedTools:
    @pytest.mark.asyncio
    async def test_returns_sorted_names_from_session(self) -> None:
        session = _FakeSession(rows=[("z_dead",), ("a_dead",)])
        with patch("src.db.session.get_session", lambda: session):
            names = await ToolPersister().unused_tools(30)
        assert names == ["a_dead", "z_dead"]

    @pytest.mark.asyncio
    async def test_empty_when_no_qualifiers(self) -> None:
        with patch("src.db.session.get_session", lambda: _FakeSession(rows=[])):
            assert await ToolPersister().unused_tools(30) == []

    @pytest.mark.asyncio
    async def test_db_error_degrades_to_empty(self) -> None:
        with patch("src.db.session.get_session", lambda: _RaisingSession()):
            assert await ToolPersister().unused_tools(30) == []

    @pytest.mark.asyncio
    async def test_disabled_when_age_gate_le_zero(self) -> None:
        """``<= 0`` short-circuits before any session is opened (no DB contact)."""

        def _forbid() -> Any:
            raise AssertionError("get_session must not open when the pass is disabled")

        with patch("src.db.session.get_session", _forbid):
            assert await ToolPersister().unused_tools(0) == []
            assert await ToolPersister().unused_tools(-5) == []

    @pytest.mark.asyncio
    async def test_filter_uses_calls_zero_and_age_gate(self) -> None:
        """The scan emits the calls + created_at age-gate predicates."""
        session = _CapturingSession()
        with patch("src.db.session.get_session", lambda: session):
            await ToolPersister().unused_tools(30)
        sql = str(session.statements[0])
        assert "calls" in sql
        assert "created_at" in sql
        assert "is_active" in sql

    @pytest.mark.asyncio
    async def test_filter_uses_le_predicate_for_max_calls(self) -> None:
        """The scan uses ``calls <= max_calls`` (inclusive), so a non-zero floor
        retires low-call abandonware too. Default max_calls=0 == the original
        never-invoked pass; a raised floor widens the selection without changing
        the predicate shape."""
        session = _CapturingSession()
        with patch("src.db.session.get_session", lambda: session):
            await ToolPersister().unused_tools(30, max_calls=3)
        sql = str(session.statements[0])
        # The predicate is ``calls <= :param`` (NOT equality to a literal 0).
        assert "<=" in sql
        assert "calls" in sql


class TestRetireUnused:
    @pytest.mark.asyncio
    async def test_no_qualifiers_returns_zero(self) -> None:
        p = ToolPersister()
        p.unused_tools = AsyncMock(return_value=[])  # type: ignore[method-assign]
        p.retire = AsyncMock(return_value=99)  # type: ignore[method-assign]
        assert await p.retire_unused(30) == 0
        p.retire.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_delegates_names_to_retire(self) -> None:
        p = ToolPersister()
        p.unused_tools = AsyncMock(return_value=["dead_a", "dead_b"])  # type: ignore[method-assign]
        p.retire = AsyncMock(return_value=2)  # type: ignore[method-assign]
        count = await p.retire_unused(30)
        assert count == 2
        # Default max_calls=0 forwards to unused_tools as a keyword (preserves
        # the original "never invoked" semantics).
        p.unused_tools.assert_awaited_once_with(30, max_calls=0)
        p.retire.assert_awaited_once_with(["dead_a", "dead_b"])

    @pytest.mark.asyncio
    async def test_forwards_max_calls_floor_to_unused_tools(self) -> None:
        """retire_unused(max_calls=N) forwards the floor so low-call abandonware
        (calls <= N) is also retired, not just zero-call dead weight."""
        p = ToolPersister()
        p.unused_tools = AsyncMock(return_value=["dead_a"])  # type: ignore[method-assign]
        p.retire = AsyncMock(return_value=1)  # type: ignore[method-assign]
        await p.retire_unused(30, max_calls=3)
        p.unused_tools.assert_awaited_once_with(30, max_calls=3)
        p.retire.assert_awaited_once_with(["dead_a"])

    @pytest.mark.asyncio
    async def test_disabled_age_zero_returns_zero(self) -> None:
        """``min_age_days=0`` → the real scan short-circuits to [], no retirement."""
        p = ToolPersister()
        p.retire = AsyncMock(return_value=99)  # type: ignore[method-assign]
        assert await p.retire_unused(0) == 0
        p.retire.assert_not_awaited()
