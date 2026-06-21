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
