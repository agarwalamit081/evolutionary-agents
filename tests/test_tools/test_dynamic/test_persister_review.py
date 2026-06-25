"""Tests for the D10 review-lifecycle methods on ToolPersister.

Mock-session unit tests mirroring tests/test_tools/test_dynamic/test_persister_recall.py:
real ORM models are instantiated (constructors are DB-free); only get_session()
is faked. Covers submit_pending_version / approve_pending / reject_pending /
list_tools / get_tool, plus the load_active_tools status filter (the regression
guard that keeps a pending_review version out of the live registry).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.dynamic.persister import ToolPersister

_HANDLER = "async def h(x: str) -> str:\n    return x\n"
_TEST = "assert h('a') == 'a' or True\n"


def _scalar_result(value: Any) -> MagicMock:
    """A session.execute() return whose scalar_one_or_none() yields ``value``."""
    r = MagicMock()
    r.scalar_one_or_none.return_value = value
    return r


def _list_result(rows: list[Any]) -> MagicMock:
    """A session.execute() return whose scalars().all() yields ``rows``."""
    r = MagicMock()
    r.scalars.return_value.all.return_value = rows
    return r


@pytest.fixture
def mock_session() -> MagicMock:
    """Mock AsyncSession. ``add`` simulates flush populating the PK default."""
    session = MagicMock()

    def _add(obj: Any) -> None:
        if getattr(obj, "id", None) is None:
            obj.id = uuid.uuid4()

    session.add = MagicMock(side_effect=_add)
    session.flush = AsyncMock()
    session.execute = AsyncMock()
    return session


@pytest.fixture
def persister(mock_session: MagicMock):
    """A ToolPersister whose get_session() yields the mock session."""

    @asynccontextmanager
    async def _fake_get_session() -> AsyncGenerator[MagicMock, None]:
        yield mock_session

    with patch("src.db.session.get_session", new=_fake_get_session):
        yield ToolPersister()


def _compiled(stmt: Any) -> str:
    return str(stmt.compile(compile_kwargs={"literal_binds": True}))


# ---------------------------------------------------------------------------
# submit_pending_version()
# ---------------------------------------------------------------------------


class TestSubmitPendingVersion:
    @pytest.mark.asyncio
    async def test_new_tool_stages_pending_review(
        self,
        persister: ToolPersister,
        mock_session: MagicMock,
    ) -> None:
        from src.db.models import ToolRegistration, ToolVersion

        # _get_registration → no existing tool.
        mock_session.execute = AsyncMock(return_value=_scalar_result(None))

        version_id = await persister.submit_pending_version(
            tool_name="new_tool",
            description="d",
            input_schema={"type": "object"},
            handler_code=_HANDLER,
            test_code=_TEST,
        )

        assert isinstance(version_id, uuid.UUID)
        added = [c.args[0] for c in mock_session.add.call_args_list]
        # registration (is_active=True) then version (pending_review, inactive).
        assert len(added) == 2
        assert isinstance(added[0], ToolRegistration)
        assert added[0].tool_name == "new_tool"
        assert added[0].is_active is True
        assert isinstance(added[1], ToolVersion)
        assert added[1].status == "pending_review"
        assert added[1].is_active is False
        assert added[1].version == 1
        assert added[1].test_content == _TEST

    @pytest.mark.asyncio
    async def test_existing_tool_bumps_version_and_leaves_live_alone(
        self,
        persister: ToolPersister,
        mock_session: MagicMock,
    ) -> None:
        from src.db.models import ToolRegistration, ToolVersion

        reg_id = uuid.uuid4()
        existing = ToolRegistration(
            id=reg_id,
            tool_name="old_tool",
            tool_type="generated",
            description="d",
            input_schema={},
            is_active=True,
        )
        latest = ToolVersion(tool_id=reg_id, version=2, code_content="old", is_active=True)
        # Real call order: _get_registration, registration UPDATE, _latest_version.
        mock_session.execute = AsyncMock(
            side_effect=[
                _scalar_result(existing),
                MagicMock(),
                _scalar_result(latest),
            ]
        )

        version_id = await persister.submit_pending_version(
            tool_name="old_tool",
            description="d2",
            input_schema={},
            handler_code=_HANDLER,
            test_code=_TEST,
        )

        assert isinstance(version_id, uuid.UUID)
        added = [c.args[0] for c in mock_session.add.call_args_list]
        # Only the pending version is added (registration is UPDATEd, not added).
        assert len(added) == 1
        pending = added[0]
        assert isinstance(pending, ToolVersion)
        assert pending.version == 3  # latest(2) + 1
        assert pending.status == "pending_review"
        assert pending.is_active is False
        # No UPDATE wrote tool_versions — the live v2 stays active (only a new
        # pending version is inserted via session.add, never an execute). The
        # registration SELECTs legitimately list an is_active *column*, so guard
        # on UPDATE+tool_versions. Compiled WITHOUT literal_binds (the UPDATE
        # carries a JSONB input_schema={} literal the SQLite renderer can't bind;
        # verb+target don't need bound values).
        for call in mock_session.execute.call_args_list:
            compiled = str(call.args[0].compile())
            assert not (
                compiled.lstrip().upper().startswith("UPDATE")
                and "tool_versions" in compiled
            )

    @pytest.mark.asyncio
    async def test_db_error_returns_none(
        self,
        persister: ToolPersister,
        mock_session: MagicMock,
    ) -> None:
        mock_session.execute = AsyncMock(side_effect=RuntimeError("db down"))
        result = await persister.submit_pending_version(
            tool_name="boom",
            description="d",
            input_schema={},
            handler_code=_HANDLER,
            test_code=_TEST,
        )
        assert result is None


# ---------------------------------------------------------------------------
# approve_pending()
# ---------------------------------------------------------------------------


class TestApprovePending:
    @pytest.mark.asyncio
    async def test_promotes_latest_pending_and_deactivates_others(
        self,
        persister: ToolPersister,
        mock_session: MagicMock,
    ) -> None:
        from src.db.models import ToolRegistration, ToolVersion

        reg_id = uuid.uuid4()
        reg = ToolRegistration(
            id=reg_id, tool_name="t", tool_type="generated", description="d",
            input_schema={}, is_active=True,
        )
        pending = ToolVersion(
            id=uuid.uuid4(), tool_id=reg_id, version=3, code_content="new",
            is_active=False, status="pending_review",
        )
        # _get_registration, _latest_version(pending), deactivate-all, activate.
        mock_session.execute = AsyncMock(
            side_effect=[
                _scalar_result(reg),
                _scalar_result(pending),
                MagicMock(),
                MagicMock(),
            ]
        )

        result = await persister.approve_pending("t")

        assert result == {"tool_name": "t", "version": 3, "status": "approved"}
        assert mock_session.execute.await_count == 4
        # The final statement activates + approves the staged version.
        activate_sql = _compiled(mock_session.execute.call_args_list[3].args[0])
        assert "approved" in activate_sql and "is_active" in activate_sql

    @pytest.mark.asyncio
    async def test_no_pending_returns_none(
        self,
        persister: ToolPersister,
        mock_session: MagicMock,
    ) -> None:
        from src.db.models import ToolRegistration

        reg = ToolRegistration(
            id=uuid.uuid4(), tool_name="t", tool_type="generated", description="d",
            input_schema={}, is_active=True,
        )
        mock_session.execute = AsyncMock(
            side_effect=[_scalar_result(reg), _scalar_result(None)]
        )
        assert await persister.approve_pending("t") is None

    @pytest.mark.asyncio
    async def test_no_tool_returns_none(
        self,
        persister: ToolPersister,
        mock_session: MagicMock,
    ) -> None:
        mock_session.execute = AsyncMock(return_value=_scalar_result(None))
        assert await persister.approve_pending("missing") is None


# ---------------------------------------------------------------------------
# reject_pending()
# ---------------------------------------------------------------------------


class TestRejectPending:
    @pytest.mark.asyncio
    async def test_marks_latest_pending_rejected(
        self,
        persister: ToolPersister,
        mock_session: MagicMock,
    ) -> None:
        from src.db.models import ToolRegistration, ToolVersion

        reg_id = uuid.uuid4()
        reg = ToolRegistration(
            id=reg_id, tool_name="t", tool_type="generated", description="d",
            input_schema={}, is_active=True,
        )
        pending = ToolVersion(
            id=uuid.uuid4(), tool_id=reg_id, version=2, code_content="new",
            is_active=False, status="pending_review",
        )
        mock_session.execute = AsyncMock(
            side_effect=[_scalar_result(reg), _scalar_result(pending), MagicMock()]
        )
        assert await persister.reject_pending("t", reason="bad") is True
        reject_sql = _compiled(mock_session.execute.call_args_list[2].args[0])
        assert "rejected" in reject_sql

    @pytest.mark.asyncio
    async def test_no_pending_returns_false(
        self,
        persister: ToolPersister,
        mock_session: MagicMock,
    ) -> None:
        from src.db.models import ToolRegistration

        reg = ToolRegistration(
            id=uuid.uuid4(), tool_name="t", tool_type="generated", description="d",
            input_schema={}, is_active=True,
        )
        mock_session.execute = AsyncMock(
            side_effect=[_scalar_result(reg), _scalar_result(None)]
        )
        assert await persister.reject_pending("t") is False


# ---------------------------------------------------------------------------
# list_tools() / get_tool()
# ---------------------------------------------------------------------------


class TestInspect:
    @pytest.mark.asyncio
    async def test_list_tools_reports_latest_status(
        self,
        persister: ToolPersister,
        mock_session: MagicMock,
    ) -> None:
        from src.db.models import ToolRegistration, ToolVersion

        reg_id = uuid.uuid4()
        reg = ToolRegistration(
            id=reg_id, tool_name="t", tool_type="generated", description="d",
            input_schema={}, is_active=True,
        )
        latest = ToolVersion(
            tool_id=reg_id, version=2, code_content="c", status="pending_review",
            is_active=False,
        )
        # regs SELECT (list), then _latest_version for the one reg.
        mock_session.execute = AsyncMock(
            side_effect=[_list_result([reg]), _scalar_result(latest)]
        )
        tools = await persister.list_tools()
        assert tools == [
            {
                "tool_name": "t",
                "description": "d",
                "is_active": True,
                "version": 2,
                "status": "pending_review",
                "version_active": False,
            }
        ]

    @pytest.mark.asyncio
    async def test_get_tool_returns_detail_and_history(
        self,
        persister: ToolPersister,
        mock_session: MagicMock,
    ) -> None:
        from src.db.models import ToolRegistration, ToolVersion

        reg_id = uuid.uuid4()
        reg = ToolRegistration(
            id=reg_id, tool_name="t", tool_type="generated", description="d",
            input_schema={"type": "object"}, is_active=True,
        )
        latest = ToolVersion(
            id=uuid.uuid4(), tool_id=reg_id, version=2, code_content="c2",
            test_content=_TEST, status="approved",
        )
        v1 = ToolVersion(tool_id=reg_id, version=1, code_content="c1", status="rejected")
        # _get_registration, _latest_version, versions history.
        mock_session.execute = AsyncMock(
            side_effect=[_scalar_result(reg), _scalar_result(latest), _list_result([latest, v1])]
        )
        detail = await persister.get_tool("t")
        assert detail is not None
        assert detail["tool_name"] == "t"
        assert detail["status"] == "approved"
        assert detail["code_content"] == "c2"
        assert detail["test_content"] == _TEST
        assert [h["version"] for h in detail["history"]] == [2, 1]

    @pytest.mark.asyncio
    async def test_get_missing_tool_returns_none(
        self,
        persister: ToolPersister,
        mock_session: MagicMock,
    ) -> None:
        mock_session.execute = AsyncMock(return_value=_scalar_result(None))
        assert await persister.get_tool("nope") is None


# ---------------------------------------------------------------------------
# load_active_tools() status filter — the regression guard
# ---------------------------------------------------------------------------


class TestLoadActiveToolsStatusFilter:
    @pytest.mark.asyncio
    async def test_version_query_requires_approved_status(
        self,
        persister: ToolPersister,
        mock_session: MagicMock,
    ) -> None:
        """load_active_tools' active-version SELECT must filter status='approved'
        (defense-in-depth alongside is_active) so a pending_review version is
        never materialized. Proved via the compiled SQL, not a real DB."""
        from src.db.models import ToolRegistration

        reg_row = ToolRegistration(
            id=uuid.uuid4(), tool_name="t", tool_type="generated", description="d",
            input_schema={}, is_active=True,
        )
        reg_result = MagicMock()
        reg_result.scalars.return_value.all.return_value = [reg_row]
        ver_result = MagicMock()
        ver_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(side_effect=[reg_result, ver_result])

        from src.tools.registry import ToolRegistry

        await persister.load_active_tools(ToolRegistry())

        # 2nd execute call is the active-version SELECT.
        ver_stmt = mock_session.execute.call_args_list[1].args[0]
        compiled = _compiled(ver_stmt)
        assert "status" in compiled and "approved" in compiled
