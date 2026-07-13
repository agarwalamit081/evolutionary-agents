"""Tests for src.tools.dynamic.persister — persist + cross-run recall wiring.

Mock-session unit tests mirroring tests/test_evolution/test_persister.py: the
real ORM models (ToolRegistration/ToolVersion) are instantiated (constructors
are DB-free); only get_session() is faked. The load_active_tools tests fake the
two SELECT results (registration row, version row) and assert the handler is
materialized + registered into a live ToolRegistry — i.e. the recall path that
main.py exercises on every startup.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.dynamic.persister import ToolPersister
from src.tools.registry import ToolRegistry

# A minimal handler that uses only builtins (safe in the constrained namespace).
_VALID_HANDLER = '''async def char_count(text: str) -> str:
    chars = len(text)
    words = len(text.split())
    lines = text.count("\\n") + (1 if text else 0)
    return f"chars={chars} words={words} lines={lines}"
'''


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


# ---------------------------------------------------------------------------
# persist()
# ---------------------------------------------------------------------------


class TestPersist:
    @pytest.mark.asyncio
    async def test_persist_new_tool_writes_registration_and_version(
        self,
        persister: ToolPersister,
        mock_session: MagicMock,
    ) -> None:
        from src.db.models import ToolRegistration, ToolVersion

        # existence-check SELECT returns "no existing tool"
        existing_result = MagicMock()
        existing_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=existing_result)

        tool_id = await persister.persist(
            tool_name="recall_demo",
            description="counts chars/words/lines",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            handler_code=_VALID_HANDLER,
        )

        assert isinstance(tool_id, uuid.UUID)
        # registration then version
        added = [call.args[0] for call in mock_session.add.call_args_list]
        assert len(added) == 2
        assert isinstance(added[0], ToolRegistration)
        assert added[0].tool_name == "recall_demo"
        assert added[0].tool_type == "generated"
        assert added[0].is_active is True
        assert isinstance(added[1], ToolVersion)
        assert added[1].tool_id == added[0].id
        assert added[1].version == 1
        assert added[1].is_active is True
        assert added[1].code_content == _VALID_HANDLER
        mock_session.flush.assert_awaited()

    @pytest.mark.asyncio
    async def test_persist_returns_none_on_db_error(
        self,
        persister: ToolPersister,
        mock_session: MagicMock,
    ) -> None:
        mock_session.execute = AsyncMock(side_effect=RuntimeError("db down"))
        result = await persister.persist(
            tool_name="boom",
            description="d",
            input_schema={},
            handler_code=_VALID_HANDLER,
        )
        assert result is None


class TestPersistOwnerRunIdAttribution:
    """Track-1: a NEW tool registration is attributed to the active run via
    ``owner_run_id`` (the contextvar the worker runner binds). The UPDATE path
    deliberately keeps ``owner_run_id`` = the original creator, so only the
    new-registration path is asserted here (the live probe_multi_orchestration
    run reused code_executor and created no tool, so this is the regression
    that guards the persist site until a tool-creating Phase-5 run exercises
    it live)."""

    @pytest.mark.asyncio
    async def test_owner_run_id_populated_when_contextvar_bound(
        self,
        persister: ToolPersister,
        mock_session: MagicMock,
    ) -> None:
        from src.db.models import ToolRegistration
        from src.tools._paths import set_active_run_id

        # existence-check SELECT returns "no existing tool" → NEW registration
        existing_result = MagicMock()
        existing_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=existing_result)

        set_active_run_id("attr-run-xyz")
        try:
            await persister.persist(
                tool_name="attributed_tool",
                description="d",
                input_schema={},
                handler_code=_VALID_HANDLER,
            )
        finally:
            set_active_run_id(None)

        added = [call.args[0] for call in mock_session.add.call_args_list]
        assert isinstance(added[0], ToolRegistration)
        assert added[0].owner_run_id == "attr-run-xyz"

    @pytest.mark.asyncio
    async def test_owner_run_id_none_when_contextvar_unset(
        self,
        persister: ToolPersister,
        mock_session: MagicMock,
    ) -> None:
        from src.db.models import ToolRegistration
        from src.tools._paths import set_active_run_id

        existing_result = MagicMock()
        existing_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=existing_result)

        set_active_run_id(None)  # explicit: no active run (operator/CLI origin)
        await persister.persist(
            tool_name="unattributed_tool",
            description="d",
            input_schema={},
            handler_code=_VALID_HANDLER,
        )

        added = [call.args[0] for call in mock_session.add.call_args_list]
        assert isinstance(added[0], ToolRegistration)
        assert added[0].owner_run_id is None


# ---------------------------------------------------------------------------
# load_active_tools() — the cross-run recall path
# ---------------------------------------------------------------------------


def _make_results(reg_row: Any, version_row: Any | None) -> list[MagicMock]:
    """Two SELECT result mocks: registration list, then active version."""
    reg_result = MagicMock()
    reg_result.scalars.return_value.all.return_value = [reg_row]
    ver_result = MagicMock()
    ver_result.scalar_one_or_none.return_value = version_row
    return [reg_result, ver_result]


class TestLoadActiveTools:
    @pytest.mark.asyncio
    async def test_materializes_and_registers_recalled_tool(
        self,
        persister: ToolPersister,
        mock_session: MagicMock,
    ) -> None:
        from src.db.models import ToolRegistration, ToolVersion

        reg_id = uuid.uuid4()
        reg_row = ToolRegistration(
            id=reg_id,
            tool_name="recall_demo",
            tool_type="generated",
            description="counts chars/words/lines",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            is_active=True,
        )
        version_row = ToolVersion(
            tool_id=reg_id,
            version=1,
            code_content=_VALID_HANDLER,
            is_active=True,
        )
        mock_session.execute = AsyncMock(side_effect=_make_results(reg_row, version_row))

        registry = ToolRegistry()
        loaded = await persister.load_active_tools(registry)

        # recall registered the tool, and the materialized handler is live + correct
        assert loaded == ["recall_demo"]
        assert registry.has("recall_demo")
        handler = registry.get_handler("recall_demo")
        assert handler is not None
        out = await handler("hello world")  # type: ignore[misc]
        assert "chars=11" in out and "words=2" in out

    @pytest.mark.asyncio
    async def test_skips_tool_with_no_active_version(
        self,
        persister: ToolPersister,
        mock_session: MagicMock,
    ) -> None:
        from src.db.models import ToolRegistration

        reg_row = ToolRegistration(
            id=uuid.uuid4(),
            tool_name="orphan",
            tool_type="generated",
            description="d",
            input_schema={},
            is_active=True,
        )
        # version SELECT returns None
        mock_session.execute = AsyncMock(side_effect=_make_results(reg_row, None))

        registry = ToolRegistry()
        loaded = await persister.load_active_tools(registry)

        assert loaded == []
        assert not registry.has("orphan")

    @pytest.mark.asyncio
    async def test_db_error_returns_empty(
        self,
        persister: ToolPersister,
        mock_session: MagicMock,
    ) -> None:
        """Recall is best-effort — a DB failure must not raise."""
        mock_session.execute = AsyncMock(side_effect=RuntimeError("db down"))

        registry = ToolRegistry()
        loaded = await persister.load_active_tools(registry)

        assert loaded == []
        assert registry.count == 0
