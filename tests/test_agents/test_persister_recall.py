"""Tests for src.agents.persister — persist + cross-run recall wiring.

Mock-session unit tests mirroring tests/test_evolution/test_persister.py: the
real ORM models (SubAgentModel) are instantiated (constructors are DB-free);
only get_session() is faked. The load_active_agents tests fake the SELECT
result and assert the row is converted to a SubAgentSpec and registered into a
live SubAgentRegistry — i.e. the recall path main.py runs on every startup.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.persister import SubAgentPersister
from src.agents.registry import SubAgentRegistry
from src.graph.enums import TaskComplexity
from src.graph.models import SubAgentSpec


def _spec(name: str = "recall_agent") -> SubAgentSpec:
    return SubAgentSpec(
        name=name,
        description="inventories python files",
        goal="",
        parent_thread_id="",
        model_tier=TaskComplexity.SIMPLE,
        tool_scope="inherit_all",
        tool_subset=[],
        max_iterations=8,
        is_active=True,
    )


@pytest.fixture
def mock_session() -> MagicMock:
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
    @asynccontextmanager
    async def _fake_get_session() -> AsyncGenerator[MagicMock, None]:
        yield mock_session

    with patch("src.db.session.get_session", new=_fake_get_session):
        yield SubAgentPersister()


# ---------------------------------------------------------------------------
# persist()
# ---------------------------------------------------------------------------


class TestPersist:
    @pytest.mark.asyncio
    async def test_persist_new_agent_writes_model(
        self,
        persister: SubAgentPersister,
        mock_session: MagicMock,
    ) -> None:
        from src.db.models import SubAgentModel

        existing_result = MagicMock()
        existing_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=existing_result)

        agent_id = await persister.persist(_spec("recall_agent"))

        assert isinstance(agent_id, uuid.UUID)
        mock_session.add.assert_called_once()
        added = mock_session.add.call_args[0][0]
        assert isinstance(added, SubAgentModel)
        assert added.name == "recall_agent"
        assert added.version == 1
        assert added.is_active is True
        mock_session.flush.assert_awaited()

    @pytest.mark.asyncio
    async def test_persist_returns_none_on_db_error(
        self,
        persister: SubAgentPersister,
        mock_session: MagicMock,
    ) -> None:
        mock_session.execute = AsyncMock(side_effect=RuntimeError("db down"))
        assert await persister.persist(_spec()) is None


# ---------------------------------------------------------------------------
# load_active_agents() — the cross-run recall path
# ---------------------------------------------------------------------------


def _model_row(*, name: str = "recall_agent", is_active: bool = True) -> Any:
    from src.db.models import SubAgentModel

    return SubAgentModel(
        id=uuid.uuid4(),
        name=name,
        description="inventories python files",
        template_type="fixed",
        tool_scope="inherit_all",
        tool_subset=[],
        budget_mode="shared",
        budget_limit=0.0,
        model_tier="simple",
        max_iterations=8,
        depth_limit=0,
        node_config={},
        system_prompt_override=None,
        is_active=is_active,
        version=1,
        total_runs=3,
        success_rate=1.0,
        avg_cost=0.01,
        avg_latency_ms=500,
        quality_score=0.8,
    )


class TestLoadActiveAgents:
    @pytest.mark.asyncio
    async def test_registers_recalled_agent(
        self,
        persister: SubAgentPersister,
        mock_session: MagicMock,
    ) -> None:
        row = _model_row(name="recall_agent", is_active=True)
        result = MagicMock()
        result.scalars.return_value.all.return_value = [row]
        mock_session.execute = AsyncMock(return_value=result)

        registry = SubAgentRegistry()
        loaded = await persister.load_active_agents(registry)

        assert loaded == ["recall_agent"]
        assert registry.has("recall_agent")
        recalled = registry.get("recall_agent")
        assert recalled is not None
        assert recalled.is_active is True
        assert "recall_agent" in registry.describe_agents()

    @pytest.mark.asyncio
    async def test_db_error_returns_empty(
        self,
        persister: SubAgentPersister,
        mock_session: MagicMock,
    ) -> None:
        """Recall is best-effort — a DB failure must not raise."""
        mock_session.execute = AsyncMock(side_effect=RuntimeError("db down"))

        registry = SubAgentRegistry()
        loaded = await persister.load_active_agents(registry)

        assert loaded == []
        assert registry.count == 0
