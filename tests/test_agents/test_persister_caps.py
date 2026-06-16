"""Tests for cumulative cap enforcement in SubAgentPersister (B3).

``load_active_agents(registry, settings)`` runs two de-bloat passes around the
load: ``retire_redundant`` (mocked here — exercised against real pgvector in
test_persister_find_similar_integration.py) before load, then the registry's
real ``enforce_caps`` after load, whose retired names are persisted via
``retire``. ``settings=None`` skips both passes (the raw recall path covered by
test_persister_recall.py).
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
from src.config.settings import AgentSettings


def _model_row(
    name: str, *, total_runs: int = 0, success_rate: float = 1.0
) -> Any:
    from src.db.models import SubAgentModel

    return SubAgentModel(
        id=uuid.uuid4(),
        name=name,
        description="d",
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
        is_active=True,
        version=1,
        total_runs=total_runs,
        success_rate=success_rate,
        avg_cost=0.01,
        avg_latency_ms=500,
        quality_score=0.8,
    )


def _settings(**overrides: Any) -> AgentSettings:
    base: dict[str, Any] = {
        "max_active_sub_agents": 15,
        "retire_min_runs": 10,
        "retire_success_floor": 0.3,
        "retire_recency_days": 30,
        "capability_redundancy_threshold": 0.92,
    }
    base.update(overrides)
    return AgentSettings(_env_file=None, **base)


@pytest.fixture
def persister_with_spies() -> Any:
    """Persister with get_session faked; retire_redundant + retire spied."""
    session = MagicMock()
    session.execute = AsyncMock()

    @asynccontextmanager
    async def _fake_get_session() -> AsyncGenerator[MagicMock, None]:
        yield session

    with patch("src.db.session.get_session", new=_fake_get_session):
        persister = SubAgentPersister()
        # retire_redundant is DB-backed; these tests assert the load wiring, so
        # stub it to a no-op (its real cosine path is covered by the integration
        # test). retire is spied to assert retired names are persisted.
        persister.retire_redundant = AsyncMock(return_value=[])
        retire_mock = AsyncMock(return_value=0)
        persister.retire = retire_mock
        yield persister, session, retire_mock


class TestLoadActiveAgentsCaps:
    @pytest.mark.asyncio
    async def test_retires_low_performer_and_persists(
        self, persister_with_spies: Any
    ) -> None:
        persister, session, retire_mock = persister_with_spies
        result = MagicMock()
        result.scalars.return_value.all.return_value = [
            _model_row("bad", total_runs=15, success_rate=0.1),
            _model_row("good", total_runs=15, success_rate=0.9),
        ]
        session.execute = AsyncMock(return_value=result)

        registry = SubAgentRegistry()
        loaded = await persister.load_active_agents(
            registry, settings=_settings()
        )

        # "bad" retired (0.1 < 0.3 over 15>=10 runs); "good" kept.
        assert loaded == ["good"]
        assert retire_mock.await_count == 1
        assert retire_mock.await_args_list[-1].args[0] == ["bad"]
        assert registry.get("bad").is_active is False

    @pytest.mark.asyncio
    async def test_enforces_cap_when_over_limit(
        self, persister_with_spies: Any
    ) -> None:
        persister, session, retire_mock = persister_with_spies
        result = MagicMock()
        result.scalars.return_value.all.return_value = [
            _model_row("low", total_runs=15, success_rate=0.6),
            _model_row("mid", total_runs=15, success_rate=0.7),
            _model_row("high", total_runs=15, success_rate=0.9),
        ]
        session.execute = AsyncMock(return_value=result)

        registry = SubAgentRegistry()
        loaded = await persister.load_active_agents(
            registry, settings=_settings(max_active_sub_agents=2)
        )

        # Overflow (3 > 2) retires lowest score; none are bad/stale.
        assert sorted(loaded) == ["high", "mid"]
        assert retire_mock.await_args_list[-1].args[0] == ["low"]
        assert registry.active_count == 2

    @pytest.mark.asyncio
    async def test_no_settings_skips_enforcement(
        self, persister_with_spies: Any
    ) -> None:
        """settings=None is the raw recall path — no cap/retire calls."""
        persister, session, retire_mock = persister_with_spies
        result = MagicMock()
        result.scalars.return_value.all.return_value = [
            _model_row("bad", total_runs=15, success_rate=0.1),
        ]
        session.execute = AsyncMock(return_value=result)

        registry = SubAgentRegistry()
        loaded = await persister.load_active_agents(registry, settings=None)

        assert loaded == ["bad"]
        assert registry.get("bad").is_active is True
        retire_mock.assert_not_awaited()
        persister.retire_redundant.assert_not_awaited()
