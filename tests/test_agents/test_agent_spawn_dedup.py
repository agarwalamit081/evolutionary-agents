"""Tests for the semantic dedup block in ``agent_spawn_node`` (B3).

Before spawning a new sub-agent, the node embeds the capability (gap + proposal)
and reuses an existing active sub-agent whose capability is semantically
identical (cosine >= capability_dedup_threshold) and already registered. Covers:

  * reuse-above-threshold -> returns ``reused`` True, skips spawn+register+persist
  * create-no-match       -> spawns, persists the embedding (api source)
  * skip-on-hash-fallback -> hash vectors are not deduped; persistence gets None
  * failure-degrades      -> a dedup error never blocks spawning

The dedup block runs AFTER proposal validation, so each test wires a valid
SubAgentProposal through the gateway + StructuredOutputManager (mirroring
``tests/test_graph/test_nodes/test_agent_spawn.py``).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.graph.enums import Phase
from src.graph.models import SubAgentSpec
from src.graph.nodes.agent_spawn import agent_spawn_node


@pytest.fixture
def mock_gateway() -> MagicMock:
    from src.llm.models import LLMResponse

    gateway = MagicMock()
    gateway.acompletion = AsyncMock(
        return_value=LLMResponse(
            content='{"name": "new_analyzer", "description": "Analyzes data", '
            '"goal_description": "Perform data analysis", "template_type": "fixed", '
            '"tool_scope": "inherit_all", "tool_subset": [], "model_tier": "simple"}',
            model="gpt-4o-mini",
            provider="openai",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_usd=0.0001,
        )
    )
    return gateway


@pytest.fixture
def mock_tools() -> MagicMock:
    tools = MagicMock()
    tools.list_names = MagicMock(return_value=["search_tool"])
    tools.has = MagicMock(return_value=True)
    return tools


@pytest.fixture
def mock_registry() -> MagicMock:
    registry = MagicMock()
    registry.has = MagicMock(return_value=False)  # proposed name is new
    registry.list_names = MagicMock(return_value=[])
    registry.register = MagicMock()
    registry.get = MagicMock(return_value=None)
    return registry


@pytest.fixture
def sample_state() -> dict[str, Any]:
    return {
        "current_goal": MagicMock(text="Main goal: analyze data"),
        "thread_id": "thread-001",
        "strategy": "react",
        "pending_agent_gaps": ["Need a specialist for data analysis"],
        "sub_agents_spawned": [],
    }


def _proposal() -> Any:
    from src.graph.schemas import SubAgentProposal

    return SubAgentProposal(
        name="new_analyzer",
        description="Analyzes data",
        goal_description="Perform data analysis",
        template_type="fixed",
        tool_scope="inherit_all",
        tool_subset=[],
        model_tier="simple",
        rationale="Need specialized data analysis",
    )


def _patch_proposal_extractor():
    """Patch StructuredOutputManager to always yield a valid proposal."""
    extractor = MagicMock()
    extractor.extract = AsyncMock(return_value=_proposal())
    return patch("src.llm.structured_output.StructuredOutputManager", return_value=extractor)


class TestAgentSpawnDedup:
    @pytest.mark.asyncio
    async def test_reuses_existing_agent_above_threshold(
        self,
        sample_state: dict[str, Any],
        mock_gateway: MagicMock,
        mock_tools: MagicMock,
        mock_registry: MagicMock,
    ) -> None:
        existing_spec = SubAgentSpec(
            name="existing_analyzer",
            description="Existing data analyzer",
            goal="x",
            parent_thread_id="t",
        )
        embed_mock = AsyncMock(return_value=([0.1] * 768, "api"))
        persister_inst = MagicMock()
        persister_inst.find_similar = AsyncMock(
            return_value=[{"name": "existing_analyzer", "description": "d", "similarity": 0.95}]
        )
        # The semantically-identical existing agent is registered -> reusable.
        mock_registry.get = MagicMock(return_value=existing_spec)

        persist_mock = AsyncMock()
        with (
            _patch_proposal_extractor(),
            patch("src.memory.embeddings.embed_capability", embed_mock),
            patch("src.agents.persister.SubAgentPersister", return_value=persister_inst),
            patch("src.graph.nodes.agent_spawn._persist_agent", new=persist_mock),
        ):
            result = await agent_spawn_node(
                sample_state,
                gateway=mock_gateway,
                tools=mock_tools,
                sub_agent_registry=mock_registry,
            )

        spawned = result["sub_agents_spawned"][0]
        assert spawned["reused"] is True
        assert spawned["name"] == "existing_analyzer"
        # Did NOT register a new agent or persist a new one.
        mock_registry.register.assert_not_called()
        persist_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_spawns_when_no_match_above_threshold(
        self,
        sample_state: dict[str, Any],
        mock_gateway: MagicMock,
        mock_tools: MagicMock,
        mock_registry: MagicMock,
    ) -> None:
        embed_mock = AsyncMock(return_value=([0.1] * 768, "api"))
        persister_inst = MagicMock()
        persister_inst.find_similar = AsyncMock(return_value=[])  # nothing similar

        persist_mock = AsyncMock()
        with (
            _patch_proposal_extractor(),
            patch("src.memory.embeddings.embed_capability", embed_mock),
            patch("src.agents.persister.SubAgentPersister", return_value=persister_inst),
            patch("src.graph.nodes.agent_spawn._persist_agent", new=persist_mock),
        ):
            result = await agent_spawn_node(
                sample_state,
                gateway=mock_gateway,
                tools=mock_tools,
                sub_agent_registry=mock_registry,
            )

        spawned = result["sub_agents_spawned"][0]
        assert "reused" not in spawned
        assert spawned["name"] == "new_analyzer"
        mock_registry.register.assert_called_once()
        # The real "api" embedding is persisted so future gaps reuse this agent.
        persist_mock.assert_awaited_once()
        assert persist_mock.await_args_list[-1].kwargs["capability_embedding"] == [0.1] * 768

    @pytest.mark.asyncio
    async def test_skips_dedup_on_hash_fallback(
        self,
        sample_state: dict[str, Any],
        mock_gateway: MagicMock,
        mock_tools: MagicMock,
        mock_registry: MagicMock,
    ) -> None:
        embed_mock = AsyncMock(return_value=([0.2] * 768, "hash"))
        persister_inst = MagicMock()
        persister_inst.find_similar = AsyncMock(
            return_value=[{"name": "existing_analyzer", "similarity": 0.95}]
        )

        persist_mock = AsyncMock()
        with (
            _patch_proposal_extractor(),
            patch("src.memory.embeddings.embed_capability", embed_mock),
            patch("src.agents.persister.SubAgentPersister", return_value=persister_inst),
            patch("src.graph.nodes.agent_spawn._persist_agent", new=persist_mock),
        ):
            result = await agent_spawn_node(
                sample_state,
                gateway=mock_gateway,
                tools=mock_tools,
                sub_agent_registry=mock_registry,
            )

        spawned = result["sub_agents_spawned"][0]
        assert "reused" not in spawned
        persister_inst.find_similar.assert_not_awaited()
        # Persistence must NOT receive a hash vector.
        assert persist_mock.await_args_list[-1].kwargs["capability_embedding"] is None
        assert persist_mock.await_args_list[-1].kwargs["capability_text"] is None

    @pytest.mark.asyncio
    async def test_dedup_failure_degrades_to_spawn(
        self,
        sample_state: dict[str, Any],
        mock_gateway: MagicMock,
        mock_tools: MagicMock,
        mock_registry: MagicMock,
    ) -> None:
        embed_mock = AsyncMock(return_value=([0.1] * 768, "api"))
        persister_inst = MagicMock()
        persister_inst.find_similar = AsyncMock(side_effect=RuntimeError("db down"))

        with (
            _patch_proposal_extractor(),
            patch("src.memory.embeddings.embed_capability", embed_mock),
            patch("src.agents.persister.SubAgentPersister", return_value=persister_inst),
            patch("src.graph.nodes.agent_spawn._persist_agent", new=AsyncMock()),
        ):
            result = await agent_spawn_node(
                sample_state,
                gateway=mock_gateway,
                tools=mock_tools,
                sub_agent_registry=mock_registry,
            )

        assert result["phase"] == Phase.DELEGATE
        spawned = result["sub_agents_spawned"][0]
        assert spawned["name"] == "new_analyzer"
        mock_registry.register.assert_called_once()
