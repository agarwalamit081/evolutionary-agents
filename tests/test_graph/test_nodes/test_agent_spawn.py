"""Tests for agent_spawn_node from src.graph.nodes.agent_spawn."""

from __future__ import annotations

from typing import Any

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.graph.enums import Phase
from src.graph.nodes.agent_spawn import agent_spawn_node
from src.graph.models import SubAgentSpec


@pytest.fixture
def mock_gateway() -> MagicMock:
    """Create a mock LLMGateway."""
    gateway = MagicMock()
    gateway.acompletion = AsyncMock()
    return gateway


@pytest.fixture
def mock_tools() -> MagicMock:
    """Create a mock ToolRegistry."""
    tools = MagicMock()
    tools.list_names = MagicMock(return_value=["search_tool", "calculate_tool"])
    tools.has = MagicMock(return_value=True)
    return tools


@pytest.fixture
def mock_registry() -> MagicMock:
    """Create a mock SubAgentRegistry."""
    registry = MagicMock()
    registry.has = MagicMock(return_value=False)
    registry.list_names = MagicMock(return_value=[])
    registry.register = MagicMock()
    return registry


@pytest.fixture
def sample_state() -> dict[str, Any]:
    """Create a sample state with pending agent gaps."""
    return {
        "current_goal": MagicMock(text="Main goal: analyze data"),
        "thread_id": "thread-001",
        "strategy": "react",
        "pending_agent_gaps": ["Need a specialist for data analysis"],
        "sub_agents_spawned": [],
    }


class TestAgentSpawnNode:
    """Tests for agent_spawn_node function."""

    @pytest.mark.asyncio
    async def test_no_gaps_returns_execute(self, sample_state: dict[str, Any]) -> None:
        """When no pending_agent_gaps, returns phase=EXECUTE."""
        sample_state["pending_agent_gaps"] = []

        result = await agent_spawn_node(sample_state)

        assert result["phase"] == Phase.EXECUTE
        assert result["pending_agent_gaps"] == []
        assert result["sub_agents_spawned"] == []

    @pytest.mark.asyncio
    async def test_no_gateway_returns_execute(self, sample_state: dict[str, Any]) -> None:
        """When gateway is None, returns phase=EXECUTE."""
        result = await agent_spawn_node(sample_state, gateway=None)

        assert result["phase"] == Phase.EXECUTE
        assert result["pending_agent_gaps"] == []
        assert result["sub_agents_spawned"] == []

    @pytest.mark.asyncio
    async def test_no_registry_returns_execute(self, sample_state: dict[str, Any], mock_gateway: MagicMock) -> None:
        """When sub_agent_registry is None, returns phase=EXECUTE."""
        result = await agent_spawn_node(
            sample_state,
            gateway=mock_gateway,
            sub_agent_registry=None,
        )

        assert result["phase"] == Phase.EXECUTE
        assert result["pending_agent_gaps"] == []
        assert result["sub_agents_spawned"] == []

    @pytest.mark.asyncio
    async def test_spawns_single_agent(self, sample_state: dict[str, Any], mock_gateway: MagicMock, mock_tools: MagicMock, mock_registry: MagicMock) -> None:
        """Successfully spawns a single sub-agent."""
        # Mock LLM response
        from src.llm.models import LLMResponse
        from src.graph.schemas import SubAgentProposal

        mock_gateway.acompletion.return_value = LLMResponse(
            content='{"name": "data_analyzer", "description": "Analyzes data", "goal_description": "Perform data analysis", "template_type": "fixed", "tool_scope": "inherit_all", "tool_subset": [], "model_tier": "simple"}',
            model="gpt-4o-mini",
            provider="openai",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_usd=0.0001,
        )

        mock_registry.has.return_value = False

        with patch("src.llm.structured_output.StructuredOutputManager") as mock_extractor_class:
            mock_extractor = MagicMock()
            mock_extractor.extract = AsyncMock(return_value=SubAgentProposal(
                name="data_analyzer",
                description="Analyzes data",
                goal_description="Perform data analysis",
                template_type="fixed",
                tool_scope="inherit_all",
                tool_subset=[],
                model_tier="simple",
                rationale="Need specialized data analysis",
            ))
            mock_extractor_class.return_value = mock_extractor

            with patch("src.graph.nodes.agent_spawn._persist_agent", new_callable=AsyncMock):
                result = await agent_spawn_node(
                    sample_state,
                    gateway=mock_gateway,
                    tools=mock_tools,
                    sub_agent_registry=mock_registry,
                )

                assert result["phase"] == Phase.DELEGATE
                assert len(result["sub_agents_spawned"]) == 1
                assert result["sub_agents_spawned"][0]["name"] == "data_analyzer"
                assert result["pending_agent_gaps"] == []

    @pytest.mark.asyncio
    async def test_spawns_multiple_agents(self, sample_state: dict[str, Any], mock_gateway: MagicMock, mock_tools: MagicMock, mock_registry: MagicMock) -> None:
        """Successfully spawns multiple sub-agents."""
        from src.llm.models import LLMResponse
        from src.graph.schemas import SubAgentProposal

        sample_state["pending_agent_gaps"] = [
            "Need data analyzer",
            "Need report generator",
        ]

        # Mock alternating responses
        call_count = 0

        async def mock_completion(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(
                    content='{"name": "data_analyzer", "description": "Analyzes data", "goal_description": "Analyze", "template_type": "fixed", "tool_scope": "inherit_all", "tool_subset": [], "model_tier": "simple"}',
                    model="gpt-4o-mini",
                    provider="openai",
                    input_tokens=100,
                    output_tokens=50,
                    total_tokens=150,
                    cost_usd=0.0001,
                )
            else:
                return LLMResponse(
                    content='{"name": "report_gen", "description": "Generates reports", "goal_description": "Generate reports", "template_type": "fixed", "tool_scope": "inherit_all", "tool_subset": [], "model_tier": "simple"}',
                    model="gpt-4o-mini",
                    provider="openai",
                    input_tokens=100,
                    output_tokens=50,
                    total_tokens=150,
                    cost_usd=0.0001,
                )

        mock_gateway.acompletion = mock_completion

        mock_registry.has.return_value = False

        with patch("src.llm.structured_output.StructuredOutputManager") as mock_extractor_class:
            mock_extractor = MagicMock()

            async def mock_extract(content, schema):
                if "data_analyzer" in str(content):
                    return SubAgentProposal(
                        name="data_analyzer",
                        description="Analyzes data",
                        goal_description="Analyze",
                        template_type="fixed",
                        tool_scope="inherit_all",
                        tool_subset=[],
                        model_tier="simple",
                        rationale="Need data analysis specialist",
                    )
                else:
                    return SubAgentProposal(
                        name="report_gen",
                        description="Generates reports",
                        goal_description="Generate reports",
                        template_type="fixed",
                        tool_scope="inherit_all",
                        tool_subset=[],
                        model_tier="simple",
                        rationale="Need report generation specialist",
                    )

            mock_extractor.extract = mock_extract
            mock_extractor_class.return_value = mock_extractor

            with patch("src.graph.nodes.agent_spawn._persist_agent", new_callable=AsyncMock):
                result = await agent_spawn_node(
                    sample_state,
                    gateway=mock_gateway,
                    tools=mock_tools,
                    sub_agent_registry=mock_registry,
                )

                assert result["phase"] == Phase.DELEGATE
                assert len(result["sub_agents_spawned"]) == 2

    @pytest.mark.asyncio
    async def test_respects_max_sub_agents_limit(self, sample_state: dict[str, Any], mock_gateway: MagicMock, mock_tools: MagicMock, mock_registry: MagicMock) -> None:
        """Stops spawning when MAX_SUB_AGENTS_PER_RUN limit reached."""
        from src.agents.registry import MAX_SUB_AGENTS_PER_RUN

        sample_state["pending_agent_gaps"] = [f"gap {i}" for i in range(MAX_SUB_AGENTS_PER_RUN + 2)]
        sample_state["sub_agents_spawned"] = [{"name": f"agent_{i}"} for i in range(MAX_SUB_AGENTS_PER_RUN)]

        # Even though we have gaps, should not spawn any new agents (limit reached)
        result = await agent_spawn_node(
            sample_state,
            gateway=mock_gateway,
            tools=mock_tools,
            sub_agent_registry=mock_registry,
        )

        # No new agents spawned in this call (delta is empty)
        assert len(result["sub_agents_spawned"]) == 0
        # Remaining gaps should be converted to tool gaps, not agent gaps
        assert result["pending_agent_gaps"] == []
        assert len(result["pending_tool_gaps"]) == MAX_SUB_AGENTS_PER_RUN + 2

    @pytest.mark.asyncio
    async def test_validation_failure_skips_agent(self, sample_state: dict[str, Any], mock_gateway: MagicMock, mock_tools: MagicMock, mock_registry: MagicMock) -> None:
        """Skips agent when proposal validation fails."""
        from src.llm.models import LLMResponse
        from src.graph.schemas import SubAgentProposal

        mock_gateway.acompletion.return_value = LLMResponse(
            content='{"name": "invalid-name", "description": "Invalid", "goal_description": "Invalid", "template_type": "fixed", "tool_scope": "invalid", "tool_subset": [], "model_tier": "simple"}',
            model="gpt-4o-mini",
            provider="openai",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_usd=0.0001,
        )

        mock_registry.has.return_value = False

        with patch("src.llm.structured_output.StructuredOutputManager") as mock_extractor_class:
            mock_extractor = MagicMock()
            mock_extractor.extract = AsyncMock(return_value=SubAgentProposal(
                name="invalid-name",
                description="Invalid",
                goal_description="Invalid",
                template_type="fixed",
                tool_scope="invalid",  # Invalid tool scope
                tool_subset=[],
                model_tier="simple",
                rationale="Test rationale",
            ))
            mock_extractor_class.return_value = mock_extractor

            result = await agent_spawn_node(
                sample_state,
                gateway=mock_gateway,
                tools=mock_tools,
                sub_agent_registry=mock_registry,
            )

            # Should not spawn, failed gap should be converted to tool gap
            assert result["phase"] == Phase.EXECUTE
            assert len(result["sub_agents_spawned"]) == 0
            assert result["pending_agent_gaps"] == []
            assert len(result["pending_tool_gaps"]) == 1

    @pytest.mark.asyncio
    async def test_duplicate_name_validation(self, sample_state: dict[str, Any], mock_gateway: MagicMock, mock_tools: MagicMock, mock_registry: MagicMock) -> None:
        """Rejects proposal when agent name already exists."""
        from src.llm.models import LLMResponse
        from src.graph.schemas import SubAgentProposal

        mock_gateway.acompletion.return_value = LLMResponse(
            content='{"name": "existing_agent", "description": "Already exists", "goal_description": "Exists", "template_type": "fixed", "tool_scope": "inherit_all", "tool_subset": [], "model_tier": "simple"}',
            model="gpt-4o-mini",
            provider="openai",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_usd=0.0001,
        )

        mock_registry.has.return_value = True  # Name already taken

        with patch("src.llm.structured_output.StructuredOutputManager") as mock_extractor_class:
            mock_extractor = MagicMock()
            mock_extractor.extract = AsyncMock(return_value=SubAgentProposal(
                name="existing_agent",
                description="Already exists",
                goal_description="Exists",
                template_type="fixed",
                tool_scope="inherit_all",
                tool_subset=[],
                model_tier="simple",
                rationale="Test rationale",
            ))
            mock_extractor_class.return_value = mock_extractor

            result = await agent_spawn_node(
                sample_state,
                gateway=mock_gateway,
                tools=mock_tools,
                sub_agent_registry=mock_registry,
            )

            # Should not spawn duplicate, gap converted to tool gap
            assert len(result["sub_agents_spawned"]) == 0
            assert result["pending_agent_gaps"] == []
            assert len(result["pending_tool_gaps"]) == 1

    @pytest.mark.asyncio
    async def test_llm_failure_returns_none(self, sample_state: dict[str, Any], mock_gateway: MagicMock, mock_tools: MagicMock, mock_registry: MagicMock) -> None:
        """Handles LLM call failure gracefully."""
        mock_gateway.acompletion = AsyncMock(side_effect=Exception("LLM error"))

        result = await agent_spawn_node(
            sample_state,
            gateway=mock_gateway,
            tools=mock_tools,
            sub_agent_registry=mock_registry,
        )

        # Gap should be converted to tool gap, no agent spawned
        assert len(result["sub_agents_spawned"]) == 0
        assert result["pending_agent_gaps"] == []
        assert len(result["pending_tool_gaps"]) == 1

    @pytest.mark.asyncio
    async def test_registers_spawned_agent(self, sample_state: dict[str, Any], mock_gateway: MagicMock, mock_tools: MagicMock, mock_registry: MagicMock) -> None:
        """Registers spawned agent in registry."""
        from src.llm.models import LLMResponse
        from src.graph.schemas import SubAgentProposal

        mock_gateway.acompletion.return_value = LLMResponse(
            content='{"name": "test_agent", "description": "Test", "goal_description": "Test", "template_type": "fixed", "tool_scope": "inherit_all", "tool_subset": [], "model_tier": "simple"}',
            model="gpt-4o-mini",
            provider="openai",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_usd=0.0001,
        )

        mock_registry.has.return_value = False

        with patch("src.llm.structured_output.StructuredOutputManager") as mock_extractor_class:
            mock_extractor = MagicMock()
            mock_extractor.extract = AsyncMock(return_value=SubAgentProposal(
                name="test_agent",
                description="Test",
                goal_description="Test",
                template_type="fixed",
                tool_scope="inherit_all",
                tool_subset=[],
                model_tier="simple",
                rationale="Test rationale",
            ))
            mock_extractor_class.return_value = mock_extractor

            with patch("src.graph.nodes.agent_spawn._persist_agent", new_callable=AsyncMock):
                await agent_spawn_node(
                    sample_state,
                    gateway=mock_gateway,
                    tools=mock_tools,
                    sub_agent_registry=mock_registry,
                )

                # Should have called register
                mock_registry.register.assert_called_once()
                call_args = mock_registry.register.call_args
                registered_spec = call_args[0][0]
                assert isinstance(registered_spec, SubAgentSpec)
                assert registered_spec.name == "test_agent"


class TestValidateProposal:
    """Tests for _validate_proposal() helper function."""

    def test_validate_proposal_valid(self, mock_registry: MagicMock, mock_tools: MagicMock) -> None:
        """Validation passes for valid proposal."""
        from src.graph.schemas import SubAgentProposal
        from src.graph.nodes.agent_spawn import _validate_proposal

        proposal = SubAgentProposal(
            name="valid_agent",
            description="Valid agent",
            goal_description="Valid goal",
            template_type="fixed",
            tool_scope="inherit_all",
            tool_subset=[],
            model_tier="simple",
            rationale="Test rationale",
        )

        errors = _validate_proposal(proposal, mock_registry, mock_tools)
        assert errors == []

    def test_validate_proposal_duplicate_name(self, mock_registry: MagicMock, mock_tools: MagicMock) -> None:
        """Validation fails when name already exists."""
        from src.graph.schemas import SubAgentProposal
        from src.graph.nodes.agent_spawn import _validate_proposal

        mock_registry.has.return_value = True

        proposal = SubAgentProposal(
            name="existing_agent",
            description="Existing",
            goal_description="Existing",
            template_type="fixed",
            tool_scope="inherit_all",
            tool_subset=[],
            model_tier="simple",
            rationale="Test rationale",
        )

        errors = _validate_proposal(proposal, mock_registry, mock_tools)
        assert len(errors) > 0
        assert any("already exists" in e for e in errors)

    def test_validate_proposal_invalid_name_format(self, mock_registry: MagicMock, mock_tools: MagicMock) -> None:
        """Validation fails for non-snake_case names."""
        from src.graph.schemas import SubAgentProposal
        from src.graph.nodes.agent_spawn import _validate_proposal

        proposal = SubAgentProposal(
            name="Invalid-Name",
            description="Invalid",
            goal_description="Invalid",
            template_type="fixed",
            tool_scope="inherit_all",
            tool_subset=[],
            model_tier="simple",
            rationale="Test rationale",
        )

        errors = _validate_proposal(proposal, mock_registry, mock_tools)
        assert len(errors) > 0
        assert any("must be snake_case" in e for e in errors)

    def test_validate_proposal_missing_tool_in_subset(self, mock_registry: MagicMock, mock_tools: MagicMock) -> None:
        """Validation fails when requested tool not in registry."""
        from src.graph.schemas import SubAgentProposal
        from src.graph.nodes.agent_spawn import _validate_proposal

        mock_tools.has.return_value = False

        proposal = SubAgentProposal(
            name="agent",
            description="Agent",
            goal_description="Goal",
            template_type="fixed",
            tool_scope="inherit_subset",
            tool_subset=["missing_tool"],
            model_tier="simple",
            rationale="Test rationale",
        )

        errors = _validate_proposal(proposal, mock_registry, mock_tools)
        assert len(errors) > 0
        assert any("not found in registry" in e for e in errors)

    def test_validate_proposal_invalid_template_type(self, mock_registry: MagicMock, mock_tools: MagicMock) -> None:
        """Validation fails for invalid template_type."""
        from src.graph.schemas import SubAgentProposal
        from src.graph.nodes.agent_spawn import _validate_proposal

        proposal = SubAgentProposal(
            name="agent",
            description="Agent",
            goal_description="Goal",
            template_type="invalid",
            tool_scope="inherit_all",
            tool_subset=[],
            model_tier="simple",
            rationale="Test rationale",
        )

        errors = _validate_proposal(proposal, mock_registry, mock_tools)
        assert len(errors) > 0
        assert any("Invalid template_type" in e for e in errors)

    def test_validate_proposal_invalid_tool_scope(self, mock_registry: MagicMock, mock_tools: MagicMock) -> None:
        """Validation fails for invalid tool_scope."""
        from src.graph.schemas import SubAgentProposal
        from src.graph.nodes.agent_spawn import _validate_proposal

        proposal = SubAgentProposal(
            name="agent",
            description="Agent",
            goal_description="Goal",
            template_type="fixed",
            tool_scope="invalid_scope",
            tool_subset=[],
            model_tier="simple",
            rationale="Test rationale",
        )

        errors = _validate_proposal(proposal, mock_registry, mock_tools)
        assert len(errors) > 0
        assert any("Invalid tool_scope" in e for e in errors)


class TestParseModelTier:
    """Tests for _parse_model_tier() helper function."""

    def test_parse_model_tier_valid(self) -> None:
        """Parses valid model tier strings."""
        from src.graph.nodes.agent_spawn import _parse_model_tier

        result = _parse_model_tier("simple")
        assert result == "simple"

        result = _parse_model_tier("complex")
        assert result == "complex"

        result = _parse_model_tier("critical")
        assert result == "critical"

    def test_parse_model_tier_invalid_defaults_to_simple(self) -> None:
        """Defaults to 'simple' for invalid tier."""
        from src.graph.nodes.agent_spawn import _parse_model_tier

        result = _parse_model_tier("invalid_tier")
        assert result == "simple"


class TestPersistAgent:
    """Tests for _persist_agent() helper function."""

    @pytest.mark.asyncio
    async def test_persist_agent_success(self) -> None:
        """Successfully persists agent to DB."""
        from src.graph.nodes.agent_spawn import _persist_agent
        from src.graph.models import SubAgentSpec

        spec = SubAgentSpec(
            name="persist_test",
            description="Test persistence",
            goal="test",
            parent_thread_id="thread-001",
        )

        with patch("src.agents.persister.SubAgentPersister") as mock_persister_class:
            mock_persister = MagicMock()
            mock_persister.persist = AsyncMock(return_value="agent-uuid-123")
            mock_persister_class.return_value = mock_persister

            await _persist_agent(spec)

            mock_persister.persist.assert_called_once_with(spec)

    @pytest.mark.asyncio
    async def test_persist_agent_failure_non_fatal(self) -> None:
        """Persistence failure is non-fatal, logged and continues."""
        from src.graph.nodes.agent_spawn import _persist_agent
        from src.graph.models import SubAgentSpec

        spec = SubAgentSpec(
            name="persist_fail",
            description="Test persistence failure",
            goal="test",
            parent_thread_id="thread-001",
        )

        with patch("src.agents.persister.SubAgentPersister") as mock_persister_class:
            mock_persister = MagicMock()
            mock_persister.persist = AsyncMock(side_effect=Exception("DB error"))
            mock_persister_class.return_value = mock_persister

            # Should not raise exception
            await _persist_agent(spec)
