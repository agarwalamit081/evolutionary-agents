"""Tests for delegate_node from src.graph.nodes.delegate."""

from __future__ import annotations

from itertools import chain, repeat
from typing import Any

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.config.settings import Settings
from src.graph.enums import Phase
from src.graph.models import SubAgentSpec
from src.graph.nodes.delegate import delegate_node


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
    tools.list_names = MagicMock(return_value=["tool1", "tool2"])
    return tools


@pytest.fixture
def mock_registry() -> MagicMock:
    """Create a mock SubAgentRegistry."""
    registry = MagicMock()
    registry.get = MagicMock(return_value=None)
    registry.check_deprecation = MagicMock(return_value=False)
    return registry


@pytest.fixture
def sample_spec() -> SubAgentSpec:
    """Create a sample SubAgentSpec."""
    return SubAgentSpec(
        name="delegate_test_agent",
        description="Test agent for delegation",
        goal="test delegation goal",
        parent_thread_id="thread-001",
        tool_scope="inherit_all",
    )


@pytest.fixture
def sample_state() -> dict[str, Any]:
    """Create a sample state with spawned agents."""
    return {
        "current_goal": MagicMock(text="Main goal: coordinate subtasks"),
        "thread_id": "thread-001",
        "sub_agents_spawned": [
            {"name": "agent1", "id": "id1"},
            {"name": "agent2", "id": "id2"},
        ],
        "pending_agent_gaps": [],
    }


class TestDelegateNode:
    """Tests for delegate_node function."""

    @pytest.mark.asyncio
    async def test_no_spawned_returns_execute(self, sample_state: dict[str, Any]) -> None:
        """When no sub_agents_spawned, returns phase=EXECUTE."""
        sample_state["sub_agents_spawned"] = []

        result = await delegate_node(sample_state)

        assert result["phase"] == Phase.EXECUTE
        assert result["delegation_results"] == []

    @pytest.mark.asyncio
    async def test_no_gateway_returns_execute(self, sample_state: dict[str, Any], mock_tools: MagicMock, mock_registry: MagicMock) -> None:
        """When gateway is None, returns phase=EXECUTE."""
        result = await delegate_node(
            sample_state,
            gateway=None,
            tools=mock_tools,
            sub_agent_registry=mock_registry,
        )

        assert result["phase"] == Phase.EXECUTE
        assert result["delegation_results"] == []

    @pytest.mark.asyncio
    async def test_no_registry_returns_execute(self, sample_state: dict[str, Any], mock_gateway: MagicMock, mock_tools: MagicMock) -> None:
        """When sub_agent_registry is None, returns phase=EXECUTE."""
        result = await delegate_node(
            sample_state,
            gateway=mock_gateway,
            tools=mock_tools,
            sub_agent_registry=None,
        )

        assert result["phase"] == Phase.EXECUTE
        assert result["delegation_results"] == []

    @pytest.mark.asyncio
    async def test_no_tools_returns_error_results(self, sample_state: dict[str, Any], mock_gateway: MagicMock, mock_registry: MagicMock, sample_spec: SubAgentSpec) -> None:
        """When tools is None, returns error results for all agents."""
        mock_registry.get.return_value = sample_spec

        result = await delegate_node(
            sample_state,
            gateway=mock_gateway,
            tools=None,
            sub_agent_registry=mock_registry,
        )

        assert result["phase"] == Phase.EXECUTE
        assert len(result["delegation_results"]) == 2
        assert all(r["success"] is False for r in result["delegation_results"])
        assert any("No tool registry available" in str(e) for r in result["delegation_results"] for e in r.get("errors", []))

    @pytest.mark.asyncio
    async def test_delegates_single_agent_success(self, sample_state: dict[str, Any], mock_gateway: MagicMock, mock_tools: MagicMock, mock_registry: MagicMock, sample_spec: SubAgentSpec) -> None:
        """Successfully delegates to a single agent."""
        sample_state["sub_agents_spawned"] = [
            {"name": "test_agent", "id": "test-id"},
        ]
        mock_registry.get.return_value = sample_spec

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value={
            "success": True,
            "result": "Subtask completed",
            "tokens_used": 100,
            "cost_usd": 0.01,
            "latency_ms": 50,
            "iterations": 1,
            "errors": [],
            "goal": "subtask",
            "sub_agent_name": "test_agent",
            "sub_agent_id": "test-id",
        })
        mock_registry.spawn.return_value = mock_runner

        result = await delegate_node(
            sample_state,
            gateway=mock_gateway,
            tools=mock_tools,
            sub_agent_registry=mock_registry,
        )

        assert result["phase"] == Phase.VERIFY
        assert len(result["delegation_results"]) == 1
        assert result["delegation_results"][0]["success"] is True
        assert result["delegation_results"][0]["result"] == "Subtask completed"

    @pytest.mark.asyncio
    async def test_surfaces_delegated_tool_activity_in_parent_state(
        self,
        sample_state: dict[str, Any],
        mock_gateway: MagicMock,
        mock_tools: MagicMock,
        mock_registry: MagicMock,
        sample_spec: SubAgentSpec,
    ) -> None:
        """Sub-agent tool_results/tools_created/tools_called surface in parent state.

        Previously delegated tool activity was siloed in SubAgentState; the
        delegate node now aggregates it so the parent's reducer-backed lists
        (and the e2e report) reflect real tool work during delegation.
        """
        from src.graph.models import ToolResult

        sample_state["sub_agents_spawned"] = [
            {"name": "test_agent", "id": "test-id"},
        ]
        mock_registry.get.return_value = sample_spec

        inner_tool_result = ToolResult(
            tool_name="code_executor",
            success=True,
            output="ran inside sub-agent",
        )
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value={
            "success": True,
            "result": "Subtask completed",
            "tokens_used": 100,
            "cost_usd": 0.01,
            "latency_ms": 50,
            "iterations": 1,
            "errors": [],
            "goal": "subtask",
            "sub_agent_name": "test_agent",
            "sub_agent_id": "test-id",
            # Propagated from SubAgentState by runner._extract_results.
            "tool_results": [inner_tool_result],
            "tools_created": [{"name": "gen_tool", "description": "generated"}],
            "tools_called": [{"name": "code_executor"}],
        })
        mock_registry.spawn.return_value = mock_runner

        result = await delegate_node(
            sample_state,
            gateway=mock_gateway,
            tools=mock_tools,
            sub_agent_registry=mock_registry,
        )

        # Parent tool_results include the sub_agent:* wrapper AND the
        # sub-agent's internal tool activity.
        tool_names = [getattr(tr, "tool_name", None) for tr in result["tool_results"]]
        assert "sub_agent:test_agent" in tool_names
        assert "code_executor" in tool_names
        # tools_created/tools_called propagate to the parent state.
        assert result["tools_created"] == [{"name": "gen_tool", "description": "generated"}]
        assert result["tools_called"] == [{"name": "code_executor"}]

    @pytest.mark.asyncio
    async def test_delegates_multiple_agents(self, sample_state: dict[str, Any], mock_gateway: MagicMock, mock_tools: MagicMock, mock_registry: MagicMock, sample_spec: SubAgentSpec) -> None:
        """Successfully delegates to multiple agents."""
        mock_registry.get.return_value = sample_spec

        # Mock successful runs
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value={
            "success": True,
            "result": "Done",
            "tokens_used": 100,
            "cost_usd": 0.01,
            "latency_ms": 50,
            "iterations": 1,
            "errors": [],
            "goal": "subtask",
            "sub_agent_name": "agent",
            "sub_agent_id": "id",
        })
        mock_registry.spawn.return_value = mock_runner

        result = await delegate_node(
            sample_state,
            gateway=mock_gateway,
            tools=mock_tools,
            sub_agent_registry=mock_registry,
        )

        assert result["phase"] == Phase.VERIFY
        assert len(result["delegation_results"]) == 2

    @pytest.mark.asyncio
    async def test_delegation_failure_routes_to_execute(self, sample_state: dict[str, Any], mock_gateway: MagicMock, mock_tools: MagicMock, mock_registry: MagicMock, sample_spec: SubAgentSpec) -> None:
        """When delegation fails, routes to EXECUTE."""
        mock_registry.get.return_value = sample_spec

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value={
            "success": False,
            "result": "",
            "tokens_used": 0,
            "cost_usd": 0.0,
            "latency_ms": 10,
            "iterations": 0,
            "errors": ["Subtask failed"],
            "goal": "subtask",
            "sub_agent_name": "agent",
            "sub_agent_id": "id",
        })
        mock_registry.spawn.return_value = mock_runner

        result = await delegate_node(
            sample_state,
            gateway=mock_gateway,
            tools=mock_tools,
            sub_agent_registry=mock_registry,
        )

        assert result["phase"] == Phase.EXECUTE
        assert len(result["delegation_results"]) == 2
        assert all(r["success"] is False for r in result["delegation_results"])

    @pytest.mark.asyncio
    async def test_mixed_success_failure_routes_to_execute(self, sample_state: dict[str, Any], mock_gateway: MagicMock, mock_tools: MagicMock, mock_registry: MagicMock, sample_spec: SubAgentSpec) -> None:
        """When some delegations succeed and some fail, routes to EXECUTE."""
        mock_registry.get.return_value = sample_spec

        mock_runner = MagicMock()

        # First delegation succeeds, all subsequent ones fail ("mixed" outcome).
        mock_runner.run = AsyncMock(side_effect=chain(
            [{
                "success": True,
                "result": "Success",
                "tokens_used": 100,
                "cost_usd": 0.01,
                "latency_ms": 50,
                "iterations": 1,
                "errors": [],
                "goal": "subtask",
                "sub_agent_name": "agent1",
                "sub_agent_id": "id1",
            }],
            repeat({
                "success": False,
                "result": "",
                "tokens_used": 0,
                "cost_usd": 0.0,
                "latency_ms": 10,
                "iterations": 0,
                "errors": ["Failed"],
                "goal": "subtask",
                "sub_agent_name": "agent2",
                "sub_agent_id": "id2",
            }),
        ))
        mock_registry.spawn.return_value = mock_runner

        result = await delegate_node(
            sample_state,
            gateway=mock_gateway,
            tools=mock_tools,
            sub_agent_registry=mock_registry,
        )

        assert result["phase"] == Phase.EXECUTE
        assert len(result["delegation_results"]) == 2

    @pytest.mark.asyncio
    async def test_creates_tool_results_for_success(self, sample_state: dict[str, Any], mock_gateway: MagicMock, mock_tools: MagicMock, mock_registry: MagicMock, sample_spec: SubAgentSpec) -> None:
        """Creates ToolResult entries for successful delegations."""
        sample_state["sub_agents_spawned"] = [
            {"name": "success_agent", "id": "success-id"},
        ]
        mock_registry.get.return_value = sample_spec

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value={
            "success": True,
            "result": "Task completed",
            "tokens_used": 150,
            "cost_usd": 0.015,
            "latency_ms": 75,
            "iterations": 2,
            "errors": [],
            "goal": "subtask",
            "sub_agent_name": "success_agent",
            "sub_agent_id": "success-id",
        })
        mock_registry.spawn.return_value = mock_runner

        result = await delegate_node(
            sample_state,
            gateway=mock_gateway,
            tools=mock_tools,
            sub_agent_registry=mock_registry,
        )

        assert "tool_results" in result
        assert len(result["tool_results"]) == 1
        tool_result = result["tool_results"][0]
        assert tool_result.tool_name == "sub_agent:success_agent"
        assert tool_result.success is True
        assert tool_result.output == "Task completed"
        assert tool_result.tokens_used == 150
        assert tool_result.duration_ms == 75

    @pytest.mark.asyncio
    async def test_creates_tool_results_for_failure(self, sample_state: dict[str, Any], mock_gateway: MagicMock, mock_tools: MagicMock, mock_registry: MagicMock, sample_spec: SubAgentSpec) -> None:
        """Creates ToolResult entries for failed delegations."""
        sample_state["sub_agents_spawned"] = [
            {"name": "failing_agent", "id": "failing-id"},
        ]
        mock_registry.get.return_value = sample_spec

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value={
            "success": False,
            "result": "",
            "tokens_used": 50,
            "cost_usd": 0.005,
            "latency_ms": 25,
            "iterations": 0,
            "errors": ["Timeout error"],
            "goal": "subtask",
            "sub_agent_name": "failing_agent",
            "sub_agent_id": "failing-id",
        })
        mock_registry.spawn.return_value = mock_runner

        result = await delegate_node(
            sample_state,
            gateway=mock_gateway,
            tools=mock_tools,
            sub_agent_registry=mock_registry,
        )

        assert "tool_results" in result
        assert len(result["tool_results"]) == 1
        tool_result = result["tool_results"][0]
        assert tool_result.tool_name == "sub_agent:failing_agent"
        assert tool_result.success is False
        assert tool_result.error == "Timeout error"

    @pytest.mark.asyncio
    async def test_agent_not_found_or_inactive(self, sample_state: dict[str, Any], mock_gateway: MagicMock, mock_tools: MagicMock, mock_registry: MagicMock) -> None:
        """Handles case where agent spec not found or inactive."""
        mock_registry.get.return_value = None  # Not found

        result = await delegate_node(
            sample_state,
            gateway=mock_gateway,
            tools=mock_tools,
            sub_agent_registry=mock_registry,
        )

        assert result["phase"] == Phase.EXECUTE
        assert len(result["delegation_results"]) == 2
        assert all(r["success"] is False for r in result["delegation_results"])
        assert any("not found or inactive" in str(e) for r in result["delegation_results"] for e in r.get("errors", []))

    @pytest.mark.asyncio
    async def test_agent_inactive_skips_delegation(self, sample_state: dict[str, Any], mock_gateway: MagicMock, mock_tools: MagicMock, mock_registry: MagicMock, sample_spec: SubAgentSpec) -> None:
        """Skips delegation when agent is inactive."""
        sample_spec.is_active = False
        mock_registry.get.return_value = sample_spec

        result = await delegate_node(
            sample_state,
            gateway=mock_gateway,
            tools=mock_tools,
            sub_agent_registry=mock_registry,
        )

        assert result["phase"] == Phase.EXECUTE
        assert len(result["delegation_results"]) == 2
        assert all("not found or inactive" in str(e) or "success" == False for r in result["delegation_results"] for e in r.get("errors", []))

    @pytest.mark.asyncio
    async def test_records_metrics_best_effort(self, sample_state: dict[str, Any], mock_gateway: MagicMock, mock_tools: MagicMock, mock_registry: MagicMock, sample_spec: SubAgentSpec) -> None:
        """Records metrics via SubAgentPersister (best-effort, non-fatal)."""
        sample_state["sub_agents_spawned"] = [
            {"name": "metric_agent", "id": "metric-uuid-123"},
        ]
        mock_registry.get.return_value = sample_spec

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value={
            "success": True,
            "result": "Done",
            "tokens_used": 100,
            "cost_usd": 0.01,
            "latency_ms": 50,
            "iterations": 1,
            "errors": [],
            "goal": "subtask",
            "sub_agent_name": "metric_agent",
            "sub_agent_id": "metric-uuid-123",
        })
        mock_registry.spawn.return_value = mock_runner

        with patch("src.graph.nodes.delegate._record_metrics", new_callable=AsyncMock) as mock_record:
            await delegate_node(
                sample_state,
                gateway=mock_gateway,
                tools=mock_tools,
                sub_agent_registry=mock_registry,
            )

            # Should have called record_metrics
            mock_record.assert_called()

    @pytest.mark.asyncio
    async def test_checks_deprecation_after_execution(self, sample_state: dict[str, Any], mock_gateway: MagicMock, mock_tools: MagicMock, mock_registry: MagicMock, sample_spec: SubAgentSpec) -> None:
        """Checks auto-deprecation after each agent execution."""
        sample_state["sub_agents_spawned"] = [
            {"name": "deprecate_agent", "id": "deprecate-id"},
        ]
        mock_registry.get.return_value = sample_spec

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value={
            "success": True,
            "result": "Done",
            "tokens_used": 100,
            "cost_usd": 0.01,
            "latency_ms": 50,
            "iterations": 1,
            "errors": [],
            "goal": "subtask",
            "sub_agent_name": "deprecate_agent",
            "sub_agent_id": "deprecate-id",
        })
        mock_registry.spawn.return_value = mock_runner

        await delegate_node(
            sample_state,
            gateway=mock_gateway,
            tools=mock_tools,
            sub_agent_registry=mock_registry,
        )

        # Should have called check_deprecation
        mock_registry.check_deprecation.assert_called_with("deprecate_agent")


class TestBuildDelegationGoal:
    """Tests for _build_delegation_goal() helper function."""

    def test_build_delegation_goal_with_main_goal(self, sample_spec: SubAgentSpec) -> None:
        """Builds goal string with main goal context."""
        from src.graph.nodes.delegate import _build_delegation_goal

        state = {
            "current_goal": MagicMock(text="Main goal: optimize database"),
        }

        goal = _build_delegation_goal(sample_spec, state)

        assert "Main goal context:" in goal
        assert "optimize database" in goal
        assert "Your specialization:" in goal
        assert sample_spec.description in goal

    def test_build_delegation_goal_without_main_goal(self, sample_spec: SubAgentSpec) -> None:
        """Handles case where current_goal is missing."""
        from src.graph.nodes.delegate import _build_delegation_goal

        state = {}

        goal = _build_delegation_goal(sample_spec, state)

        assert "Main goal context:" in goal
        assert "Your specialization:" in goal

    def test_build_delegation_goal_truncates_long_goal(self, sample_spec: SubAgentSpec) -> None:
        """Truncates main goal to 300 characters."""
        from src.graph.nodes.delegate import _build_delegation_goal

        long_goal = "Analyze and optimize the performance of the distributed database system " * 10
        state = {
            "current_goal": MagicMock(text=long_goal),
        }

        goal = _build_delegation_goal(sample_spec, state)

        # Should be truncated to ~300 chars
        main_goal_section = goal.split("Your specialization:")[0]
        assert len(main_goal_section) < 500  # Rough check


class TestRecordMetrics:
    """Tests for _record_metrics() helper function."""

    @pytest.mark.asyncio
    async def test_record_metrics_success(self, sample_spec: SubAgentSpec) -> None:
        """Successfully records metrics to DB."""
        from src.graph.nodes.delegate import _record_metrics

        # Valid UUID
        sample_spec.id = "550e8400-e29b-41d4-a716-446655440000"

        result = {
            "success": True,
            "result": "Done",
            "tokens_used": 100,
            "cost_usd": 0.01,
        }
        state = {"thread_id": "thread-001"}

        with patch("src.agents.persister.SubAgentPersister") as mock_persister_class:
            mock_persister = MagicMock()
            mock_persister.record_run_and_update_metrics = AsyncMock()
            mock_persister_class.return_value = mock_persister

            await _record_metrics(sample_spec, result, state)

            mock_persister.record_run_and_update_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_metrics_non_uuid_id(self, sample_spec: SubAgentSpec) -> None:
        """Skips metrics when ID is not a valid UUID."""
        from src.graph.nodes.delegate import _record_metrics

        # Invalid UUID (hex short ID)
        sample_spec.id = "abc123"

        result = {
            "success": True,
            "result": "Done",
        }
        state = {"thread_id": "thread-001"}

        with patch("src.agents.persister.SubAgentPersister") as mock_persister_class:
            mock_persister = MagicMock()
            mock_persister.record_run_and_update_metrics = AsyncMock()
            mock_persister_class.return_value = mock_persister

            await _record_metrics(sample_spec, result, state)

            # Should NOT have called the persister
            mock_persister.record_run_and_update_metrics.assert_not_called()

    @pytest.mark.asyncio
    async def test_record_metrics_failure_non_fatal(self, sample_spec: SubAgentSpec) -> None:
        """Metric recording failure is non-fatal."""
        from src.graph.nodes.delegate import _record_metrics

        sample_spec.id = "550e8400-e29b-41d4-a716-446655440000"

        result = {"success": True, "result": "Done"}
        state = {"thread_id": "thread-001"}

        with patch("src.agents.persister.SubAgentPersister") as mock_persister_class:
            mock_persister = MagicMock()
            mock_persister.record_run_and_update_metrics = AsyncMock(
                side_effect=Exception("DB error")
            )
            mock_persister_class.return_value = mock_persister

            # Should not raise exception
            await _record_metrics(sample_spec, result, state)


class TestDelegateSingleViaNode:
    """Tests for single-agent delegation paths through delegate_node.

    Replaces TestDelegateSingle since _delegate_single was inlined into
    the parallel delegation logic in delegate_node.
    """

    @pytest.mark.asyncio
    async def test_single_agent_success(self, sample_spec: SubAgentSpec, sample_state: dict[str, Any], mock_gateway: MagicMock, mock_tools: MagicMock, mock_registry: MagicMock) -> None:
        """Successfully delegates a single subtask via parallel path."""
        from src.graph.nodes.delegate import delegate_node

        mock_registry.get.return_value = sample_spec
        mock_runner = MagicMock()
        mock_runner.definition = sample_spec
        mock_runner.run = AsyncMock(return_value={
            "success": True,
            "result": "Subtask done",
            "tokens_used": 100,
            "cost_usd": 0.01,
            "latency_ms": 50,
            "iterations": 1,
            "errors": [],
            "goal": "subtask",
            "sub_agent_name": sample_spec.name,
            "sub_agent_id": sample_spec.id,
        })
        mock_registry.spawn.return_value = mock_runner

        state = {
            "sub_agents_spawned": [{"name": sample_spec.name}],
            "thread_id": "test-thread",
            "current_goal": sample_state.get("current_goal"),
        }

        result = await delegate_node(
            state,
            gateway=mock_gateway,
            tools=mock_tools,
            sub_agent_registry=mock_registry,
            memory=None,
        )

        assert result["phase"] == Phase.VERIFY
        assert len(result["delegation_results"]) == 1
        assert result["delegation_results"][0]["success"] is True

    @pytest.mark.asyncio
    async def test_no_tools_returns_error_for_agent(self, sample_spec: SubAgentSpec, sample_state: dict[str, Any], mock_gateway: MagicMock, mock_registry: MagicMock) -> None:
        """Returns error when tools is None during spawn phase."""
        from src.graph.nodes.delegate import delegate_node

        mock_registry.get.return_value = sample_spec

        state = {
            "sub_agents_spawned": [{"name": sample_spec.name}],
            "thread_id": "test-thread",
            "current_goal": sample_state.get("current_goal"),
        }

        result = await delegate_node(
            state,
            gateway=mock_gateway,
            tools=None,
            sub_agent_registry=mock_registry,
            memory=None,
        )

        assert len(result["delegation_results"]) == 1
        assert result["delegation_results"][0]["success"] is False
        assert "No tool registry" in result["delegation_results"][0]["errors"][0]

    @pytest.mark.asyncio
    async def test_spawn_failure_records_error(self, sample_spec: SubAgentSpec, sample_state: dict[str, Any], mock_gateway: MagicMock, mock_tools: MagicMock, mock_registry: MagicMock) -> None:
        """Returns error when spawn fails during spawn phase."""
        from src.graph.nodes.delegate import delegate_node

        mock_registry.get.return_value = sample_spec
        mock_registry.spawn.return_value = None

        state = {
            "sub_agents_spawned": [{"name": sample_spec.name}],
            "thread_id": "test-thread",
            "current_goal": sample_state.get("current_goal"),
        }

        result = await delegate_node(
            state,
            gateway=mock_gateway,
            tools=mock_tools,
            sub_agent_registry=mock_registry,
            memory=None,
        )

        assert len(result["delegation_results"]) == 1
        assert result["delegation_results"][0]["success"] is False
        assert "Failed to spawn" in result["delegation_results"][0]["errors"][0]


class TestTierRouting:
    """Phase 4 F — sub-agents route by spec.model_tier, not a flat SIMPLE set."""

    @staticmethod
    def _success(name: str, sub_id: str) -> dict[str, Any]:
        return {
            "success": True,
            "result": "done",
            "tokens_used": 10,
            "cost_usd": 0.0,
            "latency_ms": 5,
            "iterations": 1,
            "errors": [],
            "goal": "subtask",
            "sub_agent_name": name,
            "sub_agent_id": sub_id,
        }

    @pytest.mark.asyncio
    async def test_critical_spec_routes_at_critical_tier(
        self,
        sample_state: dict[str, Any],
        mock_gateway: MagicMock,
        mock_tools: MagicMock,
        mock_registry: MagicMock,
    ) -> None:
        """A single CRITICAL sub-agent is routed at its declared tier, not SIMPLE."""
        from src.graph.enums import TaskComplexity

        spec = SubAgentSpec(
            name="crit_agent",
            description="critical specialist",
            goal="g",
            parent_thread_id="t",
            tool_scope="inherit_all",
            model_tier=TaskComplexity.CRITICAL,
        )
        mock_registry.get.return_value = spec
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=self._success("crit_agent", "id1"))
        mock_registry.spawn.return_value = mock_runner

        spy = MagicMock(return_value=["glm-4.7"])
        mock_gateway._model_router.route_diverse = spy

        sample_state["sub_agents_spawned"] = [{"name": "crit_agent", "id": "id1"}]
        result = await delegate_node(
            sample_state,
            gateway=mock_gateway,
            tools=mock_tools,
            sub_agent_registry=mock_registry,
        )

        # Routed at the CRITICAL tier (previously always SIMPLE).
        spy.assert_called_once()
        assert spy.call_args.kwargs["complexity"] == TaskComplexity.CRITICAL
        assert spy.call_args.kwargs["n"] == 1
        # The tier-resolved model is pinned on the runner.
        assert mock_runner._model_affinity == "glm-4.7"
        # Delegation still succeeds.
        assert result["phase"] == Phase.VERIFY

    @pytest.mark.asyncio
    async def test_simple_spec_routes_at_simple_tier(
        self,
        sample_state: dict[str, Any],
        mock_gateway: MagicMock,
        mock_tools: MagicMock,
        mock_registry: MagicMock,
        sample_spec: SubAgentSpec,
    ) -> None:
        """A SIMPLE sub-agent routes at the SIMPLE (CHEAP) tier."""
        from src.graph.enums import TaskComplexity

        # sample_spec defaults to model_tier=SIMPLE.
        mock_registry.get.return_value = sample_spec
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=self._success("agent", "id"))
        mock_registry.spawn.return_value = mock_runner

        spy = MagicMock(return_value=["deepseek-v4-flash"])
        mock_gateway._model_router.route_diverse = spy

        sample_state["sub_agents_spawned"] = [{"name": "agent", "id": "id"}]
        await delegate_node(
            sample_state,
            gateway=mock_gateway,
            tools=mock_tools,
            sub_agent_registry=mock_registry,
        )

        assert spy.call_args.kwargs["complexity"] == TaskComplexity.SIMPLE
        assert mock_runner._model_affinity == "deepseek-v4-flash"

    @pytest.mark.asyncio
    async def test_two_critical_siblings_get_distinct_providers(
        self,
        sample_state: dict[str, Any],
        mock_gateway: MagicMock,
        mock_tools: MagicMock,
        mock_registry: MagicMock,
    ) -> None:
        """Two CRITICAL siblings spread across providers (diversity preserved)."""
        from src.graph.enums import TaskComplexity

        spec = SubAgentSpec(
            name="crit",
            description="d",
            goal="g",
            parent_thread_id="t",
            tool_scope="inherit_all",
            model_tier=TaskComplexity.CRITICAL,
        )
        mock_registry.get.return_value = spec
        r1, r2 = MagicMock(), MagicMock()
        r1.run = AsyncMock(return_value=self._success("a", "1"))
        r2.run = AsyncMock(return_value=self._success("b", "2"))
        mock_registry.spawn.side_effect = [r1, r2]

        # route_diverse(n=2, CRITICAL) → two distinct-provider MODERATE models.
        mock_gateway._model_router.route_diverse = MagicMock(
            return_value=["glm-4.7", "deepseek-v4-pro"]
        )

        sample_state["sub_agents_spawned"] = [
            {"name": "a", "id": "1"},
            {"name": "b", "id": "2"},
        ]
        await delegate_node(
            sample_state,
            gateway=mock_gateway,
            tools=mock_tools,
            sub_agent_registry=mock_registry,
        )

        assert r1._model_affinity == "glm-4.7"
        assert r2._model_affinity == "deepseek-v4-pro"
        assert r1._model_affinity != r2._model_affinity

    @pytest.mark.asyncio
    async def test_mixed_tiers_route_independently(
        self,
        sample_state: dict[str, Any],
        mock_gateway: MagicMock,
        mock_tools: MagicMock,
        mock_registry: MagicMock,
    ) -> None:
        """A SIMPLE and a CRITICAL sibling each route at their own tier."""
        from src.graph.enums import TaskComplexity

        simple_spec = SubAgentSpec(
            name="s",
            description="d",
            goal="g",
            parent_thread_id="t",
            tool_scope="inherit_all",
            model_tier=TaskComplexity.SIMPLE,
        )
        crit_spec = SubAgentSpec(
            name="c",
            description="d",
            goal="g",
            parent_thread_id="t",
            tool_scope="inherit_all",
            model_tier=TaskComplexity.CRITICAL,
        )
        mock_registry.get.side_effect = [simple_spec, crit_spec]
        r1, r2 = MagicMock(), MagicMock()
        r1.run = AsyncMock(return_value=self._success("s", "1"))
        r2.run = AsyncMock(return_value=self._success("c", "2"))
        mock_registry.spawn.side_effect = [r1, r2]

        def fake_route_diverse(n: int, complexity: Any, **_: Any) -> list[str]:
            base = "deepseek-v4-flash" if complexity == TaskComplexity.SIMPLE else "glm-4.7"
            return [base] * n

        mock_gateway._model_router.route_diverse = fake_route_diverse

        sample_state["sub_agents_spawned"] = [
            {"name": "s", "id": "1"},
            {"name": "c", "id": "2"},
        ]
        await delegate_node(
            sample_state,
            gateway=mock_gateway,
            tools=mock_tools,
            sub_agent_registry=mock_registry,
        )

        # SIMPLE sibling → CHEAP model; CRITICAL sibling → MODERATE model.
        assert r1._model_affinity == "deepseek-v4-flash"
        assert r2._model_affinity == "glm-4.7"

    @pytest.mark.asyncio
    async def test_routing_failure_is_non_fatal(
        self,
        sample_state: dict[str, Any],
        mock_gateway: MagicMock,
        mock_tools: MagicMock,
        mock_registry: MagicMock,
        sample_spec: SubAgentSpec,
    ) -> None:
        """A route_diverse exception leaves affinity unset but delegation succeeds."""
        mock_registry.get.return_value = sample_spec
        mock_runner = MagicMock()
        mock_runner._model_affinity = ""  # seed the default the helper would skip
        mock_runner.run = AsyncMock(return_value=self._success("agent", "id"))
        mock_registry.spawn.return_value = mock_runner
        mock_gateway._model_router.route_diverse = MagicMock(
            side_effect=RuntimeError("router down")
        )

        sample_state["sub_agents_spawned"] = [{"name": "agent", "id": "id"}]
        result = await delegate_node(
            sample_state,
            gateway=mock_gateway,
            tools=mock_tools,
            sub_agent_registry=mock_registry,
        )

        # Affinity stays at its default; delegation still succeeds.
        assert mock_runner._model_affinity == ""
        assert result["phase"] == Phase.VERIFY

    @pytest.mark.asyncio
    async def test_non_list_route_result_leaves_affinity_unset(
        self,
        sample_state: dict[str, Any],
        mock_gateway: MagicMock,
        mock_tools: MagicMock,
        mock_registry: MagicMock,
        sample_spec: SubAgentSpec,
    ) -> None:
        """A non-list route_diverse result (e.g. a bare mock gateway) leaves affinity unset."""
        mock_registry.get.return_value = sample_spec
        mock_runner = MagicMock()
        mock_runner._model_affinity = ""  # seed the default the helper would skip
        mock_runner.run = AsyncMock(return_value=self._success("agent", "id"))
        mock_registry.spawn.return_value = mock_runner
        mock_gateway._model_router.route_diverse = MagicMock(return_value="not-a-list")

        sample_state["sub_agents_spawned"] = [{"name": "agent", "id": "id"}]
        result = await delegate_node(
            sample_state,
            gateway=mock_gateway,
            tools=mock_tools,
            sub_agent_registry=mock_registry,
        )

        assert mock_runner._model_affinity == ""
        assert result["phase"] == Phase.VERIFY


class TestSubAgentSelection:
    """F1 — delegate prunes the spawned fan-out via select_subagents_for_subtask.

    The selection logic itself is unit-tested in tests/test_agents/test_selection.py;
    here we prove the WIRING: the subset returned by ``select_subagents_for_subtask``
    drives the spawn/execute fan-out (only survivors run), and the default-off path
    fans out to everyone (regression).
    """

    @staticmethod
    def _success(name: str) -> dict[str, Any]:
        return {
            "success": True,
            "result": "done",
            "tokens_used": 10,
            "cost_usd": 0.0,
            "latency_ms": 5,
            "iterations": 1,
            "errors": [],
            "goal": "subtask",
            "sub_agent_name": name,
            "sub_agent_id": "id",
        }

    @pytest.mark.asyncio
    async def test_selection_on_prunes_fanout_to_subset(
        self,
        mock_gateway: MagicMock,
        mock_tools: MagicMock,
        mock_registry: MagicMock,
        sample_spec: SubAgentSpec,
    ) -> None:
        """DoD: when selection prunes 3 spawned to 2 survivors, only 2 run."""
        mock_registry.get.return_value = sample_spec
        runners = []
        for n in ("a", "b"):
            runner = MagicMock()
            runner.run = AsyncMock(return_value=self._success(n))
            runners.append(runner)
        mock_registry.spawn.side_effect = runners

        state = {
            "sub_agents_spawned": [
                {"name": "a", "id": "1"},
                {"name": "b", "id": "2"},
                {"name": "c", "id": "3"},
            ],
            "thread_id": "t",
            "current_goal": MagicMock(text="g"),
            "submitted_goal": "g",
        }

        # Selection prunes to the top-2 survivors (a, b); c is deselected.
        async def fake_select(spawned: Any, subtask: str, settings: Any, persister: Any = None) -> Any:
            del subtask, settings, persister
            return [info for info in spawned if info["name"] in {"a", "b"}]

        with patch(
            "src.graph.nodes.delegate.select_subagents_for_subtask",
            new=fake_select,
        ):
            result = await delegate_node(
                state,
                gateway=mock_gateway,
                tools=mock_tools,
                sub_agent_registry=mock_registry,
            )

        # Only the 2 survivors were spawned/executed; c never ran.
        assert mock_registry.spawn.call_count == 2
        assert len(result["delegation_results"]) == 2
        assert result["phase"] == Phase.VERIFY

    @pytest.mark.asyncio
    async def test_default_off_all_spawn_regression(
        self,
        mock_gateway: MagicMock,
        mock_tools: MagicMock,
        mock_registry: MagicMock,
        sample_spec: SubAgentSpec,
    ) -> None:
        """Regression: default-off fans out to every spawned agent.

        Patches ``get_settings`` to a known default-off ``Settings`` so the REAL
        ``select_subagents_for_subtask`` runs and short-circuits (no embed/DB),
        deterministically — independent of the ambient .env value — proving the
        wiring passes the full spawned set through when selection is off.
        """
        mock_registry.get.return_value = sample_spec
        runners = []
        for n in ("a", "b", "c"):
            runner = MagicMock()
            runner.run = AsyncMock(return_value=self._success(n))
            runners.append(runner)
        mock_registry.spawn.side_effect = runners

        state = {
            "sub_agents_spawned": [
                {"name": "a", "id": "1"},
                {"name": "b", "id": "2"},
                {"name": "c", "id": "3"},
            ],
            "thread_id": "t",
            "current_goal": MagicMock(text="g"),
            "submitted_goal": "g",
        }

        default_off = Settings()
        # Pin OFF explicitly so the patched get_settings() returns a genuinely
        # default-off instance, independent of the ambient .env value of
        # AGENT_SELECTION_ENABLED.
        default_off.agent.agent_selection_enabled = False
        assert default_off.agent.agent_selection_enabled is False
        with patch(
            "src.graph.nodes.delegate.get_settings", return_value=default_off
        ):
            result = await delegate_node(
                state,
                gateway=mock_gateway,
                tools=mock_tools,
                sub_agent_registry=mock_registry,
            )

        # Default-off → no pruning → all 3 spawn/run.
        assert mock_registry.spawn.call_count == 3
        assert len(result["delegation_results"]) == 3
