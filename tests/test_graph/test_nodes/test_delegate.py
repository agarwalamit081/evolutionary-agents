"""Tests for delegate_node from src.graph.nodes.delegate."""

from __future__ import annotations

from typing import Any

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

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

        call_count = 0

        async def mock_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
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
                }
            else:
                return {
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
                }

        mock_runner.run = mock_run
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
            result = await delegate_node(
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

        result = await delegate_node(
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
