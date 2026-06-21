"""E2E tests for sub-agent subgraph execution.

Tests the full sub-agent pipeline: classify → plan → execute → reflect.
Also regression tests for the goal-passing bug fix.

Requires OPENAI_API_KEY.
Run with: python -m pytest tests/test_e2e/test_subgraph_e2e.py -v -m e2e
"""

from __future__ import annotations

import os

import pytest

from src.agents.runner import SubAgentRunner, _extract_results
from src.agents.state import initial_sub_agent_state
from src.graph.models import Goal, GoalStatus, SubAgentSpec
from src.tools.registry import ToolRegistry


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY for E2E subgraph tests",
    ),
]


@pytest.fixture
def gateway():
    """Create a real LLMGateway for E2E tests."""
    from src.config import get_settings
    from src.llm.gateway import LLMGateway

    settings = get_settings()
    return LLMGateway(settings)


@pytest.fixture
def tools() -> ToolRegistry:
    """Create a ToolRegistry with built-in tools."""
    from src.tools import create_default_registry

    return create_default_registry()


@pytest.fixture
def simple_spec() -> SubAgentSpec:
    """SubAgentSpec for simple sub-agent E2E test."""
    return SubAgentSpec(
        name="e2e_simple_agent",
        description="E2E test agent for simple tasks",
        goal="List the main components of a REST API",
        parent_thread_id="thread-e2e-simple-001",
        tool_scope="inherit_all",
        max_iterations=3,
    )


class TestSubgraphGoalPassing:
    """Regression tests for the sub-agent goal-passing bug fix."""

    def test_initial_state_has_current_goal(self) -> None:
        """initial_sub_agent_state populates current_goal correctly."""
        state = initial_sub_agent_state(
            goal_text="Analyze security vulnerabilities in the codebase",
            parent_thread_id="thread-e2e-goal-001",
        )

        goal = state.get("current_goal")
        assert goal is not None
        assert isinstance(goal, Goal)
        assert goal.text == "Analyze security vulnerabilities in the codebase"
        assert goal.status == GoalStatus.ACTIVE

    def test_goal_text_matches_current_goal(self) -> None:
        """goal_text and current_goal.text are consistent."""
        state = initial_sub_agent_state(
            goal_text="Optimize database query performance",
            parent_thread_id="thread-e2e-goal-002",
        )

        goal = state.get("current_goal")
        assert goal is not None
        assert state.get("goal_text") == goal.text


class TestSubgraphExecution:
    """Tests for full sub-agent subgraph execution with real LLM."""

    @pytest.mark.asyncio
    async def test_subgraph_runs_to_completion(
        self, gateway, tools: ToolRegistry, simple_spec: SubAgentSpec
    ) -> None:
        """Sub-agent subgraph runs classify→plan→execute→reflect successfully."""
        runner = SubAgentRunner(
            definition=simple_spec,
            gateway=gateway,
            tools=tools,
        )

        result = await runner.run(
            goal="List the main components of a REST API",
            parent_thread_id="thread-e2e-exec-001",
        )

        # Result should have standard structure
        assert "success" in result
        assert "errors" in result
        assert "result" in result
        assert "sub_agent_name" in result
        assert result["sub_agent_name"] == "e2e_simple_agent"

        # With a simple task, sub-agent should produce output
        if result["success"]:
            assert result["result"]
            assert len(result["result"].strip()) > 0


class TestDelegationResultSuccess:
    """Tests for delegation result success determination."""

    def test_delegation_success_with_transient_errors(self) -> None:
        """Completed sub-agent with output is marked successful despite errors."""
        result_state = {
            "final_output": "Analysis complete: 3 security issues found",
            "is_complete": True,
            "errors": [
                "Provider deepseek-v4-flash: 401 Unauthorized",
            ],
            "cost_records": [],
            "total_tokens_used": 500,
            "iteration_count": 3,
        }

        spec = SubAgentSpec(
            name="security_scanner",
            description="Security analysis",
            goal="Scan for vulnerabilities",
            parent_thread_id="thread-e2e-delegate-001",
        )

        result = _extract_results(
            result_state=result_state,
            latency_ms=2000,
            goal="Scan for vulnerabilities",
            spec=spec,
        )

        assert result["success"] is True
        assert "3 security issues found" in result["result"]
        assert len(result["errors"]) == 1  # Error still reported

    def test_delegation_failure_when_incomplete(self) -> None:
        """Incomplete sub-agent is marked as failed."""
        result_state = {
            "final_output": "",
            "is_complete": False,
            "errors": ["Execution exceeded max iterations"],
            "cost_records": [],
            "total_tokens_used": 1000,
            "iteration_count": 10,
        }

        spec = SubAgentSpec(
            name="stuck_agent",
            description="Agent that gets stuck",
            goal="Impossible task",
            parent_thread_id="thread-e2e-delegate-002",
        )

        result = _extract_results(
            result_state=result_state,
            latency_ms=10000,
            goal="Impossible task",
            spec=spec,
        )

        assert result["success"] is False
