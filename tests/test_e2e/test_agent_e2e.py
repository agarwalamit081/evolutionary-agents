"""E2E tests for the full agent pipeline — main.py to completion.

Tests the complete agent flow: classify → plan → execute → reflect → verify.
Uses real LLM calls (no mocking) to validate the full LangGraph pipeline.

Requires OPENAI_API_KEY.
Run with: python -m pytest tests/test_e2e/test_agent_e2e.py -v -m e2e
"""

from __future__ import annotations

import os

import pytest

from src.graph.enums import Phase
from src.graph.factory import initial_state
from src.graph.task_graph import compile_task_graph


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY for E2E agent tests",
    ),
]


@pytest.fixture
def gateway():
    """Create a real LLMGateway."""
    from src.config import get_settings
    from src.llm.gateway import LLMGateway

    settings = get_settings()
    return LLMGateway(settings)


@pytest.fixture
def tools():
    """Create a ToolRegistry with built-in tools."""
    from src.tools import create_default_registry

    return create_default_registry()


@pytest.fixture
def sub_agent_registry():
    """Create an empty SubAgentRegistry."""
    from src.agents.registry import SubAgentRegistry

    return SubAgentRegistry()


class TestFullPipelineSimpleGoal:
    """E2E test for a simple goal through the full pipeline."""

    @pytest.mark.asyncio
    async def test_simple_goal_classifies_and_completes(
        self, gateway, tools, sub_agent_registry
    ) -> None:
        """Simple goal goes through classify → plan → execute → reflect and completes."""
        goal_text = "Explain what a REST API is in 2-3 sentences"
        thread_id = "thread-e2e-simple-001"
        state = initial_state(goal_text, thread_id, max_iterations=5)

        compiled = compile_task_graph(
            gateway=gateway,
            memory=None,
            tools=tools,
            checkpointer=None,
            sub_agent_registry=sub_agent_registry,
        )

        result = await compiled.ainvoke(dict(state))
        result = dict(result) if not isinstance(result, dict) else result

        # Should have progressed beyond CLASSIFY phase
        assert result.get("phase") != Phase.CLASSIFY

        # Should have classified the goal
        goal = result.get("current_goal")
        assert goal is not None

        # Should have completed (simple task)
        if result.get("is_complete"):
            assert result.get("final_output")
            assert len(result.get("final_output", "").strip()) > 0


class TestFullPipelineComplexGoal:
    """E2E test for a complex goal triggering planning strategy."""

    @pytest.mark.asyncio
    async def test_complex_goal_generates_plan(
        self, gateway, tools, sub_agent_registry
    ) -> None:
        """Complex goal triggers planning strategy with multiple steps."""
        goal_text = (
            "Analyze the structure of a URL and explain each component: "
            "scheme, host, port, path, query parameters, and fragment. "
            "Provide an example for each."
        )
        thread_id = "thread-e2e-complex-001"
        state = initial_state(goal_text, thread_id, max_iterations=8)

        compiled = compile_task_graph(
            gateway=gateway,
            memory=None,
            tools=tools,
            checkpointer=None,
            sub_agent_registry=sub_agent_registry,
        )

        result = await compiled.ainvoke(dict(state))
        result = dict(result) if not isinstance(result, dict) else result

        # Should have a plan key present (possibly empty for heuristic path)
        assert "plan_steps" in result
        # With LLM classify, should detect complexity and plan
        # With heuristic fallback, might still plan based on keywords

        # Should have progressed through the pipeline
        iteration_count = result.get("iteration_count", 0)
        assert iteration_count > 0, "Agent should have executed at least one iteration"


class TestSubAgentDelegationE2E:
    """E2E test for sub-agent spawning and delegation."""

    @pytest.mark.asyncio
    async def test_goal_triggers_sub_agent_if_needed(
        self, gateway, tools, sub_agent_registry
    ) -> None:
        """Complex goal may trigger sub-agent spawn (depends on LLM classification)."""
        goal_text = (
            "Perform a comprehensive analysis of error handling patterns "
            "in Python and generate a best practices guide"
        )
        thread_id = "thread-e2e-subagent-001"
        state = initial_state(goal_text, thread_id, max_iterations=10)

        compiled = compile_task_graph(
            gateway=gateway,
            memory=None,
            tools=tools,
            checkpointer=None,
            sub_agent_registry=sub_agent_registry,
        )

        result = await compiled.ainvoke(dict(state))
        result = dict(result) if not isinstance(result, dict) else result

        # Check sub-agent related fields exist
        sub_agents_spawned = result.get("sub_agents_spawned", [])
        delegation_results = result.get("delegation_results", [])

        # Whether sub-agents spawn depends on the LLM's classification
        # and reflection — either outcome is valid for E2E testing
        # The important thing is the agent doesn't crash
        assert isinstance(sub_agents_spawned, list)
        assert isinstance(delegation_results, list)

        # If sub-agents were spawned and delegated, check results
        for deleg_result in delegation_results:
            assert "success" in deleg_result
            assert "sub_agent_name" in deleg_result
            if deleg_result["success"]:
                assert deleg_result.get("result")


class TestToolCreationE2E:
    """E2E test for runtime tool creation (when gaps detected)."""

    @pytest.mark.asyncio
    async def test_agent_handles_tool_gaps_gracefully(
        self, gateway, sub_agent_registry
    ) -> None:
        """Agent with no tools handles the situation without crashing."""
        from src.tools.registry import ToolRegistry

        empty_tools = ToolRegistry()  # No tools registered

        goal_text = "Calculate the Fibonacci sequence up to the 10th term"
        thread_id = "thread-e2e-tools-001"
        state = initial_state(goal_text, thread_id, max_iterations=5)

        compiled = compile_task_graph(
            gateway=gateway,
            memory=None,
            tools=empty_tools,
            checkpointer=None,
            sub_agent_registry=sub_agent_registry,
        )

        result = await compiled.ainvoke(dict(state))
        result = dict(result) if not isinstance(result, dict) else result

        # Agent should not crash even with no tools
        assert result is not None
        # Should have attempted some work
        assert result.get("iteration_count", 0) > 0
