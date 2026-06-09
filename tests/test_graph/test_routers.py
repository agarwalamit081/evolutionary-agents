"""Tests for src.graph.routers — conditional edge routing functions."""

from __future__ import annotations

from typing import Any


from src.graph.enums import Confidence
from src.graph.models import ReflectionResult, ToolResult
from src.graph.routers import (
    route_after_error,
    route_after_evolve,
    route_after_execute,
    route_after_hitl,
    route_after_reflect,
    route_after_store,
    route_after_verify,
)


class TestRouteAfterExecute:
    """Tests for route_after_execute routing function."""

    def test_route_after_execute_to_reflect(self, state_with_plan: dict[str, Any]) -> None:
        """When all plan steps are executed, route to reflect."""
        # Exhaust all steps
        state_with_plan["current_step_index"] = 3  # equal to len(plan_steps)
        result = route_after_execute(state_with_plan)
        assert result == "reflect"

    def test_route_after_execute_to_reflect_on_max_iterations(self, sample_state: dict[str, Any]) -> None:
        """When max iterations is reached, route to reflect."""
        sample_state["iteration_count"] = 25
        sample_state["max_iterations"] = 25
        result = route_after_execute(sample_state)
        assert result == "reflect"

    def test_route_after_execute_to_error(self, sample_state: dict[str, Any]) -> None:
        """When authentication errors are present, route to error_handler."""
        sample_state["errors"] = ["authentication failed for provider"]
        result = route_after_execute(sample_state)
        assert result == "error_handler"

    def test_route_after_execute_to_error_on_non_retriable_tool_error(self, sample_state: dict[str, Any]) -> None:
        """When a non-retriable tool error occurs, route to error_handler."""
        sample_state["tool_results"] = [
            ToolResult(tool_name="code_executor", success=False, output="", error="permission denied"),
        ]
        result = route_after_execute(sample_state)
        assert result == "error_handler"

    def test_route_after_execute_loops_on_retriable_tool_error(self, sample_state: dict[str, Any]) -> None:
        """When a retriable tool error occurs (timeout/rate), loop back to execute."""
        sample_state["tool_results"] = [
            ToolResult(tool_name="web_search", success=False, output="", error="timeout exceeded"),
        ]
        result = route_after_execute(sample_state)
        assert result == "execute"

    def test_route_after_execute_continues_with_remaining_steps(self, state_with_plan: dict[str, Any]) -> None:
        """When there are remaining steps, route back to execute."""
        state_with_plan["current_step_index"] = 0  # 3 steps total, only at index 0
        result = route_after_execute(state_with_plan)
        assert result == "execute"


class TestRouteAfterVerify:
    """Tests for route_after_verify routing function."""

    def test_route_after_verify_to_evolve(self, sample_state: dict[str, Any]) -> None:
        """When is_complete and should_evolve is True, route to evolve."""
        sample_state["is_complete"] = True
        sample_state["reflection"] = ReflectionResult(
            summary="Task complete",
            should_evolve=True,
        )
        result = route_after_verify(sample_state)
        assert result == "evolve"

    def test_route_after_verify_to_store(self, sample_state: dict[str, Any]) -> None:
        """When is_complete and no evolution needed, route to store_memory."""
        sample_state["is_complete"] = True
        sample_state["reflection"] = ReflectionResult(
            summary="Task complete",
            should_evolve=False,
        )
        result = route_after_verify(sample_state)
        assert result == "store_memory"

    def test_route_after_verify_to_store_when_no_reflection(self, sample_state: dict[str, Any]) -> None:
        """When is_complete with no reflection, route to store_memory (no evolve)."""
        sample_state["is_complete"] = True
        sample_state["reflection"] = None
        result = route_after_verify(sample_state)
        assert result == "store_memory"

    def test_route_after_verify_retries_on_low_confidence(self, sample_state: dict[str, Any]) -> None:
        """When not complete with low confidence, route back to execute."""
        sample_state["is_complete"] = False
        sample_state["confidence"] = Confidence.LOW
        result = route_after_verify(sample_state)
        assert result == "execute"

    def test_route_after_verify_retries_on_medium_confidence(self, sample_state: dict[str, Any]) -> None:
        """When not complete with medium confidence, still retries execute."""
        sample_state["is_complete"] = False
        sample_state["confidence"] = Confidence.MEDIUM
        result = route_after_verify(sample_state)
        assert result == "execute"


class TestRouteAfterError:
    """Tests for route_after_error routing function."""

    def test_route_after_error_to_execute(self, sample_state: dict[str, Any]) -> None:
        """When error is retryable (generic), route back to execute."""
        sample_state["errors"] = ["something went wrong"]
        sample_state["iteration_count"] = 5
        sample_state["max_iterations"] = 25
        result = route_after_error(sample_state)
        assert result == "execute"

    def test_route_after_error_to_execute_on_rate_limit(self, sample_state: dict[str, Any]) -> None:
        """When rate limited and under max iterations, retry execute."""
        sample_state["errors"] = ["rate limit exceeded, try again"]
        sample_state["iteration_count"] = 5
        sample_state["max_iterations"] = 25
        result = route_after_error(sample_state)
        assert result == "execute"

    def test_route_after_error_to_classify_on_auth(self, sample_state: dict[str, Any]) -> None:
        """When auth error, route to classify (try different provider)."""
        sample_state["errors"] = ["401 unauthorized access"]
        result = route_after_error(sample_state)
        assert result == "classify"

    def test_route_after_error_to_hitl_on_budget(self, sample_state: dict[str, Any]) -> None:
        """When budget exhausted, escalate to human via hitl_gate."""
        sample_state["errors"] = ["budget limit exceeded"]
        result = route_after_error(sample_state)
        assert result == "hitl_gate"

    def test_route_after_error_to_complete_on_max_iterations(self, sample_state: dict[str, Any]) -> None:
        """When max iterations exceeded with errors, abort to complete."""
        sample_state["errors"] = ["persistent failure"]
        sample_state["iteration_count"] = 25
        sample_state["max_iterations"] = 25
        result = route_after_error(sample_state)
        assert result == "complete"

    def test_route_after_error_to_complete_when_no_errors(self, sample_state: dict[str, Any]) -> None:
        """When no errors present, route to complete."""
        sample_state["errors"] = []
        result = route_after_error(sample_state)
        assert result == "complete"


class TestRouteAfterStore:
    """Tests for route_after_store routing function."""

    def test_route_after_store_to_complete(self, sample_state: dict[str, Any]) -> None:
        """When is_complete is True, route to complete."""
        sample_state["is_complete"] = True
        result = route_after_store(sample_state)
        assert result == "complete"

    def test_route_after_store_to_execute_when_incomplete(self, sample_state: dict[str, Any]) -> None:
        """When is_complete is False, route back to execute."""
        sample_state["is_complete"] = False
        result = route_after_store(sample_state)
        assert result == "execute"


class TestRouteAfterReflect:
    """Tests for route_after_reflect routing function."""

    def test_route_after_reflect_to_verify(self, sample_state: dict[str, Any]) -> None:
        """Medium confidence → route to verify."""
        sample_state["confidence"] = Confidence.MEDIUM
        sample_state["reflection"] = ReflectionResult(summary="ok", should_replan=False)
        result = route_after_reflect(sample_state)
        assert result == "verify"

    def test_route_after_reflect_to_plan(self, sample_state: dict[str, Any]) -> None:
        """should_replan=True → route to plan."""
        sample_state["confidence"] = Confidence.HIGH
        sample_state["reflection"] = ReflectionResult(summary="replan", should_replan=True)
        result = route_after_reflect(sample_state)
        assert result == "plan"

    def test_route_after_reflect_low_confidence_to_execute(self, sample_state: dict[str, Any]) -> None:
        """LOW confidence → route back to execute."""
        sample_state["confidence"] = Confidence.LOW
        sample_state["reflection"] = ReflectionResult(summary="low", should_replan=False)
        result = route_after_reflect(sample_state)
        assert result == "execute"

    def test_route_after_reflect_no_reflection_to_verify(self, sample_state: dict[str, Any]) -> None:
        """No reflection → default to verify."""
        sample_state["reflection"] = None
        result = route_after_reflect(sample_state)
        assert result == "verify"


class TestRouteAfterEvolve:
    """Tests for route_after_evolve routing function."""

    def test_route_after_evolve_to_store_memory(self, sample_state: dict[str, Any]) -> None:
        """No evolution errors → route to store_memory."""
        sample_state["errors"] = []
        result = route_after_evolve(sample_state)
        assert result == "store_memory"

    def test_route_after_evolve_to_error_handler(self, sample_state: dict[str, Any]) -> None:
        """Evolution error → route to error_handler."""
        sample_state["errors"] = ["evolution mutation failed"]
        result = route_after_evolve(sample_state)
        assert result == "error_handler"


class TestRouteAfterHitl:
    """Tests for route_after_hitl routing function."""

    def test_route_after_hitl_approved_to_complete(self, sample_state: dict[str, Any]) -> None:
        """is_complete=True → route to complete."""
        sample_state["is_complete"] = True
        result = route_after_hitl(sample_state)
        assert result == "complete"

    def test_route_after_hitl_rejected_to_execute(self, sample_state: dict[str, Any]) -> None:
        """is_complete=False → route to execute for revision."""
        sample_state["is_complete"] = False
        result = route_after_hitl(sample_state)
        assert result == "execute"


class TestRouteAfterReflectAgentGaps:
    """Tests for agent gap routing in route_after_reflect."""

    def test_route_after_reflect_agent_gaps(self, sample_state: dict[str, Any]) -> None:
        """Routes to agent_spawn when pending_agent_gaps present."""
        sample_state["pending_agent_gaps"] = ["Need data analysis specialist"]
        result = route_after_reflect(sample_state)
        assert result == "agent_spawn"

    def test_route_after_reflect_agent_gaps_before_tool_gaps(self, sample_state: dict[str, Any]) -> None:
        """Agent gaps take priority over tool gaps."""
        sample_state["pending_agent_gaps"] = ["Need specialist"]
        sample_state["pending_tool_gaps"] = ["missing_tool"]
        result = route_after_reflect(sample_state)
        # Agent gaps have higher priority
        assert result == "agent_spawn"

    def test_route_after_reflect_tool_gaps_when_no_agent_gaps(self, sample_state: dict[str, Any]) -> None:
        """Routes to tool_create when only tool gaps present."""
        sample_state["pending_agent_gaps"] = []
        sample_state["pending_tool_gaps"] = ["missing_tool"]
        result = route_after_reflect(sample_state)
        assert result == "tool_create"


class TestRouteAfterAgentSpawn:
    """Tests for route_after_agent_spawn routing function."""

    def test_route_after_agent_spawn_with_spawned(self, sample_state: dict[str, Any]) -> None:
        """Routes to delegate when sub_agents_spawned non-empty."""
        from src.graph.routers import route_after_agent_spawn

        sample_state["sub_agents_spawned"] = [
            {"name": "agent1", "id": "id1"},
            {"name": "agent2", "id": "id2"},
        ]
        result = route_after_agent_spawn(sample_state)
        assert result == "delegate"

    def test_route_after_agent_spawn_empty(self, sample_state: dict[str, Any]) -> None:
        """Routes to plan when no agents spawned."""
        from src.graph.routers import route_after_agent_spawn

        sample_state["sub_agents_spawned"] = []
        result = route_after_agent_spawn(sample_state)
        assert result == "plan"


class TestRouteAfterDelegate:
    """Tests for route_after_delegate routing function."""

    def test_route_after_delegate_all_success(self, sample_state: dict[str, Any]) -> None:
        """Routes to verify when all delegation_results successful."""
        from src.graph.routers import route_after_delegate

        sample_state["delegation_results"] = [
            {"success": True, "result": "Done"},
            {"success": True, "result": "Also done"},
        ]
        result = route_after_delegate(sample_state)
        assert result == "verify"

    def test_route_after_delegate_some_failure(self, sample_state: dict[str, Any]) -> None:
        """Routes to execute when any delegation fails."""
        from src.graph.routers import route_after_delegate

        sample_state["delegation_results"] = [
            {"success": True, "result": "Done"},
            {"success": False, "errors": ["Failed"]},
        ]
        result = route_after_delegate(sample_state)
        assert result == "execute"

    def test_route_after_delegate_empty_results(self, sample_state: dict[str, Any]) -> None:
        """Routes to verify when delegation_results is empty."""
        from src.graph.routers import route_after_delegate

        sample_state["delegation_results"] = []
        result = route_after_delegate(sample_state)
        assert result == "verify"
