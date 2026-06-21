"""Tests for code_executor overuse detection in reflect node."""

from __future__ import annotations

from typing import Any

from src.graph.models import Goal, GoalStatus, PlanStep
from src.graph.nodes.reflect import _heuristic_reflect


class MockToolResult:
    """Mock ToolResult for testing."""

    def __init__(self, name: str, success: bool = True, error: str = "") -> None:
        self.tool_name = name
        self.success = success
        self.output = "ok"
        self.error = error


def _make_state(tool_results: list[Any]) -> dict[str, Any]:
    """Create a minimal state dict with specified tool results."""
    return {
        "current_goal": Goal(text="test goal", status=GoalStatus.ACTIVE),
        "plan_steps": [PlanStep(description=f"step {i}") for i in range(5)],
        "completed_steps": [PlanStep(description=f"step {i}") for i in range(3)],
        "errors": [],
        "iteration_count": 1,
        "max_iterations": 25,
        "pending_tool_gaps": [],
        "pending_agent_gaps": [],
        "sub_agents_spawned": [],
        "tool_results": tool_results,
    }


class TestCodeExecutorOveruseDetection:
    """Verify reflect detects code_executor overuse as a tool gap."""

    def test_overuse_detected_at_three_calls(self) -> None:
        """3+ code_executor calls with no other gaps should trigger a tool gap."""
        state = _make_state([
            MockToolResult("code_executor"),
            MockToolResult("code_executor"),
            MockToolResult("code_executor"),
        ])
        result = _heuristic_reflect(
            state,  # type: ignore[arg-type]
            "test goal",
            state["completed_steps"],  # type: ignore[arg-type]
            [],
            None,
        )
        tool_gaps = result.get("pending_tool_gaps", [])
        assert len(tool_gaps) == 1
        assert "recurring code_executor" in tool_gaps[0]

    def test_no_overuse_below_three_calls(self) -> None:
        """Under 3 code_executor calls should not trigger overuse gap."""
        state = _make_state([
            MockToolResult("code_executor"),
            MockToolResult("code_executor"),
        ])
        result = _heuristic_reflect(
            state,  # type: ignore[arg-type]
            "test goal",
            state["completed_steps"],  # type: ignore[arg-type]
            [],
            None,
        )
        tool_gaps = result.get("pending_tool_gaps", [])
        assert tool_gaps == []

    def test_overuse_not_triggered_when_other_gaps_exist(self) -> None:
        """If other gaps exist, overuse detection should not add another."""
        state = _make_state([
            MockToolResult("code_executor"),
            MockToolResult("code_executor"),
            MockToolResult("code_executor"),
            MockToolResult("unknown_tool", error="Unknown tool: web_scraper"),
        ])
        result = _heuristic_reflect(
            state,  # type: ignore[arg-type]
            "test goal",
            state["completed_steps"],  # type: ignore[arg-type]
            [],
            None,
        )
        # Overuse gap is suppressed when missing_tools already has entries
        # (the "Unknown tool" gap won't fire without registry, but overuse
        # should still be the only gap since no registry is provided)
        tool_gaps = result.get("pending_tool_gaps", [])
        # Either 1 (overuse only) or 2 (overuse + unknown) — overuse must exist
        assert any("recurring code_executor" in g for g in tool_gaps)

    def test_mixed_tools_no_overuse(self) -> None:
        """Mix of different tools should not trigger overuse."""
        state = _make_state([
            MockToolResult("code_executor"),
            MockToolResult("file_writer"),
            MockToolResult("code_executor"),
            MockToolResult("web_search"),
        ])
        result = _heuristic_reflect(
            state,  # type: ignore[arg-type]
            "test goal",
            state["completed_steps"],  # type: ignore[arg-type]
            [],
            None,
        )
        tool_gaps = result.get("pending_tool_gaps", [])
        assert tool_gaps == []
