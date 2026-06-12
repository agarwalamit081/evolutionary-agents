"""Tests for gap deduplication across reflect cycles."""

from __future__ import annotations

from typing import Any

from src.graph.enums import Confidence
from src.graph.models import Goal, GoalStatus, PlanStep, ReflectionResult
from src.graph.nodes.reflect import _detect_agent_gaps_heuristic, _heuristic_reflect


def _make_state(
    pending_tool_gaps: list[str] | None = None,
    pending_agent_gaps: list[str] | None = None,
    sub_agents_spawned: list[dict[str, Any]] | None = None,
    tool_results: list[Any] | None = None,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    """Create a minimal state dict for testing."""
    return {
        "current_goal": Goal(text="test goal", status=GoalStatus.ACTIVE),
        "plan_steps": [PlanStep(description=f"step {i}") for i in range(3)],
        "completed_steps": [PlanStep(description="step 0")],
        "errors": errors or [],
        "iteration_count": 1,
        "max_iterations": 25,
        "pending_tool_gaps": pending_tool_gaps or [],
        "pending_agent_gaps": pending_agent_gaps or [],
        "sub_agents_spawned": sub_agents_spawned or [],
        "tool_results": tool_results or [],
    }


class MockToolResult:
    """Mock ToolResult for testing."""

    def __init__(self, name: str, error: str = "") -> None:
        self.tool_name = name
        self.success = False
        self.output = ""
        self.error = error


class TestToolGapDedup:
    """Verify pending_tool_gaps deduplicates against existing state."""

    def test_no_duplicate_tool_gaps(self) -> None:
        """If gap already exists in state, heuristic should not add it again."""
        state = _make_state(
            pending_tool_gaps=["tool matching 'web_scraper' capability"],
            tool_results=[MockToolResult("web_scraper", "Unknown tool: web_scraper")],
        )
        result = _heuristic_reflect(
            state,  # type: ignore[arg-type]
            "test goal",
            state["completed_steps"],  # type: ignore[arg-type]
            [],
            None,
        )
        tool_gaps = result.get("pending_tool_gaps", [])
        # Same gap should not appear again
        assert tool_gaps == []

    def test_new_tool_gap_is_returned(self) -> None:
        """A genuinely new gap should be returned."""
        from src.tools.registry import ToolRegistry

        registry = ToolRegistry()  # Empty — json_parser is truly missing

        state = _make_state(
            pending_tool_gaps=["tool matching 'web_scraper' capability"],
            tool_results=[MockToolResult("json_parser", "Unknown tool: json_parser")],
        )
        result = _heuristic_reflect(
            state,  # type: ignore[arg-type]
            "test goal",
            state["completed_steps"],  # type: ignore[arg-type]
            [],
            registry,
        )
        tool_gaps = result.get("pending_tool_gaps", [])
        assert len(tool_gaps) == 1
        assert "json_parser" in tool_gaps[0]


class TestAgentGapDedup:
    """Verify pending_agent_gaps deduplicates against existing state."""

    def test_no_duplicate_agent_gaps_when_already_spawned(self) -> None:
        """If sub-agents already spawned, heuristic should not add more gaps."""
        state = _make_state(
            sub_agents_spawned=[{"name": "data_analyst"}],
            pending_agent_gaps=["multi-part task with 7 steps"],
        )
        gaps = _detect_agent_gaps_heuristic(
            state,  # type: ignore[arg-type]
            "test goal",
            state["plan_steps"],  # type: ignore[arg-type]
        )
        assert gaps == []

    def test_no_duplicate_agent_gaps_in_state(self) -> None:
        """If agent gap already in state, heuristic should not re-add it."""
        state = _make_state(
            pending_agent_gaps=[
                "multi-part task with 7 steps — specialized sub-agents"
            ],
        )
        result = _heuristic_reflect(
            state,  # type: ignore[arg-type]
            "analyze data and write report and visualize",
            state["completed_steps"],  # type: ignore[arg-type]
            [],
            None,
        )
        agent_gaps = result.get("pending_agent_gaps", [])
        # The existing gap should not be duplicated
        existing = state.get("pending_agent_gaps", [])
        for gap in agent_gaps:
            assert gap not in existing


class TestRouterGuard:
    """Verify route_after_reflect skips agent_spawn when already spawned."""

    def test_skips_agent_spawn_when_spawned(self) -> None:
        """If sub_agents_spawned is set, should not route to agent_spawn."""
        from src.graph.routers import route_after_reflect

        state = {
            "pending_agent_gaps": ["some gap"],
            "pending_tool_gaps": [],
            "sub_agents_spawned": [{"name": "analyst"}],
            "reflection": ReflectionResult(
                summary="test",
                lessons_learned=[],
                confidence=Confidence.HIGH,
                should_evolve=False,
                should_replan=False,
                memory_observations=[],
                cost_efficiency=1.0,
            ),
            "confidence": Confidence.HIGH,
        }
        result = route_after_reflect(state)  # type: ignore[arg-type]
        assert result != "agent_spawn"

    def test_routes_to_agent_spawn_when_not_spawned(self) -> None:
        """If gaps exist and no agents spawned, should route to agent_spawn."""
        from src.graph.routers import route_after_reflect

        state = {
            "pending_agent_gaps": ["some gap"],
            "pending_tool_gaps": [],
            "sub_agents_spawned": [],
            "reflection": None,
            "confidence": Confidence.MEDIUM,
        }
        result = route_after_reflect(state)  # type: ignore[arg-type]
        assert result == "agent_spawn"
