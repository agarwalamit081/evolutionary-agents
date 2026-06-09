"""Tests for src.graph.factory — initial state creation and validation."""

from __future__ import annotations


from src.graph.enums import (
    Confidence,
    GoalStatus,
    Phase,
    Strategy,
)
from src.graph.factory import initial_state, validate_state


class TestInitialState:
    """Tests for the initial_state factory function."""

    def test_initial_state_creates_valid_state(self) -> None:
        """initial_state returns a dict with all required AgentState fields."""
        state = initial_state(
            goal_text="Test goal",
            thread_id="thread-001",
        )

        # All required fields must be present
        assert "phase" in state
        assert "iteration_count" in state
        assert "max_iterations" in state
        assert "current_goal" in state
        assert "strategy" in state
        assert "plan_steps" in state
        assert "current_step_index" in state
        assert "messages" in state
        assert "tools_called" in state
        assert "tool_results" in state
        assert "completed_steps" in state
        assert "retrieved_memories" in state
        assert "memory_observations" in state
        assert "reflection" in state
        assert "confidence" in state
        assert "evolution_history" in state
        assert "skills_learned" in state
        assert "sub_agents" in state
        assert "total_tokens_used" in state
        assert "cost_records" in state
        assert "budget_remaining" in state
        assert "final_output" in state
        assert "is_complete" in state
        assert "errors" in state
        assert "thread_id" in state
        assert "generation" in state

        # Sensible defaults
        assert state["phase"] == Phase.CLASSIFY
        assert state["iteration_count"] == 0
        assert state["current_step_index"] == 0
        assert state["is_complete"] is False
        assert state["final_output"] == ""
        assert state["confidence"] == Confidence.MEDIUM
        assert state["strategy"] == Strategy.DIRECT

    def test_initial_state_sets_goal_text(self) -> None:
        """initial_state embeds the provided goal_text into current_goal."""
        goal_text = "Build a REST API with authentication"
        state = initial_state(
            goal_text=goal_text,
            thread_id="thread-002",
        )

        goal = state["current_goal"]
        assert goal.text == goal_text
        assert goal.status == GoalStatus.ACTIVE

    def test_initial_state_default_iterations(self) -> None:
        """initial_state defaults max_iterations to 25 when not specified."""
        state = initial_state(
            goal_text="Test goal",
            thread_id="thread-003",
        )
        assert state["max_iterations"] == 25

    def test_initial_state_custom_iterations(self) -> None:
        """initial_state accepts a custom max_iterations value."""
        state = initial_state(
            goal_text="Test goal",
            thread_id="thread-004",
            max_iterations=50,
        )
        assert state["max_iterations"] == 50

    def test_initial_state_thread_id(self) -> None:
        """initial_state stores the provided thread_id."""
        state = initial_state(
            goal_text="Test goal",
            thread_id="custom-thread-id",
        )
        assert state["thread_id"] == "custom-thread-id"

    def test_validate_state_passes_for_valid_state(self) -> None:
        """validate_state returns an empty list for a well-formed state."""
        state = initial_state(
            goal_text="Valid goal",
            thread_id="thread-005",
        )
        violations = validate_state(state)
        assert violations == []

    def test_validate_state_catches_missing_thread_id(self) -> None:
        """validate_state reports a violation when thread_id is empty."""
        state = initial_state(
            goal_text="Test goal",
            thread_id="thread-006",
        )
        state["thread_id"] = ""
        violations = validate_state(state)
        assert any("thread_id" in v for v in violations)

    def test_validate_state_catches_zero_max_iterations(self) -> None:
        """validate_state reports a violation when max_iterations <= 0."""
        state = initial_state(
            goal_text="Test goal",
            thread_id="thread-007",
        )
        state["max_iterations"] = 0
        violations = validate_state(state)
        assert any("max_iterations" in v for v in violations)
