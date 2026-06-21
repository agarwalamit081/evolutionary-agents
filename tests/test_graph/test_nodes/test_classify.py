"""Tests for src.graph.nodes.classify — classify node function."""

from __future__ import annotations


import pytest

from src.graph.enums import (
    Confidence,
    GoalStatus,
    Phase,
    Strategy,
    TaskComplexity,
)
from src.graph.factory import initial_state
from src.graph.nodes.classify import classify_node


class TestClassifyNode:
    """Tests for the classify_node async function."""

    @pytest.mark.asyncio
    async def test_classify_trivial_task(self) -> None:
        """Tasks with 'explain' keyword are classified as TRIVIAL."""
        state = initial_state(
            goal_text="explain quicksort",
            thread_id="thread-trivial",
        )
        result = await classify_node(state)

        assert result["phase"] == Phase.PLAN
        goal = result["current_goal"]
        assert goal.complexity == TaskComplexity.TRIVIAL

    @pytest.mark.asyncio
    async def test_classify_critical_task(self) -> None:
        """Tasks with 'deploy' keyword are classified as CRITICAL."""
        state = initial_state(
            goal_text="deploy to production",
            thread_id="thread-critical",
        )
        result = await classify_node(state)

        assert result["phase"] == Phase.PLAN
        goal = result["current_goal"]
        assert goal.complexity == TaskComplexity.CRITICAL

    @pytest.mark.asyncio
    async def test_classify_simple_task(self) -> None:
        """Unknown tasks without keyword matches default to SIMPLE."""
        state = initial_state(
            goal_text="write a function to compute fibonacci",
            thread_id="thread-simple",
        )
        result = await classify_node(state)

        assert result["phase"] == Phase.PLAN
        goal = result["current_goal"]
        assert goal.complexity == TaskComplexity.SIMPLE

    @pytest.mark.asyncio
    async def test_classify_no_goal_returns_error(self) -> None:
        """Missing goal text routes to ERROR_HANDLER phase."""
        state = initial_state(
            goal_text="some goal",
            thread_id="thread-nogoal",
        )
        # Simulate a missing goal by clearing it
        state["current_goal"] = None

        result = await classify_node(state)
        assert result["phase"] == Phase.ERROR_HANDLER
        assert len(result["errors"]) > 0

    @pytest.mark.asyncio
    async def test_classify_whitespace_goal_text_is_treated_as_valid(self) -> None:
        """Whitespace-only goal text is truthy, so classify proceeds with heuristics."""
        state = initial_state(
            goal_text="   ",
            thread_id="thread-whitespace",
        )

        result = await classify_node(state)
        # Whitespace is truthy in Python, so classify treats it as a valid goal
        # and proceeds to PLAN with heuristic classification (SIMPLE by default).
        assert result["phase"] == Phase.PLAN
        goal = result["current_goal"]
        assert goal.complexity == TaskComplexity.SIMPLE

    @pytest.mark.asyncio
    async def test_classify_empty_string_goal_text_returns_error(self) -> None:
        """Empty string goal text (falsy) routes to ERROR_HANDLER phase."""
        state = initial_state(
            goal_text="some goal",
            thread_id="thread-empty",
        )
        from src.graph.models import Goal
        state["current_goal"] = Goal(text="")

        result = await classify_node(state)
        assert result["phase"] == Phase.ERROR_HANDLER

    @pytest.mark.asyncio
    async def test_classify_sets_goal_status_active(self) -> None:
        """classify_node sets the goal status to ACTIVE."""
        state = initial_state(
            goal_text="explain quicksort",
            thread_id="thread-status",
        )
        result = await classify_node(state)
        assert result["current_goal"].status == GoalStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_classify_returns_strategy(self) -> None:
        """classify_node returns a strategy field."""
        state = initial_state(
            goal_text="search for relevant papers",
            thread_id="thread-strategy",
        )
        result = await classify_node(state)
        assert "strategy" in result
        assert isinstance(result["strategy"], Strategy)

    @pytest.mark.asyncio
    async def test_classify_returns_confidence(self) -> None:
        """classify_node returns a confidence level."""
        state = initial_state(
            goal_text="explain quicksort",
            thread_id="thread-conf",
        )
        result = await classify_node(state)
        assert "confidence" in result
        assert isinstance(result["confidence"], Confidence)
