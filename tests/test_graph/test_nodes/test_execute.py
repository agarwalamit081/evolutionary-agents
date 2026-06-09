"""Tests for src.graph.nodes.execute — execute node function."""

from __future__ import annotations

from typing import Any

import pytest

from src.graph.enums import GoalStatus, Phase
from src.graph.models import PlanStep
from src.graph.nodes.execute import execute_node


class TestExecuteNode:
    """Tests for the execute_node async function."""

    @pytest.mark.asyncio
    async def test_execute_marks_step_complete(self, state_with_plan: dict[str, Any]) -> None:
        """Execute node marks the current step as COMPLETED."""
        result = await execute_node(state_with_plan)

        # The completed_steps list should contain the finished step
        completed = result["completed_steps"]
        assert len(completed) == 1
        assert completed[0].status == GoalStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_advances_step_index(self, state_with_plan: dict[str, Any]) -> None:
        """Execute node increments current_step_index by 1."""
        initial_index = state_with_plan["current_step_index"]
        assert initial_index == 0

        result = await execute_node(state_with_plan)
        assert result["current_step_index"] == initial_index + 1

    @pytest.mark.asyncio
    async def test_execute_no_plan_returns_reflect(self, sample_state: dict[str, Any]) -> None:
        """When plan_steps is empty, execute routes to REFLECT phase."""
        sample_state["plan_steps"] = []
        sample_state["current_step_index"] = 0

        result = await execute_node(sample_state)
        assert result["phase"] == Phase.REFLECT

    @pytest.mark.asyncio
    async def test_execute_index_out_of_range_returns_reflect(self, sample_state: dict[str, Any]) -> None:
        """When step_index >= len(plan_steps), route to REFLECT."""
        sample_state["plan_steps"] = [
            PlanStep(id="s1", description="Step 1"),
        ]
        sample_state["current_step_index"] = 5  # out of range

        result = await execute_node(sample_state)
        assert result["phase"] == Phase.REFLECT

    @pytest.mark.asyncio
    async def test_execute_increments_iteration_count(self, state_with_plan: dict[str, Any]) -> None:
        """Execute node increments iteration_count."""
        initial_count = state_with_plan["iteration_count"]
        result = await execute_node(state_with_plan)
        assert result["iteration_count"] == initial_count + 1

    @pytest.mark.asyncio
    async def test_execute_adds_message(self, state_with_plan: dict[str, Any]) -> None:
        """Execute node adds a user message describing the execution."""
        result = await execute_node(state_with_plan)
        messages = result.get("messages", [])
        assert len(messages) >= 1
        # The message should reference the step description
        step_desc = state_with_plan["plan_steps"][0].description
        assert any(step_desc in str(m) for m in messages)

    @pytest.mark.asyncio
    async def test_execute_sequential_steps(self, state_with_plan: dict[str, Any]) -> None:
        """Execute processes steps sequentially, advancing index each call."""
        assert state_with_plan["current_step_index"] == 0

        result1 = await execute_node(state_with_plan)
        assert result1["current_step_index"] == 1

        # Update state for next step
        state_with_plan["current_step_index"] = result1["current_step_index"]
        result2 = await execute_node(state_with_plan)
        assert result2["current_step_index"] == 2

    @pytest.mark.asyncio
    async def test_execute_step_has_result(self, state_with_plan: dict[str, Any]) -> None:
        """After execution, the completed step has a result string."""
        result = await execute_node(state_with_plan)
        completed = result["completed_steps"][0]
        assert completed.result is not None
        assert len(completed.result) > 0
