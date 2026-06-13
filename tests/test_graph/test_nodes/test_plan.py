"""Tests for src.graph.nodes.plan — plan node function."""

from __future__ import annotations

import pytest

from src.graph.enums import GoalStatus, Phase, Strategy
from src.graph.factory import initial_state
from src.graph.models import Goal
from src.graph.nodes.plan import plan_node


class TestPlanNode:
    """Tests for the plan_node async function."""

    @pytest.mark.asyncio
    async def test_plan_react_strategy(self) -> None:
        """REACT strategy generates a 3-step plan."""
        state = initial_state("implement a REST API", "thread-react")
        state["strategy"] = Strategy.REACT
        result = await plan_node(state)

        assert result["phase"] == Phase.RETRIEVE_MEMORY
        steps = result["plan_steps"]
        assert len(steps) == 3
        assert all(s.status == GoalStatus.PENDING for s in steps)
        assert result["current_step_index"] == 0

    @pytest.mark.asyncio
    async def test_plan_direct_strategy(self) -> None:
        """DIRECT strategy generates a single-step plan."""
        state = initial_state("define variable", "thread-direct")
        state["strategy"] = Strategy.DIRECT
        result = await plan_node(state)

        assert result["phase"] == Phase.RETRIEVE_MEMORY
        assert len(result["plan_steps"]) == 1

    @pytest.mark.asyncio
    async def test_plan_planning_strategy(self) -> None:
        """PLANNING strategy generates a 4-step plan."""
        state = initial_state("build end-to-end pipeline", "thread-planning")
        state["strategy"] = Strategy.PLANNING
        result = await plan_node(state)

        assert result["phase"] == Phase.RETRIEVE_MEMORY
        assert len(result["plan_steps"]) == 4

    @pytest.mark.asyncio
    async def test_plan_reflection_strategy(self) -> None:
        """REFLECTION strategy generates a 4-step plan."""
        state = initial_state("review and improve code", "thread-reflection")
        state["strategy"] = Strategy.REFLECTION
        result = await plan_node(state)

        assert result["phase"] == Phase.RETRIEVE_MEMORY
        assert len(result["plan_steps"]) == 4

    @pytest.mark.asyncio
    async def test_plan_tot_strategy(self) -> None:
        """TOT strategy generates a 3-step plan."""
        state = initial_state("compare multiple approaches", "thread-tot")
        state["strategy"] = Strategy.TOT
        result = await plan_node(state)

        assert result["phase"] == Phase.RETRIEVE_MEMORY
        assert len(result["plan_steps"]) == 3

    @pytest.mark.asyncio
    async def test_plan_debate_strategy(self) -> None:
        """DEBATE strategy generates a 3-step plan."""
        state = initial_state("argue pros and cons", "thread-debate")
        state["strategy"] = Strategy.DEBATE
        result = await plan_node(state)

        assert result["phase"] == Phase.RETRIEVE_MEMORY
        assert len(result["plan_steps"]) == 3

    @pytest.mark.asyncio
    async def test_plan_rewoo_strategy_falls_back(self) -> None:
        """REWOO strategy has no dedicated branch, falls back to single-step plan."""
        state = initial_state("some goal", "thread-rewoo")
        state["strategy"] = Strategy.REWOO
        result = await plan_node(state)

        assert result["phase"] == Phase.RETRIEVE_MEMORY
        assert len(result["plan_steps"]) == 1

    @pytest.mark.asyncio
    async def test_plan_no_goal_returns_error(self) -> None:
        """Missing goal routes to ERROR_HANDLER phase."""
        state = initial_state("some goal", "thread-nogoal")
        state["current_goal"] = None
        result = await plan_node(state)

        assert result["phase"] == Phase.ERROR_HANDLER
        assert len(result["errors"]) > 0

    @pytest.mark.asyncio
    async def test_plan_empty_goal_text_returns_error(self) -> None:
        """Empty goal text routes to ERROR_HANDLER phase."""
        state = initial_state("some goal", "thread-empty")
        state["current_goal"] = Goal(text="")
        result = await plan_node(state)

        assert result["phase"] == Phase.ERROR_HANDLER
        assert len(result["errors"]) > 0

    @pytest.mark.asyncio
    async def test_plan_with_gateway_falls_back_to_heuristic(self, mock_gateway: object) -> None:
        """When gateway returns unparseable JSON, falls back to heuristics."""
        state = initial_state("implement feature", "thread-gw")
        state["strategy"] = Strategy.REACT
        result = await plan_node(state, gateway=mock_gateway)

        # Mock gateway returns classify JSON, not plan JSON → falls back to heuristic
        assert result["phase"] == Phase.RETRIEVE_MEMORY
        assert len(result["plan_steps"]) == 3

    @pytest.mark.asyncio
    async def test_plan_caps_heuristic_plan_to_remaining_iterations(self) -> None:
        """Plan is capped to the remaining iteration budget (PLANNING=4 steps, budget=3)."""
        state = initial_state("build end-to-end pipeline", "thread-cap", max_iterations=3)
        state["strategy"] = Strategy.PLANNING
        result = await plan_node(state)

        # remaining = max(0, 3 - 0) = 3 → cap = min(planning_max_steps=10, 3) = 3
        assert result["phase"] == Phase.RETRIEVE_MEMORY
        assert len(result["plan_steps"]) == 3

    @pytest.mark.asyncio
    async def test_plan_passes_remaining_iterations_to_prompt(self, mock_gateway: object) -> None:
        """The LLM plan prompt receives the remaining/max iteration budget."""
        state = initial_state("implement feature", "thread-budget", max_iterations=10)
        state["strategy"] = Strategy.REACT
        await plan_node(state, gateway=mock_gateway)

        # The mock gateway returns classify JSON → acompletion is still called once
        # before the parse fallback; inspect the user message it received.
        messages = mock_gateway.acompletion.call_args.kwargs["messages"]
        user_content = messages[1]["content"]
        assert "Remaining iterations" in user_content
        assert "10 / 10" in user_content  # 10 remaining of 10 at iteration 0
