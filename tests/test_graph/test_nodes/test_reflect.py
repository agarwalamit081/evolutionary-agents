"""Tests for src.graph.nodes.reflect — reflect node function."""

from __future__ import annotations

import pytest

from src.graph.enums import Confidence, Phase
from src.graph.factory import initial_state
from src.graph.models import PlanStep, ReflectionResult
from src.graph.nodes.reflect import reflect_node


class TestReflectNode:
    """Tests for the reflect_node async function."""

    @pytest.mark.asyncio
    async def test_reflect_medium_completion_medium_confidence(self, state_after_execution: dict) -> None:
        """2/3 steps done (67%), no errors → MEDIUM confidence, VERIFY phase."""
        result = await reflect_node(state_after_execution)

        assert result["phase"] == Phase.VERIFY
        # 2/3 = 67% → below 0.8 threshold for HIGH, so MEDIUM
        assert result["confidence"] == Confidence.MEDIUM
        assert isinstance(result["reflection"], ReflectionResult)

    @pytest.mark.asyncio
    async def test_reflect_with_errors_yields_low_confidence(self, state_after_execution: dict) -> None:
        """Errors present → LOW confidence."""
        state_after_execution["errors"] = ["something failed"]
        result = await reflect_node(state_after_execution)

        assert result["confidence"] == Confidence.LOW

    @pytest.mark.asyncio
    async def test_reflect_low_completion_with_errors_triggers_replan(self, state_with_errors: dict) -> None:
        """0 completed steps + errors → should_replan=True."""
        state_with_errors["plan_steps"] = [
            PlanStep(id="s1", description="Step 1", status="pending"),
            PlanStep(id="s2", description="Step 2", status="pending"),
            PlanStep(id="s3", description="Step 3", status="pending"),
        ]
        state_with_errors["completed_steps"] = []
        result = await reflect_node(state_with_errors)

        reflection = result["reflection"]
        assert reflection.should_replan is True

    @pytest.mark.asyncio
    async def test_reflect_no_steps_done_no_errors(self) -> None:
        """Empty completed, no errors → LOW confidence."""
        state = initial_state("test goal", "thread-nosteps")
        state["plan_steps"] = [
            PlanStep(id="s1", description="Step 1", status="pending"),
        ]
        state["completed_steps"] = []
        result = await reflect_node(state)

        assert result["confidence"] == Confidence.LOW

    @pytest.mark.asyncio
    async def test_reflect_80_percent_done_high_confidence(self) -> None:
        """4/5 steps done → HIGH confidence."""
        state = initial_state("test goal", "thread-80pct")
        state["plan_steps"] = [PlanStep(id=f"s{i}", description=f"Step {i}", status="pending") for i in range(5)]
        state["completed_steps"] = [
            PlanStep(id=f"s{i}", description=f"Step {i}", status="completed", result="done")
            for i in range(4)
        ]
        result = await reflect_node(state)

        assert result["confidence"] == Confidence.HIGH

    @pytest.mark.asyncio
    async def test_reflect_50_percent_done_medium_confidence(self) -> None:
        """1/2 steps done → MEDIUM confidence."""
        state = initial_state("test goal", "thread-50pct")
        state["plan_steps"] = [
            PlanStep(id="s1", description="Step 1", status="pending"),
            PlanStep(id="s2", description="Step 2", status="pending"),
        ]
        state["completed_steps"] = [
            PlanStep(id="s1", description="Step 1", status="completed", result="done"),
        ]
        result = await reflect_node(state)

        assert result["confidence"] == Confidence.MEDIUM

    @pytest.mark.asyncio
    async def test_reflect_returns_reflection_result(self, state_after_execution: dict) -> None:
        """Result contains a ReflectionResult with a non-empty summary."""
        result = await reflect_node(state_after_execution)

        assert "reflection" in result
        reflection = result["reflection"]
        assert isinstance(reflection, ReflectionResult)
        assert len(reflection.summary) > 0

    @pytest.mark.asyncio
    async def test_reflect_with_gateway_falls_back(self, state_after_execution: dict, mock_gateway: object) -> None:
        """Mock gateway returns unparseable JSON → heuristic fallback."""
        result = await reflect_node(state_after_execution, gateway=mock_gateway)

        # Falls back to heuristics since mock returns classify JSON, not reflection JSON
        assert result["phase"] == Phase.VERIFY
        assert isinstance(result["confidence"], Confidence)

    @pytest.mark.asyncio
    async def test_reflect_memory_observations_on_errors(self, state_with_errors: dict) -> None:
        """Errors present → memory_observations list is non-empty."""
        state_with_errors["plan_steps"] = [
            PlanStep(id="s1", description="Step 1", status="pending"),
        ]
        state_with_errors["completed_steps"] = []
        result = await reflect_node(state_with_errors)

        assert len(result["memory_observations"]) > 0
