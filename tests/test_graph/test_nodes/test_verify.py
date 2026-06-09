"""Tests for src.graph.nodes.verify — verify node function."""

from __future__ import annotations

import pytest

from src.graph.enums import Confidence, Phase
from src.graph.factory import initial_state
from src.graph.models import PlanStep, ReflectionResult
from src.graph.nodes.verify import verify_node


class TestVerifyNode:
    """Tests for the verify_node async function."""

    @pytest.mark.asyncio
    async def test_verify_all_done_no_errors_high_conf(self) -> None:
        """All steps done, no errors, HIGH confidence → complete."""
        state = initial_state("test goal", "thread-done")
        state["plan_steps"] = [
            PlanStep(id="s1", description="Step 1", status="completed", result="done"),
            PlanStep(id="s2", description="Step 2", status="completed", result="done"),
        ]
        state["completed_steps"] = list(state["plan_steps"])
        state["current_step_index"] = 2
        state["confidence"] = Confidence.HIGH
        state["errors"] = []

        result = await verify_node(state)

        assert result["phase"] == Phase.COMPLETE
        assert result["is_complete"] is True
        assert len(result["final_output"]) > 0

    @pytest.mark.asyncio
    async def test_verify_steps_remaining(self) -> None:
        """step_index < total steps → not complete, EXECUTE phase."""
        state = initial_state("test goal", "thread-remaining")
        state["plan_steps"] = [
            PlanStep(id="s1", description="Step 1", status="completed", result="done"),
            PlanStep(id="s2", description="Step 2", status="pending"),
        ]
        state["completed_steps"] = [state["plan_steps"][0]]
        state["current_step_index"] = 1
        state["confidence"] = Confidence.HIGH
        state["errors"] = []

        result = await verify_node(state)

        assert result["phase"] == Phase.EXECUTE
        assert result["is_complete"] is False

    @pytest.mark.asyncio
    async def test_verify_errors_present(self) -> None:
        """All steps done but errors present → not complete."""
        state = initial_state("test goal", "thread-errs")
        state["plan_steps"] = [
            PlanStep(id="s1", description="Step 1", status="completed", result="done"),
        ]
        state["completed_steps"] = list(state["plan_steps"])
        state["current_step_index"] = 1
        state["confidence"] = Confidence.HIGH
        state["errors"] = ["something went wrong"]

        result = await verify_node(state)

        assert result["phase"] == Phase.EXECUTE
        assert result["is_complete"] is False

    @pytest.mark.asyncio
    async def test_verify_low_confidence(self) -> None:
        """All done, no errors, LOW confidence → not complete."""
        state = initial_state("test goal", "thread-low")
        state["plan_steps"] = [
            PlanStep(id="s1", description="Step 1", status="completed", result="done"),
        ]
        state["completed_steps"] = list(state["plan_steps"])
        state["current_step_index"] = 1
        state["confidence"] = Confidence.LOW
        state["errors"] = []

        result = await verify_node(state)

        assert result["is_complete"] is False
        assert result["phase"] == Phase.EXECUTE

    @pytest.mark.asyncio
    async def test_verify_medium_confidence(self) -> None:
        """All done, no errors, MEDIUM confidence → not complete."""
        state = initial_state("test goal", "thread-med")
        state["plan_steps"] = [
            PlanStep(id="s1", description="Step 1", status="completed", result="done"),
        ]
        state["completed_steps"] = list(state["plan_steps"])
        state["current_step_index"] = 1
        state["confidence"] = Confidence.MEDIUM
        state["errors"] = []

        result = await verify_node(state)

        assert result["is_complete"] is False
        assert result["phase"] == Phase.EXECUTE

    @pytest.mark.asyncio
    async def test_verify_very_high_confidence(self) -> None:
        """All done, no errors, VERY_HIGH confidence → complete."""
        state = initial_state("test goal", "thread-vhigh")
        state["plan_steps"] = [
            PlanStep(id="s1", description="Step 1", status="completed", result="done"),
        ]
        state["completed_steps"] = list(state["plan_steps"])
        state["current_step_index"] = 1
        state["confidence"] = Confidence.VERY_HIGH
        state["errors"] = []

        result = await verify_node(state)

        assert result["is_complete"] is True
        assert result["phase"] == Phase.COMPLETE

    @pytest.mark.asyncio
    async def test_verify_uses_reflection_summary(self) -> None:
        """Reflection present → final_output uses reflection summary."""
        state = initial_state("test goal", "thread-refl")
        state["plan_steps"] = [
            PlanStep(id="s1", description="Step 1", status="completed", result="done"),
        ]
        state["completed_steps"] = list(state["plan_steps"])
        state["current_step_index"] = 1
        state["confidence"] = Confidence.HIGH
        state["errors"] = []
        state["reflection"] = ReflectionResult(
            summary="Custom reflection summary for testing",
            lessons_learned=[],
            confidence=Confidence.HIGH,
            should_evolve=False,
            should_replan=False,
            memory_observations=[],
            cost_efficiency=1.0,
        )

        result = await verify_node(state)

        assert result["is_complete"] is True
        assert "Custom reflection summary" in result["final_output"]

    @pytest.mark.asyncio
    async def test_verify_no_reflection_uses_step_descriptions(self) -> None:
        """No reflection → final_output contains 'Task completed successfully'."""
        state = initial_state("test goal", "thread-norefl")
        state["plan_steps"] = [
            PlanStep(id="s1", description="Step 1", status="completed", result="done"),
        ]
        state["completed_steps"] = list(state["plan_steps"])
        state["current_step_index"] = 1
        state["confidence"] = Confidence.HIGH
        state["errors"] = []
        state["reflection"] = None

        result = await verify_node(state)

        assert result["is_complete"] is True
        assert "Task completed successfully" in result["final_output"]
