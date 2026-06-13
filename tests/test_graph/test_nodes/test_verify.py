"""Tests for src.graph.nodes.verify — verify node function."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.graph.enums import Confidence, Phase
from src.graph.factory import initial_state
from src.graph.models import PlanStep, ReflectionResult
from src.graph.nodes.verify import verify_node
from src.llm.models import LLMResponse


class TestVerifyNode:
    """Tests for the verify_node async function."""

    @pytest.mark.asyncio
    async def test_verify_all_done_no_errors_high_conf(self) -> None:
        """All steps done, no errors, HIGH confidence -> complete."""
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
        """step_index < total steps -> not complete, EXECUTE phase."""
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
        """All steps done but errors present -> not complete."""
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
        """All done, no errors, LOW confidence -> not complete."""
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
        """All done, no errors, MEDIUM confidence -> not complete."""
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
        """All done, no errors, VERY_HIGH confidence -> complete."""
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
        """Reflection present -> final_output uses reflection summary."""
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
        """No reflection -> final_output contains 'Task completed successfully'."""
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


class TestVerifyNodeLLM:
    """Tests for the verify_node LLM verification path."""

    def _build_complete_state(self) -> dict:
        """Build a state that would pass heuristic verification."""
        state = initial_state("test goal for LLM verify", "thread-llm-verify")
        state["plan_steps"] = [
            PlanStep(id="s1", description="Step 1", status="completed", result="done"),
        ]
        state["completed_steps"] = list(state["plan_steps"])
        state["current_step_index"] = 1
        state["confidence"] = Confidence.HIGH
        state["errors"] = []
        return state

    @pytest.mark.asyncio
    async def test_llm_verify_confirms_completion(self) -> None:
        """LLM returns complete=True -> verification passes, phase COMPLETE."""
        state = self._build_complete_state()

        llm_json = (
            '{"is_complete": true, "completion_percentage": 100.0, '
            '"gaps": [], "quality_assessment": "Excellent", "should_evolve": false}'
        )

        gateway = MagicMock()
        gateway.acompletion = AsyncMock(return_value=LLMResponse(
            content=llm_json,
            model="gpt-4o-mini-2024-07-18",
            provider="openai",
            input_tokens=10,
            output_tokens=50,
            total_tokens=60,
            cost_usd=0.0001,
        ))

        result = await verify_node(state, gateway=gateway)

        assert result["phase"] == Phase.COMPLETE
        assert result["is_complete"] is True
        assert "100%" in result["final_output"]
        assert "Excellent" in result["final_output"]

    @pytest.mark.asyncio
    async def test_llm_verify_says_incomplete(self) -> None:
        """LLM returns is_complete=False -> phase EXECUTE, is_complete False."""
        state = self._build_complete_state()

        llm_json = (
            '{"is_complete": false, "completion_percentage": 60.0, '
            '"gaps": ["missing tests", "no error handling"], '
            '"quality_assessment": "Partial", "should_evolve": false}'
        )

        gateway = MagicMock()
        gateway.acompletion = AsyncMock(return_value=LLMResponse(
            content=llm_json,
            model="gpt-4o-mini-2024-07-18",
            provider="openai",
            input_tokens=10,
            output_tokens=50,
            total_tokens=60,
            cost_usd=0.0001,
        ))

        result = await verify_node(state, gateway=gateway)

        assert result["phase"] == Phase.EXECUTE
        assert result["is_complete"] is False
        assert "60%" in result["final_output"]
        assert "Gaps:" in result["final_output"]

    @pytest.mark.asyncio
    async def test_llm_verify_failure_falls_back_to_heuristic(self) -> None:
        """LLM call raises exception -> falls back to heuristic verification."""
        state = self._build_complete_state()

        gateway = MagicMock()
        gateway.acompletion = AsyncMock(side_effect=RuntimeError("API down"))

        result = await verify_node(state, gateway=gateway)

        # Falls back to heuristic — with HIGH conf, no errors, all steps done
        assert result["phase"] == Phase.COMPLETE
        assert result["is_complete"] is True
        assert "Task completed successfully" in result["final_output"]

    @pytest.mark.asyncio
    async def test_llm_verify_returns_unparseable_content_falls_back(self) -> None:
        """LLM returns non-JSON content -> StructuredOutputManager fails, falls back."""
        state = self._build_complete_state()

        gateway = MagicMock()
        gateway.acompletion = AsyncMock(return_value=LLMResponse(
            content="This is not valid JSON at all!",
            model="gpt-4o-mini-2024-07-18",
            provider="openai",
            input_tokens=10,
            output_tokens=10,
            total_tokens=20,
            cost_usd=0.0001,
        ))

        result = await verify_node(state, gateway=gateway)

        # _llm_verify returns None when extraction fails, heuristic runs
        assert result["phase"] == Phase.COMPLETE
        assert result["is_complete"] is True

    @pytest.mark.asyncio
    async def test_llm_verify_with_reflection_summary_in_output(self) -> None:
        """LLM verify with reflection present includes reflection context."""
        state = self._build_complete_state()
        state["reflection"] = ReflectionResult(
            summary="All objectives met successfully",
            lessons_learned=[],
            confidence=Confidence.HIGH,
            should_evolve=False,
            should_replan=False,
            memory_observations=[],
            cost_efficiency=1.0,
        )

        llm_json = (
            '{"is_complete": true, "completion_percentage": 95.0, '
            '"gaps": [], "quality_assessment": "Good", "should_evolve": false}'
        )

        gateway = MagicMock()
        gateway.acompletion = AsyncMock(return_value=LLMResponse(
            content=llm_json,
            model="gpt-4o-mini-2024-07-18",
            provider="openai",
            input_tokens=10,
            output_tokens=50,
            total_tokens=60,
            cost_usd=0.0001,
        ))

        result = await verify_node(state, gateway=gateway)

        assert result["phase"] == Phase.COMPLETE
        assert result["is_complete"] is True
        assert "95%" in result["final_output"]

    @pytest.mark.asyncio
    async def test_verify_gateway_none_uses_heuristic(self) -> None:
        """Gateway is None -> pure heuristic verification, all branches covered."""
        state = self._build_complete_state()
        # This state should pass heuristic: all steps done, HIGH conf, no errors
        result = await verify_node(state, gateway=None)
        assert result["phase"] == Phase.COMPLETE
        assert result["is_complete"] is True

        # Incomplete state with low confidence
        state2 = self._build_complete_state()
        state2["confidence"] = Confidence.LOW
        result2 = await verify_node(state2, gateway=None)
        assert result2["phase"] == Phase.EXECUTE
        assert result2["is_complete"] is False


def _patch_deliverable_roots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> str:
    """Point verify's get_settings at isolated tmp results/workspace roots.

    verify_node reads deliverables off the filesystem via the configured
    ``results_root``/``workspace_root``. Routing both at a tmp dir keeps these
    regression tests hermetic (no real ``results/`` pollution).
    """
    results_root = tmp_path / "results"
    workspace_root = tmp_path / "workspace"
    results_root.mkdir(parents=True, exist_ok=True)
    workspace_root.mkdir(parents=True, exist_ok=True)
    fake_settings = SimpleNamespace(
        agent=SimpleNamespace(
            results_root=str(results_root), workspace_root=str(workspace_root)
        )
    )
    monkeypatch.setattr("src.graph.nodes.verify.get_settings", lambda: fake_settings)
    return str(results_root)


class TestVerifyDeliverableEvidence:
    """Regression tests for independent deliverable verification (F1).

    The verify node must base its verdict on filesystem evidence, not the
    agent's self-report: a declared deliverable that is missing or empty forces
    an incomplete verdict — even when the LLM optimistically says "complete".
    """

    @staticmethod
    def _state_with_file_writer(goal_text: str, file_path: str) -> dict:
        state = initial_state(goal_text, "thread-deliv")
        step = PlanStep(
            id="fw1",
            description=f"Write deliverable to {file_path}",
            tool_name="file_writer",
            tool_input={"file_path": file_path, "content": "payload"},
            status="completed",
            result="wrote file",
        )
        state["plan_steps"] = [step]
        state["completed_steps"] = [step]
        state["current_step_index"] = 1
        state["confidence"] = Confidence.HIGH
        state["errors"] = []
        return state

    @pytest.mark.asyncio
    async def test_missing_deliverable_forces_incomplete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A declared deliverable that does not exist -> not complete + error."""
        _patch_deliverable_roots(monkeypatch, tmp_path)
        state = self._state_with_file_writer(
            "Compute the answer and save it to results/ghost_report.md",
            "results/ghost_report.md",
        )
        result = await verify_node(state, gateway=None)
        assert result["is_complete"] is False
        assert result["phase"] == Phase.EXECUTE
        assert any("ghost_report.md" in e for e in result.get("errors", []))

    @pytest.mark.asyncio
    async def test_present_deliverable_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A declared, non-empty deliverable that exists -> complete."""
        results_root = _patch_deliverable_roots(monkeypatch, tmp_path)
        Path(results_root, "report.md").write_text("the answer", encoding="utf-8")
        state = self._state_with_file_writer(
            "Save the answer to report.md", "report.md"
        )
        result = await verify_node(state, gateway=None)
        assert result["is_complete"] is True
        assert result["phase"] == Phase.COMPLETE

    @pytest.mark.asyncio
    async def test_empty_deliverable_forces_incomplete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A declared deliverable that exists but is 0 bytes -> not complete."""
        results_root = _patch_deliverable_roots(monkeypatch, tmp_path)
        Path(results_root, "empty.md").write_text("", encoding="utf-8")
        state = self._state_with_file_writer(
            "Save the answer to empty.md", "empty.md"
        )
        result = await verify_node(state, gateway=None)
        assert result["is_complete"] is False
        assert any("empty.md" in e for e in result.get("errors", []))

    @pytest.mark.asyncio
    async def test_llm_rubberstamp_overridden_by_missing_deliverable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Filesystem evidence wins over an optimistic LLM 'complete' verdict."""
        _patch_deliverable_roots(monkeypatch, tmp_path)
        state = self._state_with_file_writer(
            "Generate a report and save it to results/final_report.md",
            "results/final_report.md",
        )
        llm_json = (
            '{"is_complete": true, "completion_percentage": 100.0, '
            '"gaps": [], "quality_assessment": "Looks done", "should_evolve": false}'
        )
        gateway = MagicMock()
        gateway.acompletion = AsyncMock(
            return_value=LLMResponse(
                content=llm_json,
                model="gpt-4o-mini-2024-07-18",
                provider="openai",
                input_tokens=10,
                output_tokens=20,
                total_tokens=30,
                cost_usd=0.0001,
            )
        )
        result = await verify_node(state, gateway=gateway)
        assert result["is_complete"] is False
        assert result["phase"] == Phase.EXECUTE
        assert any("final_report.md" in e for e in result.get("errors", []))

    @pytest.mark.asyncio
    async def test_input_file_not_treated_as_deliverable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing READ target must not cause a false negative (T5 scenario)."""
        results_root = _patch_deliverable_roots(monkeypatch, tmp_path)
        Path(results_root, "note.txt").write_text("handled", encoding="utf-8")
        state = initial_state(
            "Read /nonexistent/missing_file.txt; if it fails, save a note to note.txt",
            "thread-t5",
        )
        step = PlanStep(
            id="fw1",
            description="Save error note to note.txt",
            tool_name="file_writer",
            tool_input={"file_path": "note.txt", "content": "handled"},
            status="completed",
            result="wrote note",
        )
        state["plan_steps"] = [step]
        state["completed_steps"] = [step]
        state["current_step_index"] = 1
        state["confidence"] = Confidence.HIGH
        state["errors"] = []
        result = await verify_node(state, gateway=None)
        # note.txt is present -> complete; missing_file.txt is an input, ignored.
        assert result["is_complete"] is True
