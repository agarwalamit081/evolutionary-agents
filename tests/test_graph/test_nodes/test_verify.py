"""Tests for src.graph.nodes.verify — verify node function."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.graph.enums import Confidence, Phase, TaskComplexity
from src.graph.factory import initial_state
from src.graph.models import Goal, GoalStatus, PlanStep, ReflectionResult, ToolResult
from src.graph.nodes.verify import (
    _extract_deliverable_paths,
    _load_deliverable_content,
    _summarize_data_tool_outputs,
    verify_node,
)
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

    @pytest.mark.asyncio
    async def test_critical_complexity_threaded_to_gateway(self) -> None:
        """A CRITICAL goal routes verification to a stronger model (§5 C.1)."""
        state = self._build_complete_state()
        state["current_goal"] = Goal(
            text="test goal for LLM verify",
            status=GoalStatus.ACTIVE,
            complexity=TaskComplexity.CRITICAL,
        )

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

        await verify_node(state, gateway=gateway)

        assert (
            gateway.acompletion.call_args.kwargs["complexity"]
            == TaskComplexity.CRITICAL
        )

    @pytest.mark.asyncio
    async def test_unclassified_goal_defaults_to_simple(self) -> None:
        """A goal without a classified complexity falls back to SIMPLE."""
        state = self._build_complete_state()
        # initial_state builds a Goal with default complexity=SIMPLE.
        assert state["current_goal"].complexity == TaskComplexity.SIMPLE

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

        await verify_node(state, gateway=gateway)

        assert (
            gateway.acompletion.call_args.kwargs["complexity"]
            == TaskComplexity.SIMPLE
        )


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
    async def test_pessimistic_llm_overridden_by_present_deliverable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Filesystem evidence wins over a PESSIMISTIC LLM verdict too (Q9).

        Symmetric to test_llm_rubberstamp_overridden_by_missing_deliverable: a
        declared deliverable that EXISTS + all steps done + no errors forces
        is_complete=True even when the verify LLM self-reports 0%. Without this,
        a deliverable-producing goal loops verify→plan until the iteration cap
        (Q9 wrote results/q9_onboarding.md but verify returned complete=False).
        """
        results_root = _patch_deliverable_roots(monkeypatch, tmp_path)
        Path(results_root, "final_report.md").write_text(
            "the full deliverable content", encoding="utf-8"
        )
        state = self._state_with_file_writer(
            "Generate a report and save it to results/final_report.md",
            "results/final_report.md",
        )
        llm_json = (
            '{"is_complete": false, "completion_percentage": 0.0, '
            '"gaps": ["uncertain"], "quality_assessment": "Cannot confirm", '
            '"should_evolve": false}'
        )
        gateway = MagicMock()
        gateway.acompletion = AsyncMock(
            return_value=LLMResponse(
                content=llm_json,
                model="claude-haiku-4-5-20251001",
                provider="anthropic",
                input_tokens=10,
                output_tokens=20,
                total_tokens=30,
                cost_usd=0.0001,
            )
        )
        result = await verify_node(state, gateway=gateway)
        assert result["is_complete"] is True
        assert result["phase"] == Phase.COMPLETE
        assert "Objective deliverable evidence" in result["final_output"]

    @pytest.mark.asyncio
    async def test_present_deliverable_not_forced_complete_with_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Errors block the optimistic override — evidence of success is not
        enough when the run also has unresolved errors."""
        results_root = _patch_deliverable_roots(monkeypatch, tmp_path)
        Path(results_root, "final_report.md").write_text("content", encoding="utf-8")
        state = self._state_with_file_writer(
            "Save the answer to results/final_report.md",
            "results/final_report.md",
        )
        state["errors"] = ["something went wrong mid-run"]
        llm_json = (
            '{"is_complete": false, "completion_percentage": 50.0, '
            '"gaps": [], "quality_assessment": "partial", "should_evolve": false}'
        )
        gateway = MagicMock()
        gateway.acompletion = AsyncMock(
            return_value=LLMResponse(
                content=llm_json,
                model="claude-haiku-4-5-20251001",
                provider="anthropic",
                input_tokens=10,
                output_tokens=20,
                total_tokens=30,
                cost_usd=0.0001,
            )
        )
        result = await verify_node(state, gateway=gateway)
        assert result["is_complete"] is False

    @pytest.mark.asyncio
    async def test_present_deliverable_not_forced_complete_with_steps_remaining(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unfinished plan blocks the override — don't complete mid-plan."""
        results_root = _patch_deliverable_roots(monkeypatch, tmp_path)
        Path(results_root, "final_report.md").write_text("content", encoding="utf-8")
        state = self._state_with_file_writer(
            "Save the answer to results/final_report.md",
            "results/final_report.md",
        )
        # Two steps, only the first done.
        state["plan_steps"] = list(state["plan_steps"]) + [
            PlanStep(id="s2", description="Review the report", status="pending")
        ]
        state["current_step_index"] = 1
        llm_json = (
            '{"is_complete": false, "completion_percentage": 50.0, '
            '"gaps": [], "quality_assessment": "partial", "should_evolve": false}'
        )
        gateway = MagicMock()
        gateway.acompletion = AsyncMock(
            return_value=LLMResponse(
                content=llm_json,
                model="claude-haiku-4-5-20251001",
                provider="anthropic",
                input_tokens=10,
                output_tokens=20,
                total_tokens=30,
                cost_usd=0.0001,
            )
        )
        result = await verify_node(state, gateway=gateway)
        assert result["is_complete"] is False

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

    def test_determiner_captures_are_not_deliverables(self) -> None:
        """Regression: "create ... in a module" must not yield deliverable "a".

        ``_DIR_OUTPUT_RE`` can capture the determiner after the preposition
        ("Create the tool in a module" -> "a"; "Produce the report under the
        results dir" -> "the"). Treated as a deliverable, such a token reads as
        missing and loops verify->plan until the iteration hard-cap (observed on
        battery-02 N1: 455s looping on a phantom "a"). ``_add()`` rejects single
        chars and prose tokens so only real paths survive.
        """
        state = initial_state(
            "Create a tool called char_counter. Save the metrics to "
            "results/n1_char_counter.md.",
            "thread-noise",
        )
        state["plan_steps"] = [
            PlanStep(
                id="s1",
                description="Create the char_counter tool in a new Python module.",
                tool_name=None,
                tool_input={},
                status="pending",
                result="",
            ),
            PlanStep(
                id="s2",
                description="Produce the report under the results directory.",
                tool_name=None,
                tool_input={},
                status="pending",
                result="",
            ),
            PlanStep(
                id="s3",
                description="Save the metrics to results/n1_char_counter.md.",
                tool_name="file_writer",
                tool_input={"file_path": "results/n1_char_counter.md", "content": "x"},
                status="completed",
                result="wrote",
            ),
        ]
        state["completed_steps"] = [state["plan_steps"][2]]
        paths = _extract_deliverable_paths(state)
        assert "results/n1_char_counter.md" in paths
        assert "a" not in paths
        assert "the" not in paths

    def test_quantifier_captures_are_not_deliverables(self) -> None:
        """Regression: "Create ... appears in multiple files" must not yield "multiple".

        ``_DIR_OUTPUT_RE`` matches ``create ... in <token>`` across a full plan
        step, so an LLM-authored step like "Create the duplicate_finder tool
        ... tracks which lines appear in multiple files" captures the quantifier
        "multiple" as a deliverable. That phantom is never on disk, so
        ``_check_deliverables`` flags it missing and ``_force_complete_on_evidence``
        bails — looping verify 4+ cycles until the iteration hard-cap (observed on
        the post-fix N5 re-run). ``_PATH_NOISE_TOKENS`` now rejects these
        plurality/quantifier adjectives so only the real ``save ... to`` path
        survives.
        """
        state = initial_state(
            "Create a short tool called duplicate_finder that takes a glob and "
            "returns duplicate non-empty lines across matched files. Run it on "
            "results/*.md and save the output to results/n5_dups.md.",
            "thread-phantom",
        )
        state["plan_steps"] = [
            PlanStep(
                id="s1",
                description=(
                    "Create the duplicate_finder tool as a Python script that: "
                    "(1) accepts a glob pattern, (2) finds matching files, "
                    "(3) reads all non-empty lines from each file, (4) tracks "
                    "which lines appear in multiple files, (5) returns/outputs "
                    "duplicates with file locations"
                ),
                tool_name=None,
                tool_input={},
                status="pending",
                result="",
            ),
            PlanStep(
                id="s2",
                description="Save the output to results/n5_dups.md.",
                tool_name="file_writer",
                tool_input={"file_path": "results/n5_dups.md", "content": "x"},
                status="completed",
                result="wrote",
            ),
        ]
        state["completed_steps"] = [state["plan_steps"][1]]
        paths = _extract_deliverable_paths(state)
        assert "results/n5_dups.md" in paths
        assert "multiple" not in paths
        assert "several" not in paths
        assert "various" not in paths


def _user_prompt_from(messages: list) -> str:
    """Extract the user-role message content captured by a mock gateway."""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


class TestVerifyDeliverableHonesty:
    """Deliverable-honesty check (battery-02 N6): the verify LLM must receive
    the deliverable's own content AND the real tool outputs as ground truth so a
    well-structured but fabricated deliverable (synthesized counts) is detected
    rather than rubber-stamped."""

    @staticmethod
    def _state(goal_text: str, file_path: str, content: str) -> dict:
        state = initial_state(goal_text, "thread-honesty")
        step = PlanStep(
            id="fw1",
            description=f"Save the report to {file_path}",
            tool_name="file_writer",
            tool_input={"file_path": file_path, "content": content},
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
    async def test_prompt_includes_deliverable_content_and_tool_outputs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The verify user prompt must carry the deliverable content, the tool
        outputs, and the honesty instruction that flags ungrounded claims."""
        results_root = _patch_deliverable_roots(monkeypatch, tmp_path)
        Path(results_root, "log_dups.md").write_text(
            "Connection timeout error | 24\n", encoding="utf-8"
        )
        state = self._state(
            "Scan logs/*.log for duplicate errors and save to results/log_dups.md",
            "results/log_dups.md",
            "Connection timeout error | 24\n",
        )
        state["tool_results"] = [
            ToolResult(
                tool_name="duplicate_finder",
                success=True,
                output="Disk full: 2\nConnection timeout: 5",
            ),
        ]

        llm_json = (
            '{"is_complete": true, "completion_percentage": 100.0, '
            '"gaps": [], "quality_assessment": "ok", "should_evolve": false}'
        )
        gateway = MagicMock()
        gateway.acompletion = AsyncMock(
            return_value=LLMResponse(
                content=llm_json,
                model="claude-haiku-4-5-20251001",
                provider="anthropic",
                input_tokens=10,
                output_tokens=20,
                total_tokens=30,
                cost_usd=0.0001,
            )
        )
        await verify_node(state, gateway=gateway)

        prompt = _user_prompt_from(gateway.acompletion.call_args.kwargs["messages"])
        # Deliverable content + tool outputs are present as ground truth ...
        assert "Connection timeout error | 24" in prompt
        assert "duplicate_finder" in prompt
        assert "Disk full: 2" in prompt
        # ... and the honesty instruction that drives the comparison.
        assert "HONESTY CHECK" in prompt
        assert "UNGROUND" in prompt.upper()

    def test_load_deliverable_content_reads_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_load_deliverable_content reads a present deliverable's text."""
        results_root = _patch_deliverable_roots(monkeypatch, tmp_path)
        Path(results_root, "report.md").write_text("real content here", encoding="utf-8")
        text = _load_deliverable_content(["results/report.md"])
        assert "real content here" in text

    def test_load_deliverable_content_truncates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Content beyond the cap is truncated (prompt stays bounded)."""
        results_root = _patch_deliverable_roots(monkeypatch, tmp_path)
        Path(results_root, "big.md").write_text("X" * 7000, encoding="utf-8")
        text = _load_deliverable_content(["results/big.md"])
        assert "…[truncated]" in text
        assert len(text) < 7000

    def test_summarize_data_tool_outputs_excludes_file_writer(self) -> None:
        """Only successful data-producing tools are summarized; file_writer
        confirmations and failed tools are excluded."""
        tool_results = [
            ToolResult(
                tool_name="file_writer",
                success=True,
                output="Successfully wrote 100 bytes to results/x.md",
            ),
            ToolResult(
                tool_name="duplicate_finder",
                success=True,
                output="timeout: 5\ndisk full: 2",
            ),
            ToolResult(
                tool_name="broken_tool",
                success=False,
                output="boom",
            ),
        ]
        summary = _summarize_data_tool_outputs(tool_results)
        assert "duplicate_finder" in summary
        assert "file_writer" not in summary
        assert "broken_tool" not in summary
