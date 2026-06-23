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
    _extract_goal_deliverables,
    _failure_reason,
    _force_complete_on_evidence,
    _load_deliverable_content,
    _spot_check_cited_paths,
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
    """Point the shared settings source at isolated tmp results/workspace roots.

    verify_node resolves deliverables via the shared path resolver, which reads
    ``src.config.settings.get_settings`` (the single source) — so that is what we
    patch, not a per-module symbol. Routing both roots at a tmp dir keeps these
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
    monkeypatch.setattr("src.config.settings.get_settings", lambda: fake_settings)
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
        enough when the run also has unresolved errors AND the goal did not name
        a specific deliverable that is now satisfied. (battery-04 q2 F-h.2: when
        the GOAL's own named deliverable is present + non-empty, stale errors are
        advisory and force-complete fires — that case is covered separately in
        ``test_verify_grounding.py::TestGoalDeliverableSufficiencyFH``. Here the
        goal is generic, so the deliverable is file_writer-declared only and the
        unresolved error must still prevent completion.)"""
        results_root = _patch_deliverable_roots(monkeypatch, tmp_path)
        Path(results_root, "final_report.md").write_text("content", encoding="utf-8")
        state = self._state_with_file_writer(
            "Analyze the data and produce a concise final report.",
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

    def test_abbreviation_and_prose_tokens_are_not_deliverables(self) -> None:
        """Regression: battery-03 q2/q3 leaked "e.g" and "mixed" as deliverables.

        The ``[stem].[ext]`` capture group matches the abbreviation "e.g"
        (stem "e" + ext "g") and "i.e" ("i" + "e"), and ``_DIR_OUTPUT_RE``
        captures the prose word "mixed" from "generate ... in mixed human
        formats". Each phantom is never on disk, so ``_check_deliverables``
        flags it missing and ``_force_complete_on_evidence`` bails — forcing
        ``is_complete=False`` on runs whose real deliverable IS present
        (observed: battery-03 q3 finished with a correct results/q03_stats.csv
        yet Completed=False, verify WARNING listed "e.g" as a missing output).
        ``_PATH_NOISE_TOKENS`` now rejects these so only the real path survives.
        """
        state = initial_state(
            "Create a reusable tool named date_normalizer that extracts dates "
            "written in mixed human formats (e.g. 'Jan 5th, 2024', '05/01/2024') "
            "and writes the normalized ISO-8601 results to results/q09_dates.jsonl.",
            "thread-battery03-noise",
        )
        state["plan_steps"] = [
            PlanStep(
                id="s1",
                description=(
                    "Generate example inputs in mixed human formats "
                    "(e.g. 'Jan 5th, 2024') and write the normalized output, "
                    "i.e. to results/q09_dates.jsonl, documenting each match"
                ),
                tool_name=None,
                tool_input={},
                status="pending",
                result="",
            ),
            PlanStep(
                id="s2",
                description="Write the normalized results to results/q09_dates.jsonl.",
                tool_name="file_writer",
                tool_input={"file_path": "results/q09_dates.jsonl", "content": "{}"},
                status="completed",
                result="wrote",
            ),
        ]
        state["completed_steps"] = [state["plan_steps"][1]]
        paths = _extract_deliverable_paths(state)
        assert "results/q09_dates.jsonl" in paths
        assert "mixed" not in paths
        assert "e.g" not in paths
        assert "i.e" not in paths


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

    @pytest.mark.asyncio
    async def test_fabrication_warning_is_advisory_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The filesystem spot-check is advisory: its fabrication warning
        surfaces in the verify prompt but never forces an incomplete verdict.

        The deliverable physically exists yet claims "Found 12 duplicate files"
        while ``results/`` holds only that one file, so
        ``_spot_check_cited_paths`` emits a fabrication warning. That warning is
        interpolated into the prompt (advisory input to the LLM), but it does
        NOT itself touch ``_enforce_deliverables`` /
        ``_force_complete_on_evidence``: with a present, non-empty deliverable,
        all steps done, and no errors, the verdict is still forced COMPLETE. The
        spot-check therefore cannot reintroduce the verify→plan loop (the P0c
        design contract). This is NOT "we accept fabrication" — the verdict is
        governed by the deliverable-evidence clamps; the spot-check only
        *advises* the LLM.
        """
        results_root = _patch_deliverable_roots(monkeypatch, tmp_path)
        # Only the deliverable itself is on disk — it claims 12 files, not 1.
        Path(results_root, "n6_dups.md").write_text(
            "Found 12 duplicate files\n", encoding="utf-8"
        )
        state = self._state(
            "Scan for duplicate lines and save the summary to results/n6_dups.md",
            "results/n6_dups.md",
            "Found 12 duplicate files\n",
        )

        llm_json = (
            '{"is_complete": false, "completion_percentage": 0.0, '
            '"gaps": ["uncertain counts"], "quality_assessment": "Cannot confirm", '
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

        prompt = _user_prompt_from(gateway.acompletion.call_args.kwargs["messages"])
        # (1) Advisory: the fabrication warning reached the verify prompt.
        assert "counts may be fabricated" in prompt
        # (2) Advisory-only: it did not force incomplete — the present
        #     deliverable still completes (no verify→plan loop).
        assert result["is_complete"] is True
        assert result["phase"] == Phase.COMPLETE


class TestSpotCheckCitedPaths:
    """Unit tests for the filesystem spot-check (P0c fabrication guard).

    ``_spot_check_cited_paths`` cross-checks the counts/paths the deliverable
    cites against the actual filesystem and returns an *advisory* warning string
    — it never forces a verdict. Each test points the shared settings source at
    an isolated ``tmp_path`` via ``_patch_deliverable_roots`` so no real
    ``results/`` is touched.
    """

    def test_no_counts_no_cited_paths_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Innocuous text with no counts or cited paths → no warning."""
        _patch_deliverable_roots(monkeypatch, tmp_path)
        assert _spot_check_cited_paths("The task is essentially done.", "") == ""

    def test_cited_path_present_no_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cited results/<file> that exists on disk → no warning."""
        results_root = _patch_deliverable_roots(monkeypatch, tmp_path)
        Path(results_root, "report.md").write_text("payload", encoding="utf-8")
        assert _spot_check_cited_paths("See results/report.md for details.", "") == ""

    def test_missing_cited_path_warns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cited results/<file> that does NOT exist → cited-path warning."""
        _patch_deliverable_roots(monkeypatch, tmp_path)
        warning = _spot_check_cited_paths("Saved to results/ghost.md.", "")
        assert "Cited deliverable paths not found on disk" in warning
        assert "results/ghost.md" in warning

    def test_count_with_empty_results_warns_fabrication(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file-count claim against an empty results/ → fabrication warning."""
        _patch_deliverable_roots(monkeypatch, tmp_path)
        warning = _spot_check_cited_paths("Found 12 duplicate files.", "")
        assert "~12" in warning
        assert "holds 0" in warning
        assert "fabricated" in warning

    def test_count_satisfied_no_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file-count claim that matches the on-disk count → no warning."""
        results_root = _patch_deliverable_roots(monkeypatch, tmp_path)
        for i in range(3):
            Path(results_root, f"f{i}.md").write_text("x", encoding="utf-8")
        assert _spot_check_cited_paths("Identified 3 files.", "") == ""

    def test_non_file_count_noun_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Counts of non-file entities (rows/entries) do not trigger the guard.

        Rows/entries may live in a DB or in-memory, so a high count is not a
        reliable fabrication signal here — this documents the false-positive
        guard (battery-02 N6 synthesized DB-style counts).
        """
        _patch_deliverable_roots(monkeypatch, tmp_path)
        assert _spot_check_cited_paths("Found 500 rows in the table.", "") == ""

    def test_counts_scanned_across_both_inputs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Counts in the deliverable AND the tool outputs are scanned together;
        the largest file-count drives the warning (claimed = max)."""
        _patch_deliverable_roots(monkeypatch, tmp_path)
        warning = _spot_check_cited_paths("Wrote 3 files.", "matched 50 duplicates")
        assert "~50" in warning
        assert "fabricated" in warning


class TestExtractGoalDeliverables:
    """F4: extract deliverable cues named by the GOAL only (text + criteria).

    Distinct from ``_extract_deliverable_paths``: the goal helper builds its
    input-exclusion set over a goal-only blob, so a deliverable the goal says to
    "write" survives even when a later plan step *reads* the same path as input.
    That is exactly the F4 false-positive — the agent writes an intermediate and
    the real goal file is excluded from the deliverable list, so the missing-file
    clamp never fires and force-complete marks a half-finished run done.
    """

    def test_extracts_named_file_from_goal_text(self) -> None:
        state = initial_state(
            "Normalize e-commerce events and save the output to "
            "results/q01/normalized.csv",
            "thread-f4-text",
        )
        assert _extract_goal_deliverables(state) == ["results/q01/normalized.csv"]

    def test_extracts_named_file_from_success_criteria(self) -> None:
        state = initial_state("Produce a normalized dataset.", "thread-f4-criteria")
        state["current_goal"] = Goal(
            text="Produce a normalized dataset.",
            status=GoalStatus.ACTIVE,
            success_criteria=[
                "Write the normalized rows to results/q01/normalized.csv",
            ],
        )
        assert _extract_goal_deliverables(state) == ["results/q01/normalized.csv"]

    def test_survives_when_a_plan_step_reads_the_same_path(self) -> None:
        """The goal-only blob means a plan-step *read* never excludes the goal file.

        ``_extract_deliverable_paths`` computes its input-exclusion set over the
        whole goal+plan blob, so "read results/q01/normalized.csv to summarize"
        in a plan step would drop ``normalized.csv`` from the deliverable list.
        The goal helper is blind to plan steps, so the goal deliverable survives.
        """
        state = initial_state(
            "Normalize events and save the output to results/q01/normalized.csv",
            "thread-f4-read",
        )
        state["plan_steps"] = [
            PlanStep(
                id="s1",
                description="Read results/q01/normalized.csv and summarize it.",
                tool_name=None,
                tool_input={},
                status="completed",
                result="",
            ),
        ]
        # Goal helper still sees the deliverable the goal asks for...
        assert _extract_goal_deliverables(state) == ["results/q01/normalized.csv"]
        # ...whereas the whole-blob deliverable extractor excludes it as an input.
        assert "results/q01/normalized.csv" not in _extract_deliverable_paths(state)

    def test_no_goal_returns_empty(self) -> None:
        state = initial_state("A goal with no named output file.", "thread-f4-none")
        assert _extract_goal_deliverables(state) == []

    # ── Goal-deliverable LABEL cues (_GOAL_DELIVERABLE_LABEL_RE) ──────────────
    # The goal often names its PRIMARY deliverable with an explicit label
    # ("DELIVERABLE: final report results/vector_db_brief.md") where the word
    # preceding the path is a descriptor noun ("report"), not a save-verb.
    # _SAVE_TO_RE/_DIR_OUTPUT_RE both miss such a path, so the primary
    # deliverable silently drops from the required set and goal_satisfied can
    # flip True on present secondaries alone → premature force-complete with the
    # primary never written (showcase vector-db run ended 50% this way). These
    # lock the label-cue capture.

    def test_extracts_label_cue_deliverable_capitalized(self) -> None:
        """'DELIVERABLE: final report results/vector_db_brief.md' — the
        showcase phrasing. The label regex captures the brief despite the
        'report' descriptor between the label and the path."""
        state = initial_state(
            "Build a competitive-intelligence brief. "
            "DELIVERABLE: final report results/vector_db_brief.md embedding "
            "the comparison table and a recommendation.",
            "thread-label-brief",
        )
        assert "results/vector_db_brief.md" in _extract_goal_deliverables(state)

    def test_extracts_final_output_label_deliverable(self) -> None:
        state = initial_state(
            "Compute the score and produce it. final output: results/answer.json",
            "thread-label-output",
        )
        assert "results/answer.json" in _extract_goal_deliverables(state)

    def test_extracts_output_file_label_deliverable(self) -> None:
        state = initial_state(
            "Analyze the dataset. output file results/out.csv must list every row.",
            "thread-label-outfile",
        )
        assert "results/out.csv" in _extract_goal_deliverables(state)

    def test_label_deliverable_and_save_verb_deliverable_both_captured(self) -> None:
        """A goal naming a label-deliverable AND a save-verb deliverable
        captures BOTH — so goal_satisfied cannot satisfy on the secondary alone.

        This is the exact showcase scenario: comparison.csv (write-verb) +
        vector_db_brief.md (DELIVERABLE label). Before the fix only
        comparison.csv was captured, so a present comparison.csv read as
        goal-satisfied and the run force-completed with the brief absent.
        """
        state = initial_state(
            "3. COMPARISON: write results/comparison.csv via the tool. "
            "5. DELIVERABLE: final report results/vector_db_brief.md with the "
            "comparison table and trade-offs.",
            "thread-label-both",
        )
        captured = _extract_goal_deliverables(state)
        assert "results/comparison.csv" in captured
        assert "results/vector_db_brief.md" in captured

    def test_label_cue_ignores_non_path_descriptor(self) -> None:
        """A label cue NOT followed by a path-like token (only prose) captures
        nothing — 'final report embedding a recommendation' has no dotted path,
        so the [^.]*? skip finds no deliverable and no phantom is invented."""
        state = initial_state(
            "DELIVERABLE: final report embedding a thorough recommendation.",
            "thread-label-nopath",
        )
        assert _extract_goal_deliverables(state) == []


class TestForceCompleteGoalCrossCheck:
    """F4 regression: force-complete must decline when the agent produced only
    intermediates.

    Before the fix, ``_force_complete_on_evidence`` trusted any declared
    deliverable: if the agent wrote an intermediate file, all steps were done,
    and no errors remained, the run was marked COMPLETE even though the GOAL's
    expected deliverable was never produced. The cross-check now requires at
    least one declared deliverable to match the goal's expected basename.
    """

    @staticmethod
    def _state(goal_text: str) -> dict:
        """All-steps-done, error-free state whose goal names a specific file."""
        state = initial_state(goal_text, "thread-f4-force")
        state["plan_steps"] = [
            PlanStep(
                id="s1", description="Do the work.", status="completed", result="ok"
            )
        ]
        state["completed_steps"] = list(state["plan_steps"])
        state["current_step_index"] = 1
        state["confidence"] = Confidence.HIGH
        state["errors"] = []
        return state

    @staticmethod
    def _incomplete_result() -> dict:
        return {
            "is_complete": False,
            "phase": Phase.VERIFY,
            "final_output": "Cannot confirm completion yet.",
        }

    def test_intermediate_only_declines_force_complete(self) -> None:
        """Goal wants ``normalized.csv``; agent declared only ``raw_dump.csv``.

        Both are present on disk (deliverable_problems is empty), but the goal
        deliverable basename is absent from the declared set → decline.
        """
        state = self._state(
            "Normalize events and save the output to results/q01/normalized.csv"
        )
        result = _force_complete_on_evidence(
            self._incomplete_result(),
            state,
            deliverable_paths=["results/q01/raw_dump.csv"],
            deliverable_problems=[],
        )
        assert result["is_complete"] is False
        assert result["phase"] == Phase.VERIFY

    def test_goal_deliverable_present_forces_complete(self) -> None:
        """Positive control: declared deliverable matches the goal basename → complete."""
        state = self._state(
            "Normalize events and save the output to results/q01/normalized.csv"
        )
        result = _force_complete_on_evidence(
            self._incomplete_result(),
            state,
            deliverable_paths=["results/q01/normalized.csv"],
            deliverable_problems=[],
        )
        assert result["is_complete"] is True
        assert result["phase"] == Phase.COMPLETE

    def test_no_goal_expectation_still_forces_complete(self) -> None:
        """Backward-compat: a goal that names no specific file skips the cross-check."""
        state = self._state("Summarize the dataset in a report.")
        result = _force_complete_on_evidence(
            self._incomplete_result(),
            state,
            deliverable_paths=["results/summary.md"],
            deliverable_problems=[],
        )
        assert result["is_complete"] is True
        assert result["phase"] == Phase.COMPLETE

    def test_basename_match_across_subdir_forces_complete(self) -> None:
        """The cross-check matches on basename, so ``results/<run>/normalized.csv``
        satisfies a goal that named ``results/q01/normalized.csv`` (per-run paths)."""
        state = self._state(
            "Normalize events and save the output to results/q01/normalized.csv"
        )
        result = _force_complete_on_evidence(
            self._incomplete_result(),
            state,
            deliverable_paths=["results/q01_run_a/normalized.csv"],
            deliverable_problems=[],
        )
        assert result["is_complete"] is True
        assert result["phase"] == Phase.COMPLETE


class TestForceCompleteGoalSatisfied:
    """F-h.3 regression: a run whose GOAL deliverables are satisfied must
    force-complete even when plan steps remain — the all_steps_done gate
    re-looped battery-04 q3 (~19 wasted iterations).

    ``_force_complete_on_evidence``'s ``goal_satisfied`` branch previously
    required ``all_steps_done``; a reflect replan or verify-retry churn kept
    ``step_index < len(plan_steps)`` forever, so an objectively-complete goal
    looped verify→execute until the iteration hard-cap. The fix drops the step
    gate (goal deliverables present = success) and keeps only a small iteration
    floor to rule out stale-prior-run deliverables.
    """

    @staticmethod
    def _state(goal_text: str, *, step_index: int, iteration: int) -> dict:
        """State with a 5-step plan; ``step_index``/``iteration`` are tunable so
        all_steps_done and the iteration floor can be exercised independently."""
        state = initial_state(goal_text, "thread-fh3-force")
        state["plan_steps"] = [
            PlanStep(id=f"s{i}", description=f"Step {i}.", status="pending")
            for i in range(5)
        ]
        state["current_step_index"] = step_index
        state["iteration_count"] = iteration
        state["errors"] = []
        return state

    @staticmethod
    def _incomplete_result() -> dict:
        return {
            "is_complete": False,
            "phase": Phase.VERIFY,
            "final_output": "LLM judge: 75% complete, data gaps remain.",
        }

    def test_goal_satisfied_forces_complete_with_steps_remaining(self) -> None:
        """The q3 loop's exact shape: goal satisfied, steps remaining, LLM
        pessimistic → must STILL force-complete once past the iteration floor."""
        state = self._state(
            "Create a cohort retention tool, writing results/q03/retention.csv "
            "and results/q03/churn_flags.csv.",
            step_index=3,  # 3 < 5 → all_steps_done is False (the old blocker)
            iteration=38,  # well past the floor; deliverables were written this run
        )
        result = _force_complete_on_evidence(
            self._incomplete_result(),
            state,
            deliverable_paths=["results/q03/retention.csv", "results/q03/churn_flags.csv"],
            deliverable_problems=["normalized.csv (malformed)"],  # advisory intermediate
            goal_satisfied=True,
        )
        assert result["is_complete"] is True
        assert result["phase"] == Phase.COMPLETE

    def test_goal_satisfied_below_iter_floor_declines(self) -> None:
        """Stale-deliverable guard: goal satisfied on the first verify of a fresh
        run (iteration below floor) must NOT force-complete — those deliverables
        may be leftovers from a prior, uncleaned run."""
        state = self._state(
            "Write results/q/retention.csv.",
            step_index=0,
            iteration=1,  # below _GOAL_COMPLETE_MIN_ITER
        )
        result = _force_complete_on_evidence(
            self._incomplete_result(),
            state,
            deliverable_paths=["results/q/retention.csv"],
            deliverable_problems=[],
            goal_satisfied=True,
        )
        assert result["is_complete"] is False
        assert result["phase"] == Phase.VERIFY

    def test_goal_satisfied_all_steps_done_completes(self) -> None:
        """Positive control: all steps done + goal satisfied still completes."""
        state = self._state(
            "Write results/q/retention.csv.",
            step_index=5,  # 5 >= 5 → all_steps_done True
            iteration=20,
        )
        result = _force_complete_on_evidence(
            self._incomplete_result(),
            state,
            deliverable_paths=["results/q/retention.csv"],
            deliverable_problems=[],
            goal_satisfied=True,
        )
        assert result["is_complete"] is True
        assert result["phase"] == Phase.COMPLETE


class TestCorrectnessFailureReason:
    """``_failure_reason`` extracts an actionable verdict from a failing check so
    eval_enforce can tell the agent *what* to fix (battery-04 q4: a
    {"passed":0,"failed":0} deliverable must surface "no tests executed", not a
    bare "failed"). Duck-typed over the CheckResult evidence/error shape.
    """

    def _check(self, **kw: object) -> SimpleNamespace:
        # Defaults mimic a failing, non-skipped CheckResult; kw overrides.
        defaults: dict[str, object] = {
            "passed": False,
            "skipped": False,
            "evidence": {},
            "error": None,
        }
        defaults.update(kw)
        return SimpleNamespace(**defaults)

    def test_prefers_execution_probe_stdout_last_line(self) -> None:
        check = self._check(
            evidence={"stdout": "running probe...\nno tests executed (all counts zero)"}
        )
        assert _failure_reason(check) == "no tests executed (all counts zero)"

    def test_falls_back_to_evidence_reason(self) -> None:
        check = self._check(evidence={"reason": "target deliverable not on disk"})
        assert _failure_reason(check) == "target deliverable not on disk"

    def test_falls_back_to_error_field(self) -> None:
        check = self._check(error="parse failed: invalid JSON: ...")
        assert _failure_reason(check).startswith("parse failed")

    def test_truncates_long_stdout(self) -> None:
        check = self._check(evidence={"stdout": "x" * 500})
        assert len(_failure_reason(check)) <= 200

    def test_empty_evidence_yields_default(self) -> None:
        assert _failure_reason(self._check()) == "failed"

