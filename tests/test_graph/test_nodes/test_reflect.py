"""Tests for src.graph.nodes.reflect — reflect node function."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.graph.enums import Confidence, Phase, TaskComplexity
from src.graph.factory import initial_state
from src.graph.models import Goal, GoalStatus, PlanStep, ReflectionResult, ToolResult
from src.graph.nodes.reflect import (
    _check_and_fold,
    _derive_verified_actions,
    _format_available_tools,
    _format_step_results,
    _format_successful_tools,
    _ground_should_evolve,
    reflect_node,
)


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
    async def test_reflect_critical_complexity_threaded_to_gateway(
        self, state_after_execution: dict, mock_gateway: object
    ) -> None:
        """A CRITICAL goal routes reflection to a stronger model (§5 C.1)."""
        original = state_after_execution["current_goal"]
        state_after_execution["current_goal"] = Goal(
            text=original.text if original else "test goal",
            status=GoalStatus.ACTIVE,
            complexity=TaskComplexity.CRITICAL,
        )

        await reflect_node(state_after_execution, gateway=mock_gateway)

        assert (
            mock_gateway.acompletion.call_args.kwargs["complexity"]
            == TaskComplexity.CRITICAL
        )

    @pytest.mark.asyncio
    async def test_reflect_simple_goal_routes_simple_not_complex(
        self, state_after_execution: dict, mock_gateway: object
    ) -> None:
        """A SIMPLE-classified goal routes reflect to SIMPLE (goal-driven, §5 C.1).

        The COMPLEX fallback only fires for an explicit-None complexity
        (defensive); a normally-constructed Goal defaults to SIMPLE, so the
        classified value wins.
        """
        original = state_after_execution["current_goal"]
        state_after_execution["current_goal"] = Goal(
            text=original.text if original else "test goal",
            status=GoalStatus.ACTIVE,
            complexity=TaskComplexity.SIMPLE,
        )

        await reflect_node(state_after_execution, gateway=mock_gateway)

        assert (
            mock_gateway.acompletion.call_args.kwargs["complexity"]
            == TaskComplexity.SIMPLE
        )

    @pytest.mark.asyncio
    async def test_reflect_memory_observations_on_errors(self, state_with_errors: dict) -> None:
        """Errors present → memory_observations list is non-empty."""
        state_with_errors["plan_steps"] = [
            PlanStep(id="s1", description="Step 1", status="pending"),
        ]
        state_with_errors["completed_steps"] = []
        result = await reflect_node(state_with_errors)

        assert len(result["memory_observations"]) > 0


class TestDeriveVerifiedActions:
    """WS5: _derive_verified_actions grounds folded tool memory in real execution.

    Implements GenericAgent's "No Execution, No Memory" principle — only tools
    that actually ran and succeeded ≥1 time are 'verified'. Pure function, no
    LLM call; deterministic.
    """

    def test_successful_tool_is_verified(self) -> None:
        """A tool with ≥1 success is included with calls/successes counts."""
        results = [
            ToolResult(tool_name="web_search", success=True, output="ok"),
            ToolResult(tool_name="web_search", success=False, output="", error="timeout"),
        ]
        verified = _derive_verified_actions(results)  # type: ignore[arg-type]
        assert verified == {"web_search": {"calls": 2, "successes": 1}}

    def test_never_succeeded_tool_is_excluded(self) -> None:
        """A tool that only ever failed is dropped (no execution → no memory)."""
        results = [
            ToolResult(tool_name="broken_tool", success=False, output="", error="boom"),
            ToolResult(tool_name="broken_tool", success=False, output="", error="boom"),
        ]
        verified = _derive_verified_actions(results)  # type: ignore[arg-type]
        assert verified == {}

    def test_dict_shaped_results_handled(self) -> None:
        """Dict-shaped tool results (as _serialize_tool_history sees) work too."""
        results: list[Any] = [
            {"tool_name": "file_reader", "success": True},
            {"tool_name": "file_reader", "success": True},
        ]
        verified = _derive_verified_actions(results)
        assert verified == {"file_reader": {"calls": 2, "successes": 2}}

    def test_empty_results_yield_empty(self) -> None:
        """No tool results → nothing verified."""
        assert _derive_verified_actions([]) == {}

    def test_mixed_verified_and_unverified(self) -> None:
        """Verified tools kept, unverified-only tools dropped, in one pass."""
        results: list[Any] = [
            ToolResult(tool_name="web_search", success=True, output="r"),
            ToolResult(tool_name="flaky", success=False, output="", error="x"),
            ToolResult(tool_name="flaky", success=False, output="", error="y"),
        ]
        verified = _derive_verified_actions(results)  # type: ignore[arg-type]
        assert "web_search" in verified
        assert "flaky" not in verified


class TestCheckAndFoldPersistence:
    """WS5: _check_and_fold shrinks real messages AND persists verified actions.

    The fold path is triggered by stubbing MemoryFolder so the test doesn't
    depend on the trigger ladder or a live LLM fold call. The SUT
    (_check_and_fold) is exercised for real; only its dependencies are mocked.
    """

    @pytest.fixture
    def foldable_state(self) -> dict[str, Any]:
        """A state with real messages (ids) and mixed-success tool results."""
        from langchain_core.messages import HumanMessage

        state = dict(initial_state("goal with tools", "fold-thread", 10))
        state["messages"] = [
            HumanMessage(content="turn 1", id="msg-1"),
            HumanMessage(content="turn 2", id="msg-2"),
        ]
        state["tool_results"] = [
            ToolResult(tool_name="web_search", success=True, output="hit"),
            ToolResult(tool_name="broken", success=False, output="", error="x"),
        ]
        return state

    @pytest.mark.asyncio
    async def test_fold_shrinks_messages_and_persists_verified_actions(
        self, foldable_state: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fold emits RemoveMessage entries + a summary, and persists the tool
        summary grounded with verified_actions under the 'verified' tag."""
        from src.memory.folding import MemoryFolder, MemoryFoldResult

        # Stub the trigger + fold so the persistence/verified-actions logic runs
        # without a live LLM call or satisfying the trigger ladder.
        monkeypatch.setattr(MemoryFolder, "should_fold", lambda self, s: True)

        async def _fake_fold(self: MemoryFolder, state: dict[str, Any]) -> MemoryFoldResult:
            return MemoryFoldResult(
                episode_memory={"summary": "ep"},
                working_memory={"summary": "wk"},
                tool_memory={"summary": "tool summary"},
                fold_number=1,
            )

        monkeypatch.setattr(MemoryFolder, "fold", _fake_fold)

        gateway = MagicMock()
        memory = MagicMock()
        memory.store_skill = AsyncMock()

        result = await _check_and_fold(foldable_state, gateway, memory, {"enabled": True})  # type: ignore[arg-type]

        assert result is not None
        msgs = result["messages"]
        # Fold shrinks: one RemoveMessage per existing message, then the summary.
        from langchain_core.messages import RemoveMessage

        assert sum(1 for m in msgs if isinstance(m, RemoveMessage)) == 2
        assert len(msgs) == 3  # 2 removals + 1 summary

        # Tool summary persisted with grounded verified_actions + 'verified' tag.
        skill_calls = [
            c for c in memory.store_skill.call_args_list
            if c.kwargs.get("name") == "fold_1_tool"
        ]
        assert len(skill_calls) == 1
        payload = json.loads(skill_calls[0].kwargs["content"])
        assert payload["verified_actions"] == {"web_search": {"calls": 1, "successes": 1}}
        assert "broken" not in payload["verified_actions"]
        assert skill_calls[0].kwargs["tags"] == ["folded_memory", "tool", "verified"]

    @pytest.mark.asyncio
    async def test_fold_tags_unverified_when_no_tool_succeeded(
        self, foldable_state: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When no tool ever succeeded, the tool summary is tagged 'unverified'."""
        from src.memory.folding import MemoryFolder, MemoryFoldResult

        foldable_state["tool_results"] = [
            ToolResult(tool_name="broken", success=False, output="", error="x"),
        ]
        monkeypatch.setattr(MemoryFolder, "should_fold", lambda self, s: True)

        async def _fake_fold(self: MemoryFolder, state: dict[str, Any]) -> MemoryFoldResult:
            return MemoryFoldResult(
                episode_memory={"summary": "ep"},
                working_memory={"summary": "wk"},
                tool_memory={"summary": "tool summary"},
                fold_number=2,
            )

        monkeypatch.setattr(MemoryFolder, "fold", _fake_fold)

        gateway = MagicMock()
        memory = MagicMock()
        memory.store_skill = AsyncMock()

        result = await _check_and_fold(foldable_state, gateway, memory, {"enabled": True})  # type: ignore[arg-type]
        assert result is not None

        skill_calls = [
            c for c in memory.store_skill.call_args_list
            if c.kwargs.get("name") == "fold_2_tool"
        ]
        assert len(skill_calls) == 1
        payload = json.loads(skill_calls[0].kwargs["content"])
        assert payload["verified_actions"] == {}  # honest: nothing verified
        assert skill_calls[0].kwargs["tags"] == ["folded_memory", "tool", "unverified"]


class TestFormatAvailableTools:
    """Tests for _format_available_tools — the inventory that grounds the
    reflect LLM's missing_tools judgement (prevents re-flagging capabilities
    already satisfied by builtins/dynamic tools → endless tool_create loop)."""

    def test_lists_name_and_description(self) -> None:
        """Registered tools render as '- name: description' lines."""
        tools = MagicMock()
        tools.list_tools = MagicMock(return_value=[
            {"type": "function", "function": {
                "name": "file_writer",
                "description": "Write content to a file in the sandbox.",
            }},
            {"type": "function", "function": {
                "name": "code_executor",
                "description": "Run Python code in a sandbox.",
            }},
        ])
        out = _format_available_tools(tools)
        assert "file_writer" in out
        assert "Write content to a file" in out
        assert "code_executor" in out

    def test_none_registry_returns_placeholder(self) -> None:
        """No ToolRegistry injected → placeholder string (prompt stays valid)."""
        assert _format_available_tools(None) == "none registered"

    def test_empty_registry_returns_placeholder(self) -> None:
        tools = MagicMock()
        tools.list_tools = MagicMock(return_value=[])
        assert _format_available_tools(tools) == "none registered"

    def test_list_tools_failure_degrades_gracefully(self) -> None:
        """A registry access error never breaks reflection."""
        tools = MagicMock()
        tools.list_tools = MagicMock(side_effect=RuntimeError("registry unavailable"))
        assert _format_available_tools(tools) == "none registered"

    def test_description_first_line_only_and_capped(self) -> None:
        """Multi-line descriptions collapse to the first line, capped."""
        tools = MagicMock()
        tools.list_tools = MagicMock(return_value=[
            {"type": "function", "function": {
                "name": "web_search",
                "description": "Search the web.\nReturns links and snippets.\nMore detail.",
            }},
        ])
        out = _format_available_tools(tools)
        assert "web_search" in out
        assert "Search the web." in out
        assert "snippets" not in out  # second line dropped


class TestFormatStepResults:
    """Tests for _format_step_results — the evidence that lets the reflect LLM
    judge confidence from what each step PRODUCED, not just that it was marked
    done. Without this, a cautious model returns low confidence + should_replan
    even when every step completed (the Q9 reflect→replan loop)."""

    def test_renders_description_and_result(self) -> None:
        """A step with a result renders as 'description → result'."""
        steps = [
            PlanStep(
                id="s1",
                description="Write summary to results/q9.md",
                status=GoalStatus.COMPLETED,
                result="Wrote results/q9.md (4673 bytes)",
            ),
        ]
        out = _format_step_results(steps)
        assert "Write summary to results/q9.md" in out
        assert "→" in out
        assert "Wrote results/q9.md (4673 bytes)" in out

    def test_step_without_result_shows_description_only(self) -> None:
        """A step with no result still lists its description (no arrow)."""
        steps = [
            PlanStep(id="s1", description="Read the doc", status=GoalStatus.COMPLETED),
        ]
        out = _format_step_results(steps)
        assert "Read the doc" in out
        assert "→" not in out

    def test_empty_returns_placeholder(self) -> None:
        """No completed steps → placeholder (prompt stays valid)."""
        assert _format_step_results([]) == "None yet"
        assert _format_step_results(None) == "None yet"

    def test_result_whitespace_collapsed_and_capped(self) -> None:
        """Newlines in a result collapse to spaces; output is capped."""
        long = "line one\nline two\n" + "x" * 300
        steps = [
            PlanStep(
                id="s1",
                description="d",
                status=GoalStatus.COMPLETED,
                result=long,
            ),
        ]
        out = _format_step_results(steps)
        assert "line one line two" in out  # newlines → spaces
        assert "\nline two" not in out


class TestFormatSuccessfulTools:
    """Tests for _format_successful_tools — complements tool_errors by showing
    what SUCCEEDED (esp. file_writer paths), so reflect can confirm a
    deliverable was produced instead of replanning forever."""

    def test_lists_successful_tool_with_output(self) -> None:
        results = [
            ToolResult(
                tool_name="file_writer",
                success=True,
                output="Wrote results/q9_onboarding.md (4673 bytes)",
            ),
        ]
        out = _format_successful_tools(results)
        assert "file_writer" in out
        assert "results/q9_onboarding.md" in out

    def test_excludes_failed_tools(self) -> None:
        """Only successful tool calls are evidence; failures stay in tool_errors."""
        results = [
            ToolResult(tool_name="broken", success=False, output="", error="boom"),
            ToolResult(tool_name="file_writer", success=True, output="ok"),
        ]
        out = _format_successful_tools(results)
        assert "file_writer" in out
        assert "broken" not in out

    def test_excludes_successful_tools_with_empty_output(self) -> None:
        """A success with no output carries no evidence — skip it."""
        results = [
            ToolResult(tool_name="noop", success=True, output=""),
        ]
        assert _format_successful_tools(results) == "None"

    def test_empty_returns_placeholder(self) -> None:
        assert _format_successful_tools([]) == "None"
        assert _format_successful_tools(None) == "None"

    def test_output_whitespace_collapsed_and_capped(self) -> None:
        results = [
            ToolResult(
                tool_name="t",
                success=True,
                output="a\nb\n" + "y" * 300,
            ),
        ]
        out = _format_successful_tools(results)
        assert "a b" in out
        assert "\nb" not in out


class TestGroundShouldEvolve:
    """``_ground_should_evolve`` forces evolution on objective success. Without
    it the LLM path returned should_evolve=False even on HIGH-confidence
    deliverable successes (Q7/Q8), so evolution never fired across the battery
    (0 mutations). The helper preserves a model's own should_evolve=True via OR
    and otherwise requires no errors + ≥3 steps + the deliverable on disk."""

    def _steps(self, n: int = 3) -> list[PlanStep]:
        return [PlanStep(id=f"s{i}", description=f"Step {i}") for i in range(n)]

    def test_preserves_model_true(self) -> None:
        """If the model already said should_evolve=True, keep it."""
        assert _ground_should_evolve(
            True, "merge into results/x.md", self._steps(), [], Confidence.LOW
        ) is True

    def test_forces_true_on_deliverable_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Model=False but deliverable on disk + no errors + 3 steps → True."""
        monkeypatch.setattr("src.graph.nodes.execute._deliverable_on_disk", lambda _p: True)
        assert _ground_should_evolve(
            False, "merge the results into results/q3_overview.md", self._steps(), [], Confidence.LOW
        ) is True

    def test_false_when_deliverable_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deliverable goal but file not on disk (a failed run) → False."""
        monkeypatch.setattr("src.graph.nodes.execute._deliverable_on_disk", lambda _p: False)
        assert _ground_should_evolve(
            False, "merge into results/q3_overview.md", self._steps(), [], Confidence.HIGH
        ) is False

    def test_false_with_errors(self) -> None:
        assert _ground_should_evolve(
            False, "merge into results/x.md", self._steps(), ["boom"], Confidence.HIGH
        ) is False

    def test_false_with_fewer_than_three_steps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.graph.nodes.execute._deliverable_on_disk", lambda _p: True)
        assert _ground_should_evolve(
            False, "merge into results/x.md", self._steps(2), [], Confidence.HIGH
        ) is False

    def test_true_for_nondeliverable_high_conf(self) -> None:
        """Non-deliverable goal: HIGH confidence + 3 steps + no errors → True."""
        assert _ground_should_evolve(
            False, "Explain the system architecture.", self._steps(), [], Confidence.HIGH
        ) is True

    def test_false_for_nondeliverable_low_conf(self) -> None:
        assert _ground_should_evolve(
            False, "Explain the system architecture.", self._steps(), [], Confidence.LOW
        ) is False


