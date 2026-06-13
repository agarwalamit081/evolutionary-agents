"""Tests for src.graph.nodes.reflect — reflect node function."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.graph.enums import Confidence, Phase
from src.graph.factory import initial_state
from src.graph.models import PlanStep, ReflectionResult, ToolResult
from src.graph.nodes.reflect import _check_and_fold, _derive_verified_actions, reflect_node


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
