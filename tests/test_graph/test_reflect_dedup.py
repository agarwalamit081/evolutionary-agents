"""Tests for gap deduplication across reflect cycles."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.graph.enums import Confidence
from src.graph.models import Goal, GoalStatus, PlanStep, ReflectionResult
from src.graph.nodes.reflect import (
    _detect_agent_gaps_heuristic,
    _ground_should_evolve,
    _heuristic_reflect,
    _llm_reflect,
)


def _make_state(
    pending_tool_gaps: list[str] | None = None,
    pending_agent_gaps: list[str] | None = None,
    attempted_tool_gaps: list[str] | None = None,
    sub_agents_spawned: list[dict[str, Any]] | None = None,
    tool_results: list[Any] | None = None,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    """Create a minimal state dict for testing."""
    return {
        "current_goal": Goal(text="test goal", status=GoalStatus.ACTIVE),
        "plan_steps": [PlanStep(description=f"step {i}") for i in range(3)],
        "completed_steps": [PlanStep(description="step 0")],
        "errors": errors or [],
        "iteration_count": 1,
        "max_iterations": 25,
        "pending_tool_gaps": pending_tool_gaps or [],
        "attempted_tool_gaps": attempted_tool_gaps or [],
        "pending_agent_gaps": pending_agent_gaps or [],
        "sub_agents_spawned": sub_agents_spawned or [],
        "tool_results": tool_results or [],
    }


class MockToolResult:
    """Mock ToolResult for testing."""

    def __init__(self, name: str, error: str = "") -> None:
        self.tool_name = name
        self.success = False
        self.output = ""
        self.error = error


class TestToolGapDedup:
    """Verify pending_tool_gaps deduplicates against existing state."""

    def test_no_duplicate_tool_gaps(self) -> None:
        """If gap already attempted, heuristic should not add it again."""
        state = _make_state(
            attempted_tool_gaps=["tool matching 'web_scraper' capability"],
            tool_results=[MockToolResult("web_scraper", "Unknown tool: web_scraper")],
        )
        result = _heuristic_reflect(
            state,  # type: ignore[arg-type]
            "test goal",
            state["completed_steps"],  # type: ignore[arg-type]
            [],
            None,
        )
        tool_gaps = result.get("pending_tool_gaps", [])
        # Same gap should not appear again
        assert tool_gaps == []

    def test_new_tool_gap_is_returned(self) -> None:
        """A genuinely new gap should be returned."""
        from src.tools.registry import ToolRegistry

        registry = ToolRegistry()  # Empty — json_parser is truly missing

        state = _make_state(
            attempted_tool_gaps=["tool matching 'web_scraper' capability"],
            tool_results=[MockToolResult("json_parser", "Unknown tool: json_parser")],
        )
        result = _heuristic_reflect(
            state,  # type: ignore[arg-type]
            "test goal",
            state["completed_steps"],  # type: ignore[arg-type]
            [],
            registry,
        )
        tool_gaps = result.get("pending_tool_gaps", [])
        assert len(tool_gaps) == 1
        assert "json_parser" in tool_gaps[0]


class TestAgentGapDedup:
    """Verify pending_agent_gaps deduplicates against existing state."""

    def test_no_duplicate_agent_gaps_when_already_spawned(self) -> None:
        """If sub-agents already spawned, heuristic should not add more gaps."""
        state = _make_state(
            sub_agents_spawned=[{"name": "data_analyst"}],
            pending_agent_gaps=["multi-part task with 7 steps"],
        )
        gaps = _detect_agent_gaps_heuristic(
            state,  # type: ignore[arg-type]
            "test goal",
            state["plan_steps"],  # type: ignore[arg-type]
        )
        assert gaps == []

    def test_no_duplicate_agent_gaps_in_state(self) -> None:
        """If agent gap already in state, heuristic should not re-add it."""
        state = _make_state(
            pending_agent_gaps=[
                "multi-part task with 7 steps — specialized sub-agents"
            ],
        )
        result = _heuristic_reflect(
            state,  # type: ignore[arg-type]
            "analyze data and write report and visualize",
            state["completed_steps"],  # type: ignore[arg-type]
            [],
            None,
        )
        agent_gaps = result.get("pending_agent_gaps", [])
        # The existing gap should not be duplicated
        existing = state.get("pending_agent_gaps", [])
        for gap in agent_gaps:
            assert gap not in existing


class TestRouterGuard:
    """Verify route_after_reflect skips agent_spawn when already spawned."""

    def test_skips_agent_spawn_when_spawned(self) -> None:
        """If sub_agents_spawned is set, should not route to agent_spawn."""
        from src.graph.routers import route_after_reflect

        state = {
            "pending_agent_gaps": ["some gap"],
            "pending_tool_gaps": [],
            "sub_agents_spawned": [{"name": "analyst"}],
            "reflection": ReflectionResult(
                summary="test",
                lessons_learned=[],
                confidence=Confidence.HIGH,
                should_evolve=False,
                should_replan=False,
                memory_observations=[],
                cost_efficiency=1.0,
            ),
            "confidence": Confidence.HIGH,
        }
        result = route_after_reflect(state)  # type: ignore[arg-type]
        assert result != "agent_spawn"

    def test_routes_to_agent_spawn_when_not_spawned(self) -> None:
        """If gaps exist and no agents spawned, should route to agent_spawn."""
        from src.graph.routers import route_after_reflect

        state = {
            "pending_agent_gaps": ["some gap"],
            "pending_tool_gaps": [],
            "sub_agents_spawned": [],
            "reflection": None,
            "confidence": Confidence.MEDIUM,
        }
        result = route_after_reflect(state)  # type: ignore[arg-type]
        assert result == "agent_spawn"


class TestAgentGapAttemptedDedupLLM:
    """Bug C (battery-02 N6): the LLM reflect path re-detected the same
    sub-agent gap every cycle because it deduped against ``pending_agent_gaps``
    (cleared each cycle by agent_spawn) instead of ``attempted_agent_gaps``. A
    failed spawn then re-converted the gap to a tool gap, re-firing
    tool_create — 19 node entries / ~56 generations / 764s. The fix dedups the
    LLM-emitted ``missing_sub_agents`` against ``attempted_agent_gaps`` so a gap
    whose spawn already failed is never re-flagged.
    """

    @staticmethod
    def _gateway_returning(missing_sub_agents: list[str]) -> MagicMock:
        """Gateway whose acompletion yields a ReflectionAnalysis JSON."""
        from src.llm.models import LLMResponse

        content = json.dumps(
            {
                "progress_assessment": "incomplete",
                "confidence": 0.2,
                "should_replan": True,
                "should_evolve": False,
                "lessons_learned": [],
                "memory_observations": [],
                "next_action": "replan",
                "missing_tools": [],
                "missing_sub_agents": missing_sub_agents,
            }
        )
        gateway = MagicMock()
        gateway.acompletion = AsyncMock(
            return_value=LLMResponse(
                content=content,
                model="claude-haiku-4-5-20251001",
                provider="anthropic",
                input_tokens=10,
                output_tokens=20,
                total_tokens=30,
                cost_usd=0.0001,
            )
        )
        return gateway

    @pytest.mark.asyncio
    async def test_attempted_agent_gap_not_re_emitted(self) -> None:
        """Gap already in attempted_agent_gaps must not return to pending."""
        gap = "Log analysis specialist: would handle log parsing and dedup"
        state = _make_state()
        state["attempted_agent_gaps"] = [gap]  # spawn already failed for it

        result = await _llm_reflect(
            self._gateway_returning([gap]), state, None  # type: ignore[arg-type]
        )

        assert result is not None
        assert result.get("pending_agent_gaps", []) == []

    @pytest.mark.asyncio
    async def test_fresh_agent_gap_is_emitted(self) -> None:
        """Positive control: a gap NOT in attempted is still propagated."""
        gap = "doc_outline: reads a markdown doc and emits its section outline"
        state = _make_state()  # attempted_agent_gaps empty

        result = await _llm_reflect(
            self._gateway_returning([gap]), state, None  # type: ignore[arg-type]
        )

        assert result is not None
        assert result.get("pending_agent_gaps", []) == [gap]


class TestGroundShouldEvolve:
    """Cover the evolution-grounding gate (battery-02 N8 root cause).

    N8 root cause: a delegate-style run produces the deliverable via a sub-agent
    (``repo_map_builder``), leaving the main graph's ``completed_steps`` empty,
    so the old ``len(completed_steps) < 3`` guard suppressed evolution on a run
    verify had marked complete. The deliverable-on-disk check must now fire
    regardless of step count (errors still block; non-deliverable goals keep the
    step-count + confidence bar).
    """

    @staticmethod
    def _patch_deliverables(
        monkeypatch: pytest.MonkeyPatch,
        *,
        path: str | None,
        on_disk: bool,
    ) -> None:
        """Stub the lazily-imported execute helpers so tests are hermetic.

        ``_ground_should_evolve`` imports ``_extract_goal_deliverable`` /
        ``_deliverable_on_disk`` from ``execute`` at call time, so patching the
        module attribute is seen on the next call.
        """
        import src.graph.nodes.execute as execute_mod

        monkeypatch.setattr(execute_mod, "_extract_goal_deliverable", lambda _goal: path)
        monkeypatch.setattr(execute_mod, "_deliverable_on_disk", lambda _p: on_disk)

    def test_proposed_true_short_circuits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A model-proposed should_evolve wins even with errors + no steps."""
        self._patch_deliverables(monkeypatch, path=None, on_disk=False)
        assert _ground_should_evolve(
            True, "any goal", [], ["boom"], Confidence.LOW
        ) is True

    def test_errors_block_even_with_deliverable_on_disk(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Errors are the one hard block, even when the artifact exists."""
        self._patch_deliverables(monkeypatch, path="results/x.md", on_disk=True)
        assert _ground_should_evolve(
            False, "save to results/x.md", [], ["boom"], Confidence.HIGH
        ) is False

    def test_deliverable_on_disk_fires_with_zero_steps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The N8 case: delegate did all work (main steps empty), artifact on disk."""
        self._patch_deliverables(
            monkeypatch, path="results/n8_repomap.md", on_disk=True
        )
        assert _ground_should_evolve(
            False,
            "save the comparison to results/n8_repomap.md",
            [],  # zero main-graph steps
            [],
            Confidence.MEDIUM,  # confidence irrelevant once deliverable is on disk
        ) is True

    def test_deliverable_missing_blocks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Deliverable goal whose artifact is NOT on disk stays un-evolved."""
        self._patch_deliverables(monkeypatch, path="results/x.md", on_disk=False)
        assert _ground_should_evolve(
            False, "save to results/x.md", [], [], Confidence.HIGH
        ) is False

    def test_non_deliverable_blocks_on_few_steps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-deliverable goal still needs >=3 steps (no artifact to confirm)."""
        self._patch_deliverables(monkeypatch, path=None, on_disk=False)
        assert _ground_should_evolve(
            False, "explain quicksort", ["s1", "s2"], [], Confidence.HIGH
        ) is False

    def test_non_deliverable_blocks_on_low_confidence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-deliverable goal with many steps but low confidence stays False."""
        self._patch_deliverables(monkeypatch, path=None, on_disk=False)
        assert _ground_should_evolve(
            False, "explain quicksort", ["s1", "s2", "s3", "s4"], [], Confidence.LOW
        ) is False

    def test_non_deliverable_fires_high_confidence_many_steps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-deliverable goal fires only at >=3 steps AND HIGH+ confidence."""
        self._patch_deliverables(monkeypatch, path=None, on_disk=False)
        assert _ground_should_evolve(
            False, "explain quicksort", ["s1", "s2", "s3", "s4"], [], Confidence.HIGH
        ) is True
