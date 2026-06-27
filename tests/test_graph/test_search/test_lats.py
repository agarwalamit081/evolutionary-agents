"""Unit tests for src.graph.search.lats — LATS/MCTS tree-search primitive (G3a).

Mocked-gateway, deterministic: proves LATS (1) picks the higher-scored branch,
(2) honors the evaluation budget, (3) fails safe, (4) is byte-identical when off,
and (5) engages only under the documented guard. No real LLM calls.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config.settings import get_settings
from src.graph.enums import Confidence, GoalStatus, TaskComplexity
from src.graph.models import Goal, PlanStep
from src.graph.routers import route_after_reflect
from src.graph.search.lats import _lats_should_engage, lats_search_node
from src.graph.task_graph import build_task_graph
from src.llm.models import LLMResponse

# ── Helpers ────────────────────────────────────────────────────────────────────


def _resp(content: str) -> LLMResponse:
    """Build a minimal LLMResponse carrying ``content`` (the gateway idiom)."""
    return LLMResponse(
        content=content,
        model="test-model",
        provider="test",
        input_tokens=1,
        output_tokens=1,
        total_tokens=2,
        cost_usd=0.0,
    )


def _expand_resp(candidates: list[dict[str, Any]]) -> LLMResponse:
    return _resp(json.dumps({"candidates": candidates}))


def _value_resp(score: float) -> LLMResponse:
    return _resp(json.dumps({"score": score, "rationale": "test"}))


def _critical_low_state(idx: int = 0, n_steps: int = 1) -> dict[str, Any]:
    """An engaging LATS state: CRITICAL + LOW confidence + a remaining step."""
    goal = Goal(
        text="solve a hard multi-constraint optimization under uncertainty",
        complexity=TaskComplexity.CRITICAL,
    )
    steps = [
        PlanStep(
            id=f"inc{i}",
            description=f"incumbent step {i}",
            tool_name="code_executor",
            tool_input={},
            expected_output="result",
            status=GoalStatus.PENDING,
        )
        for i in range(n_steps)
    ]
    return {
        "current_goal": goal,
        "submitted_goal": goal.text,
        "confidence": Confidence.LOW,
        "plan_steps": steps,
        "current_step_index": idx,
        "iteration_count": 1,
        "errors": [],
        "pending_agent_gaps": [],
        "pending_tool_gaps": [],
        "consecutive_cap_blocks": 0,
    }


def _router_state(confidence: Confidence) -> dict[str, Any]:
    """Engaging state for route_after_reflect (reflection present, not replan)."""
    state = _critical_low_state()
    state["confidence"] = confidence
    # A non-None reflection with should_replan=False reaches the confidence check.
    state["reflection"] = SimpleNamespace(should_replan=False)
    return state


def _mock_tools() -> MagicMock:
    tools = MagicMock()
    tools.list_names = MagicMock(return_value=["code_executor", "web_search"])
    return tools


@pytest.fixture
def lats_enabled(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Enable LATS with small, deterministic caps on the cached settings singleton."""
    lats = get_settings().lats
    monkeypatch.setattr(lats, "enabled", True)
    monkeypatch.setattr(lats, "scope", "stall")
    monkeypatch.setattr(lats, "max_expansions", 2)
    monkeypatch.setattr(lats, "rollout_depth", 0)  # flat 1-ply value pick
    monkeypatch.setattr(lats, "max_evaluations", 3)  # exactly one pass over 3 children
    monkeypatch.setattr(lats, "max_depth", 2)
    monkeypatch.setattr(lats, "exploration", 1.41)
    return lats


# ── 1. Picks the higher-scored branch ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_picks_higher_scored_branch(lats_enabled: Any) -> None:
    """Expansion proposes A/B; the value function scores A highest ⇒ commit A."""
    gateway = MagicMock()
    gateway.acompletion = AsyncMock(
        side_effect=[
            _expand_resp(
                [
                    {"description": "alt-A", "tool_name": "code_executor", "tool_input": {}, "expected_output": "a"},
                    {"description": "alt-B", "tool_name": "code_executor", "tool_input": {}, "expected_output": "b"},
                ]
            ),
            _value_resp(0.5),  # incumbent (child[0]) — "stay the course"
            _value_resp(0.8),  # alt-A
            _value_resp(0.3),  # alt-B
        ]
    )
    state = _critical_low_state()

    result = await lats_search_node(state, gateway=gateway, tools=_mock_tools())

    assert "plan_steps" in result, "LATS should commit a non-incumbent winner"
    revised = result["plan_steps"]
    assert len(revised) == 1, "commit swaps in place — plan length unchanged"
    assert revised[0].description == "alt-A", "the highest-scored branch wins"
    assert revised[0].id == "inc0", "the incumbent id is preserved (dependents resolve)"


# ── 2. Budget caps honored ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_budget_caps_honored(lats_enabled: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A tiny max_evaluations stops the search early; incumbent wins, never raises."""
    monkeypatch.setattr(lats_enabled, "max_evaluations", 1)  # only the incumbent is valued
    gateway = MagicMock()
    gateway.acompletion = AsyncMock(
        side_effect=[
            _expand_resp(
                [
                    {"description": "alt-A", "tool_name": "code_executor", "tool_input": {}, "expected_output": "a"},
                    {"description": "alt-B", "tool_name": "code_executor", "tool_input": {}, "expected_output": "b"},
                ]
            ),
            _value_resp(0.6),  # incumbent — beats the neutral 0.5 of unvalued alts
        ]
    )
    state = _critical_low_state()

    result = await lats_search_node(state, gateway=gateway, tools=_mock_tools())

    assert result == {}, "with the incumbent winning, LATS stays the course (no-op)"
    # Only 1 expand + 1 value call — the budget cap stopped further evaluation.
    assert gateway.acompletion.await_count == 2


# ── 3. Fail safe ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fail_safe_returns_unchanged(lats_enabled: Any) -> None:
    """Any gateway error ⇒ state unchanged (incumbent step runs)."""
    gateway = MagicMock()
    gateway.acompletion = AsyncMock(side_effect=RuntimeError("gateway exploded"))
    state = _critical_low_state()
    original_desc = state["plan_steps"][0].description

    result = await lats_search_node(state, gateway=gateway, tools=_mock_tools())

    assert result == {}, "a failure must be a transparent pass-through"
    assert state["plan_steps"][0].description == original_desc, "state never mutated in place"


@pytest.mark.asyncio
async def test_fail_safe_on_garbage_value(lats_enabled: Any) -> None:
    """Unparseable value JSON ⇒ neutral score, search completes, never raises."""
    gateway = MagicMock()
    gateway.acompletion = AsyncMock(
        side_effect=[
            _expand_resp([{"description": "alt-A", "tool_name": "code_executor", "tool_input": {}, "expected_output": "a"}]),
            _resp("this is not json at all"),  # incumbent → neutral 0.5
            _resp("}}}garbage{{{"),  # alt-A → neutral 0.5
        ]
    )
    state = _critical_low_state()

    result = await lats_search_node(state, gateway=gateway, tools=_mock_tools())

    # Tie at 0.5 ⇒ first child (incumbent) wins ⇒ stay the course.
    assert result == {}


# ── 4. Off ⇒ byte-identical ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_off_is_byte_identical() -> None:
    """With LATS disabled (default), the node is a no-op and routing is unchanged."""
    lats = get_settings().lats
    assert lats.enabled is False, "LATS must ship default-off"

    # (a) The node returns {} for an otherwise-engaging state.
    gateway = MagicMock()
    gateway.acompletion = AsyncMock(return_value=_resp("{}"))
    result = await lats_search_node(_critical_low_state(), gateway=gateway, tools=_mock_tools())
    assert result == {}
    assert gateway.acompletion.await_count == 0, "disabled ⇒ zero gateway calls"

    # (b) route_after_reflect never returns "lats_search" when disabled.
    assert route_after_reflect(_router_state(Confidence.LOW)) == "execute"

    # (c) The lats_search node + its edge target exist in the built graph.
    graph = build_task_graph()
    assert "lats_search" in set(graph.nodes.keys())


# ── 5. Engage guard ────────────────────────────────────────────────────────────


def test_engage_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """_lats_should_engage fires only for CRITICAL + stalled + remaining steps."""
    lats = get_settings().lats

    def _configure(**overrides: Any) -> None:
        defaults: dict[str, Any] = {"enabled": True, "scope": "stall"}
        defaults.update(overrides)
        for key, value in defaults.items():
            monkeypatch.setattr(lats, key, value)

    # Positive: CRITICAL + LOW + remaining step.
    _configure()
    assert _lats_should_engage(_critical_low_state()) is True

    # Confidence gate (stall scope): HIGH confidence does not engage.
    high = _critical_low_state()
    high["confidence"] = Confidence.HIGH
    _configure()
    assert _lats_should_engage(high) is False

    # scope=always engages on HIGH confidence too (CRITICAL only required).
    _configure(scope="always")
    assert _lats_should_engage(high) is True

    # Non-CRITICAL never engages.
    simple = _critical_low_state()
    simple["current_goal"] = Goal(text="easy goal", complexity=TaskComplexity.SIMPLE)
    _configure()
    assert _lats_should_engage(simple) is False

    # No remaining step never engages.
    exhausted = _critical_low_state(idx=2, n_steps=2)
    _configure()
    assert _lats_should_engage(exhausted) is False

    # Disabled never engages.
    _configure(enabled=False)
    assert _lats_should_engage(_critical_low_state()) is False
