"""Tests for src.graph.nodes.classify — classify node function."""

from __future__ import annotations


import pytest
from unittest.mock import AsyncMock, MagicMock

from src.graph.enums import (
    Confidence,
    GoalStatus,
    Phase,
    Strategy,
    TaskComplexity,
)
from src.graph.factory import initial_state
from src.graph.nodes.classify import classify_node
from src.llm.models import LLMResponse


class TestClassifyNode:
    """Tests for the classify_node async function."""

    @pytest.mark.asyncio
    async def test_classify_trivial_task(self) -> None:
        """Tasks with 'explain' keyword are classified as TRIVIAL."""
        state = initial_state(
            goal_text="explain quicksort",
            thread_id="thread-trivial",
        )
        result = await classify_node(state)

        assert result["phase"] == Phase.PLAN
        goal = result["current_goal"]
        assert goal.complexity == TaskComplexity.TRIVIAL

    @pytest.mark.asyncio
    async def test_classify_critical_task(self) -> None:
        """Tasks with 'deploy' keyword are classified as CRITICAL."""
        state = initial_state(
            goal_text="deploy to production",
            thread_id="thread-critical",
        )
        result = await classify_node(state)

        assert result["phase"] == Phase.PLAN
        goal = result["current_goal"]
        assert goal.complexity == TaskComplexity.CRITICAL

    @pytest.mark.asyncio
    async def test_classify_simple_task(self) -> None:
        """Unknown tasks without keyword matches default to SIMPLE."""
        state = initial_state(
            goal_text="write a function to compute fibonacci",
            thread_id="thread-simple",
        )
        result = await classify_node(state)

        assert result["phase"] == Phase.PLAN
        goal = result["current_goal"]
        assert goal.complexity == TaskComplexity.SIMPLE

    @pytest.mark.asyncio
    async def test_classify_no_goal_returns_error(self) -> None:
        """Missing goal text routes to ERROR_HANDLER phase."""
        state = initial_state(
            goal_text="some goal",
            thread_id="thread-nogoal",
        )
        # Simulate a missing goal by clearing it
        state["current_goal"] = None

        result = await classify_node(state)
        assert result["phase"] == Phase.ERROR_HANDLER
        assert len(result["errors"]) > 0

    @pytest.mark.asyncio
    async def test_classify_whitespace_goal_text_is_treated_as_valid(self) -> None:
        """Whitespace-only goal text is truthy, so classify proceeds with heuristics."""
        state = initial_state(
            goal_text="   ",
            thread_id="thread-whitespace",
        )

        result = await classify_node(state)
        # Whitespace is truthy in Python, so classify treats it as a valid goal
        # and proceeds to PLAN with heuristic classification (SIMPLE by default).
        assert result["phase"] == Phase.PLAN
        goal = result["current_goal"]
        assert goal.complexity == TaskComplexity.SIMPLE

    @pytest.mark.asyncio
    async def test_classify_empty_string_goal_text_returns_error(self) -> None:
        """Empty string goal text (falsy) routes to ERROR_HANDLER phase."""
        state = initial_state(
            goal_text="some goal",
            thread_id="thread-empty",
        )
        from src.graph.models import Goal
        state["current_goal"] = Goal(text="")

        result = await classify_node(state)
        assert result["phase"] == Phase.ERROR_HANDLER

    @pytest.mark.asyncio
    async def test_classify_sets_goal_status_active(self) -> None:
        """classify_node sets the goal status to ACTIVE."""
        state = initial_state(
            goal_text="explain quicksort",
            thread_id="thread-status",
        )
        result = await classify_node(state)
        assert result["current_goal"].status == GoalStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_classify_returns_strategy(self) -> None:
        """classify_node returns a strategy field."""
        state = initial_state(
            goal_text="search for relevant papers",
            thread_id="thread-strategy",
        )
        result = await classify_node(state)
        assert "strategy" in result
        assert isinstance(result["strategy"], Strategy)

    @pytest.mark.asyncio
    async def test_classify_returns_confidence(self) -> None:
        """classify_node returns a confidence level."""
        state = initial_state(
            goal_text="explain quicksort",
            thread_id="thread-conf",
        )
        result = await classify_node(state)
        assert "confidence" in result
        assert isinstance(result["confidence"], Confidence)


def _mock_gateway(content: str) -> MagicMock:
    """Build a gateway mock whose acompletion returns the given JSON string."""
    gw = MagicMock()
    gw.acompletion = AsyncMock(return_value=LLMResponse(
        content=content,
        model="gpt-4o-mini-2024-07-18",
        provider="openai",
        input_tokens=10,
        output_tokens=50,
        total_tokens=60,
        cost_usd=0.0001,
    ))
    return gw


class TestIntentAndAmbiguity:
    """Feature A — classify surfaces refined_intent + ambiguity assessment.

    The literal goal text stays the OBJECTIVE (never rewritten); refined_intent
    is advisory only and feeds the downstream disambiguate cascade (Feature B).
    """

    @pytest.mark.asyncio
    async def test_llm_path_surfaces_intent_and_ambiguity(self) -> None:
        classify_json = (
            '{"complexity": "complex", "strategy": "planning", '
            '"estimated_steps": 5, "confidence": 0.85, "reasoning": "multi-step", '
            '"refined_intent": "a dependency-ordered build plan", '
            '"ambiguity_type": "referential", "ambiguity_severity": 0.6, '
            '"ambiguity_notes": ["\'this\' has no referent", '
            '"no output format given"]}'
        )
        state = initial_state(
            goal_text="summarise this and build it",
            thread_id="thread-intent",
        )
        result = await classify_node(state, gateway=_mock_gateway(classify_json))

        assert result["phase"] == Phase.PLAN
        assert result["refined_intent"] == "a dependency-ordered build plan"
        assert result["ambiguity_type"] == "referential"
        assert result["ambiguity_severity"] == 0.6
        assert result["ambiguity_notes"] == ["'this' has no referent", "no output format given"]

    @pytest.mark.asyncio
    async def test_legacy_5_field_json_still_parses(self) -> None:
        """Additive fields default when an older LLM omits them (backcompat)."""
        classify_json = (
            '{"complexity": "simple", "strategy": "direct", '
            '"estimated_steps": 2, "confidence": 0.9, "reasoning": "trivial"}'
        )
        state = initial_state("explain quicksort", "thread-legacy")
        result = await classify_node(state, gateway=_mock_gateway(classify_json))

        assert result["phase"] == Phase.PLAN
        # Defaults applied — no KeyError, no None, advisory-safe.
        assert result["refined_intent"] == ""
        assert result["ambiguity_type"] == "none"
        assert result["ambiguity_severity"] == 0.0
        assert result["ambiguity_notes"] == []

    @pytest.mark.asyncio
    async def test_refined_intent_never_overwrites_literal_goal(self) -> None:
        """Drift guard: the OBJECTIVE (current_goal.text) stays literal even
        when the LLM proposes a restated refined_intent."""
        goal_text = "fix it"
        classify_json = (
            '{"complexity": "simple", "strategy": "react", "estimated_steps": 3, '
            '"confidence": 0.5, "reasoning": "pronoun", '
            '"refined_intent": "repair the broken login flow", '
            '"ambiguity_type": "referential", "ambiguity_severity": 0.8, '
            '"ambiguity_notes": ["\'it\' ambiguous"]}'
        )
        state = initial_state(goal_text, "thread-drift")
        result = await classify_node(state, gateway=_mock_gateway(classify_json))

        # The advisory restatement is captured...
        assert result["refined_intent"] == "repair the broken login flow"
        # ...but the OBJECTIVE is unchanged.
        assert result["current_goal"].text == goal_text

    @pytest.mark.asyncio
    async def test_heuristic_path_uses_safe_ambiguity_defaults(self) -> None:
        """No gateway → heuristic path must still emit advisory-safe defaults."""
        state = initial_state("explain quicksort", "thread-heur")
        result = await classify_node(state)

        assert result["refined_intent"] == ""
        assert result["ambiguity_type"] == "none"
        assert result["ambiguity_severity"] == 0.0
        assert result["ambiguity_notes"] == []
