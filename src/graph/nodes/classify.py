"""Classify node — analyzes task complexity and selects strategy."""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.graph.enums import (
    Confidence,
    GoalStatus,
    Phase,
    Strategy,
    TaskComplexity,
)
from src.graph.models import Goal
from src.graph.state import AgentState

# Heuristic keyword-based classification for fast path
_COMPLEXITY_KEYWORDS: dict[TaskComplexity, list[str]] = {
    TaskComplexity.TRIVIAL: [
        "define", "what is", "explain", "list", "convert", "format",
        "summarize briefly", "label", "classify", "translate",
    ],
    TaskComplexity.CRITICAL: [
        "production", "deploy", "critical", "security", "audit",
        "migrate", "refactor entire", "redesign", "rewrite",
    ],
}

_STRATEGY_KEYWORDS: dict[Strategy, list[str]] = {
    Strategy.REACT: ["search", "find", "look up", "investigate", "explore", "analyze data"],
    Strategy.PLANNING: ["plan", "step by step", "multi-step", "roadmap", "sequence"],
    Strategy.REFLECTION: ["review", "critique", "improve", "optimize", "refine"],
    Strategy.TOT: ["compare", "evaluate options", "best approach", "alternatives"],
    Strategy.DEBATE: ["argue", "pros and cons", "debate", "from multiple perspectives"],
}


async def classify_node(state: AgentState) -> dict[str, Any]:
    """Classify task complexity and select execution strategy.

    Uses keyword heuristics for fast classification. For ambiguous cases,
    falls back to LLM-based classification via the gateway.

    Args:
        state: Current agent state.

    Returns:
        Partial state update with classification results.
    """
    goal = state.get("current_goal")
    if not goal or not goal.text:
        return {
            "phase": Phase.ERROR_HANDLER,
            "errors": ["classify: No goal text provided"],
        }

    goal_text = goal.text.lower()
    logger.info(f"Classifying task: {goal_text[:100]}...")

    # ─── Classify Complexity ────────────────────────────────────────────
    complexity = _classify_complexity(goal_text)

    # ─── Select Strategy ────────────────────────────────────────────────
    strategy = _select_strategy(goal_text)

    # ─── Estimate Steps ─────────────────────────────────────────────────
    estimated_steps = _estimate_steps(complexity)

    # ─── Build Updated Goal ─────────────────────────────────────────────
    updated_goal = Goal(
        id=goal.id,
        text=goal.text,
        priority=goal.priority,
        status=GoalStatus.ACTIVE,
        complexity=complexity,
        parent_goal_id=goal.parent_goal_id,
        sub_goals=goal.sub_goals,
        success_criteria=goal.success_criteria,
    )

    logger.info(
        f"Classification: complexity={complexity.value}, "
        f"strategy={strategy.value}, "
        f"estimated_steps={estimated_steps}"
    )

    return {
        "phase": Phase.PLAN,
        "current_goal": updated_goal,
        "strategy": strategy,
        "confidence": Confidence.MEDIUM,
    }


def _classify_complexity(goal_text: str) -> TaskComplexity:
    """Determine task complexity from goal text using keyword matching."""
    # Check critical first (highest priority)
    for keyword in _COMPLEXITY_KEYWORDS.get(TaskComplexity.CRITICAL, []):
        if keyword in goal_text:
            return TaskComplexity.CRITICAL

    # Check trivial
    for keyword in _COMPLEXITY_KEYWORDS.get(TaskComplexity.TRIVIAL, []):
        if keyword in goal_text:
            return TaskComplexity.TRIVIAL

    # Check for complex indicators
    complex_indicators = [
        "build", "create", "implement", "design", "develop",
        "integrate", "full-stack", "end-to-end", "comprehensive",
    ]
    for keyword in complex_indicators:
        if keyword in goal_text:
            return TaskComplexity.COMPLEX

    # Default to simple
    return TaskComplexity.SIMPLE


def _select_strategy(goal_text: str) -> Strategy:
    """Select execution strategy from goal text."""
    for strategy, keywords in _STRATEGY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in goal_text:
                return strategy

    return Strategy.REACT  # Default: ReAct is the most versatile


def _estimate_steps(complexity: TaskComplexity) -> int:
    """Estimate number of execution steps based on complexity."""
    estimates: dict[TaskComplexity, int] = {
        TaskComplexity.TRIVIAL: 1,
        TaskComplexity.SIMPLE: 3,
        TaskComplexity.COMPLEX: 7,
        TaskComplexity.CRITICAL: 12,
    }
    return estimates.get(complexity, 5)
