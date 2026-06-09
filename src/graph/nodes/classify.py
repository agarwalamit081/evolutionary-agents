"""Classify node — analyzes task complexity and selects strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from src.llm.gateway import LLMGateway

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


async def classify_node(
    state: AgentState,
    *,
    gateway: LLMGateway | None = None,
) -> dict[str, Any]:
    """Classify task complexity and select execution strategy.

    When a gateway is provided, uses LLM for classification.
    Falls back to keyword heuristics when gateway is unavailable.

    Args:
        state: Current agent state.
        gateway: Optional LLM gateway for LLM-enhanced classification.

    Returns:
        Partial state update with classification results.
    """
    goal = state.get("current_goal")
    if not goal or not goal.text:
        return {
            "phase": Phase.ERROR_HANDLER,
            "errors": ["classify: No goal text provided"],
        }

    goal_text = goal.text
    logger.info(f"Classifying task: {goal_text[:100]}...")

    # Try LLM classification first, fall back to heuristics
    if gateway is not None:
        result = await _llm_classify(gateway, goal_text)
        if result is not None:
            complexity, strategy, estimated_steps, confidence = result
            logger.info(
                f"LLM classification: complexity={complexity.value}, "
                f"strategy={strategy.value}, "
                f"steps={estimated_steps}, "
                f"confidence={confidence:.2f}"
            )
        else:
            complexity, strategy, estimated_steps = _heuristic_classify(goal_text.lower())
            confidence = Confidence.MEDIUM
    else:
        complexity, strategy, estimated_steps = _heuristic_classify(goal_text.lower())
        confidence = Confidence.MEDIUM

    # Build updated goal
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
        "confidence": confidence,
    }


async def _llm_classify(
    gateway: LLMGateway,
    goal_text: str,
) -> tuple[TaskComplexity, Strategy, int, Confidence] | None:
    """Attempt LLM-based classification. Returns None on failure."""
    try:
        from src.graph.prompts import CLASSIFY_SYSTEM, CLASSIFY_USER
        from src.graph.schemas import TaskClassification
        from src.llm.structured_output import StructuredOutputManager

        user_prompt = CLASSIFY_USER.format(goal_text=goal_text)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": CLASSIFY_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        response = await gateway.acompletion(
            messages=messages,
        )

        extractor = StructuredOutputManager()
        classification = await extractor.extract(response.content, TaskClassification)
        if classification is None:
            return None

        conf = Confidence.HIGH if classification.confidence >= 0.7 else Confidence.MEDIUM

        return (
            classification.complexity,
            classification.strategy,
            classification.estimated_steps,
            conf,
        )
    except Exception as e:
        logger.debug(f"LLM classification failed, using heuristics: {e}")
        return None


def _heuristic_classify(
    goal_text: str,
) -> tuple[TaskComplexity, Strategy, int]:
    """Classify using keyword heuristics."""
    complexity = _classify_complexity(goal_text)
    strategy = _select_strategy(goal_text)
    estimated_steps = _estimate_steps(complexity)
    return complexity, strategy, estimated_steps


def _classify_complexity(goal_text: str) -> TaskComplexity:
    """Determine task complexity from goal text using keyword matching."""
    for keyword in _COMPLEXITY_KEYWORDS.get(TaskComplexity.CRITICAL, []):
        if keyword in goal_text:
            return TaskComplexity.CRITICAL

    for keyword in _COMPLEXITY_KEYWORDS.get(TaskComplexity.TRIVIAL, []):
        if keyword in goal_text:
            return TaskComplexity.TRIVIAL

    complex_indicators = [
        "build", "create", "implement", "design", "develop",
        "integrate", "full-stack", "end-to-end", "comprehensive",
    ]
    for keyword in complex_indicators:
        if keyword in goal_text:
            return TaskComplexity.COMPLEX

    return TaskComplexity.SIMPLE


def _select_strategy(goal_text: str) -> Strategy:
    """Select execution strategy from goal text."""
    for strategy, keywords in _STRATEGY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in goal_text:
                return strategy

    return Strategy.REACT


def _estimate_steps(complexity: TaskComplexity) -> int:
    """Estimate number of execution steps based on complexity."""
    estimates: dict[TaskComplexity, int] = {
        TaskComplexity.TRIVIAL: 1,
        TaskComplexity.SIMPLE: 3,
        TaskComplexity.COMPLEX: 7,
        TaskComplexity.CRITICAL: 12,
    }
    return estimates.get(complexity, 5)
