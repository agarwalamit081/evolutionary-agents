"""Classify node — analyzes task complexity and selects strategy."""

from __future__ import annotations

import re
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

    from src.graph.schemas import TaskClassification

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

# P2 — deterministic complexity floor. The LLM classifier can under-rate a
# multi-deliverable / verification goal as SIMPLE because it keys off step
# count ("complex = 6-12 steps" per the prompt). A goal that asks for several
# distinct artifacts OR an explicit recomputation/verification step is
# objectively COMPLEX regardless of step count; under-rating it routes the run
# to a SIMPLE-tier model (deepseek-v4-flash) that then struggles and burns
# tokens on the cascade. These signals promote TRIVIAL/SIMPLE → COMPLEX.
_FLOOR_VERIFY_KEYWORDS: tuple[str, ...] = (
    "verify", "recompute", "re-compute", "cross-check", "cross check",
    "re-derive", "rederive", "validate that", "assert that", "check that",
    "independently confirm",
)

# File extensions that signal a distinct deliverable artifact (not a decimal
# number or version string). ≥3 distinct ones ⟹ multi-artifact ⟹ COMPLEX.
_FLOOR_ARTIFACT_EXTS: frozenset[str] = frozenset({
    "csv", "tsv", "json", "jsonl", "xml", "yaml", "yml", "md", "rst", "txt",
    "py", "js", "ts", "rs", "go", "java", "sql", "html", "htm", "pdf",
    "xlsx", "xls", "png", "jpg", "jpeg", "svg", "parquet", "db", "sqlite",
})

_FLOOR_EXT_RE = re.compile(r"\.([a-z0-9]{2,4})\b")


def _apply_complexity_floor(
    goal_text: str, complexity: TaskComplexity
) -> TaskComplexity:
    """Promote an under-rated complexity to COMPLEX on objective structural
    signals, so multi-deliverable / verification goals never get a SIMPLE-tier
    model. Never demotes; only TRIVIAL/SIMPLE are candidates for promotion.
    """
    if complexity not in (TaskComplexity.TRIVIAL, TaskComplexity.SIMPLE):
        return complexity
    lowered = goal_text.lower()
    # (a) explicit verification / recomputation requirement.
    if any(kw in lowered for kw in _FLOOR_VERIFY_KEYWORDS):
        return TaskComplexity.COMPLEX
    # (b) ≥3 distinct output file extensions → multi-artifact deliverable.
    exts = {
        m.group(1)
        for m in _FLOOR_EXT_RE.finditer(lowered)
        if m.group(1) in _FLOOR_ARTIFACT_EXTS
    }
    if len(exts) >= 3:
        return TaskComplexity.COMPLEX
    return complexity


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

    # Advisory metadata (Feature A). Defaults for the heuristic path; the LLM
    # path overwrites them from the parsed TaskClassification model. The
    # literal goal text is NEVER replaced — refined_intent only carries an
    # advisory restatement, consumed downstream by the disambiguate cascade.
    refined_intent = ""
    ambiguity_type = "none"
    ambiguity_severity = 0.0
    ambiguity_notes: list[str] = []

    # Try LLM classification first, fall back to heuristics
    classification: TaskClassification | None = None
    if gateway is not None:
        classification = await _llm_classify(gateway, goal_text)

    if classification is not None:
        complexity = classification.complexity
        strategy = classification.strategy
        estimated_steps = classification.estimated_steps
        confidence = (
            Confidence.HIGH if classification.confidence >= 0.7 else Confidence.MEDIUM
        )
        refined_intent = classification.refined_intent
        ambiguity_type = classification.ambiguity_type
        ambiguity_severity = classification.ambiguity_severity
        ambiguity_notes = list(classification.ambiguity_notes)
        logger.info(
            f"LLM classification: complexity={complexity.value}, "
            f"strategy={strategy.value}, "
            f"steps={estimated_steps}, "
            f"confidence={confidence.value}, "
            f"ambiguity={ambiguity_type}@{ambiguity_severity:.2f}"
        )
    else:
        complexity, strategy, estimated_steps = _heuristic_classify(goal_text.lower())
        confidence = Confidence.MEDIUM

    # P2 — deterministic complexity floor. Promote TRIVIAL/SIMPLE → COMPLEX on
    # hard structural signals (multi-deliverable artifacts / explicit
    # verification or recomputation), overriding an LLM that under-rated the
    # goal by step count. Floor the step estimate to the COMPLEX baseline when
    # promoted so the planning node is not misled. Never demotes.
    _floored = _apply_complexity_floor(goal_text, complexity)
    if _floored != complexity:
        logger.info(
            f"Complexity floor promoted {complexity.value} → {_floored.value} "
            f"(multi-deliverable / verification signals)"
        )
        complexity = _floored
        estimated_steps = max(
            estimated_steps, _estimate_steps(TaskComplexity.COMPLEX)
        )

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
        "refined_intent": refined_intent,
        "ambiguity_type": ambiguity_type,
        "ambiguity_severity": ambiguity_severity,
        "ambiguity_notes": ambiguity_notes,
    }


async def _llm_classify(
    gateway: LLMGateway,
    goal_text: str,
) -> TaskClassification | None:
    """Attempt LLM-based classification. Returns the parsed model, or None.

    Returning the full ``TaskClassification`` (not a positional tuple) keeps
    the new intent/ambiguity fields (Feature A) flowing without a brittle
    N-tuple. Confidence-enum derivation stays in ``classify_node``.
    """
    try:
        from src.graph.prompts import CLASSIFY_SYSTEM, CLASSIFY_USER
        from src.graph.schemas import TaskClassification
        from src.llm.structured_output import StructuredOutputManager

        user_prompt = CLASSIFY_USER.format(goal_text=goal_text)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": str(CLASSIFY_SYSTEM)},
            {"role": "user", "content": user_prompt},
        ]
        response = await gateway.acompletion(
            messages=messages,
            # node identity for per-node routing (findings-05 A); classify has
            # no complexity yet so this is documentary + future-proof.
            node="classify",
        )

        extractor = StructuredOutputManager()
        return await extractor.extract(
            response.content, TaskClassification, gateway=gateway, messages=messages
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
