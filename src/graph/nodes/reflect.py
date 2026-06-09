"""Reflect node — self-reflection on execution progress."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from src.graph.enums import Confidence, Phase
from src.graph.models import ReflectionResult
from src.graph.state import AgentState

if TYPE_CHECKING:
    from src.llm.gateway import LLMGateway


async def reflect_node(
    state: AgentState,
    *,
    gateway: LLMGateway | None = None,
) -> dict[str, Any]:
    """Perform self-reflection on the execution so far.

    Evaluates completed steps, tool results, and progress toward the goal.
    Determines whether to continue, replan, or trigger evolution.

    Args:
        state: Current agent state with execution results.

    Returns:
        Partial state update with reflection results.
    """
    goal = state.get("current_goal")
    completed_steps = state.get("completed_steps", [])
    tool_results = state.get("tool_results", [])
    errors = state.get("errors", [])

    goal_text = goal.text if goal else "Unknown goal"
    logger.info(f"Reflecting on execution of: {goal_text[:60]}...")

    # Calculate completion ratio
    plan_steps = state.get("plan_steps", [])
    total_steps = len(plan_steps) if plan_steps else 1
    completed_count = len(completed_steps) if completed_steps else 0
    completion_ratio = completed_count / total_steps if total_steps > 0 else 0.0

    # Determine confidence based on progress
    has_errors = bool(errors)
    if has_errors:
        confidence = Confidence.LOW
    elif completion_ratio >= 0.8:
        confidence = Confidence.HIGH
    elif completion_ratio >= 0.5:
        confidence = Confidence.MEDIUM
    else:
        confidence = Confidence.LOW

    # Build reflection result
    lessons: list[str] = []
    memory_obs: list[str] = []

    if completed_steps:
        lessons.append(f"Completed {completed_count}/{total_steps} steps successfully")

    if has_errors:
        lessons.append(f"Encountered {len(errors)} errors during execution")
        memory_obs.append(f"Error pattern: {errors[-1][:100] if errors else 'none'}")

    # Determine if replanning is needed
    should_replan = completion_ratio < 0.3 and has_errors

    # Determine if evolution should be triggered
    should_evolve = (
        completion_ratio >= 0.5
        and confidence in {Confidence.HIGH, Confidence.VERY_HIGH}
        and len(completed_steps) >= 3
    )

    reflection = ReflectionResult(
        summary=f"Executed {completed_count}/{total_steps} steps for: {goal_text[:80]}",
        lessons_learned=lessons,
        confidence=confidence,
        should_evolve=should_evolve,
        should_replan=should_replan,
        memory_observations=memory_obs,
        cost_efficiency=1.0,
    )

    logger.info(
        f"Reflection: confidence={confidence.value}, "
        f"complete={completion_ratio:.0%}, "
        f"should_evolve={should_evolve}, should_replan={should_replan}"
    )

    return {
        "phase": Phase.VERIFY,
        "reflection": reflection,
        "confidence": confidence,
        "memory_observations": memory_obs,
    }
