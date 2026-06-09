"""Verify node — validates execution results against success criteria."""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.graph.enums import Confidence, Phase
from src.graph.state import AgentState


async def verify_node(state: AgentState) -> dict[str, Any]:
    """Verify execution results against the goal's success criteria.

    Checks if completed steps satisfy the goal requirements and
    determines if the task is complete or needs more work.

    Args:
        state: Current agent state with reflection and execution results.

    Returns:
        Partial state update with verification result.
    """
    goal = state.get("current_goal")
    reflection = state.get("reflection")
    completed_steps = state.get("completed_steps", [])
    errors = state.get("errors", [])
    plan_steps = state.get("plan_steps", [])
    step_index = state.get("current_step_index", 0)

    goal_text = goal.text if goal else "Unknown goal"
    logger.info(f"Verifying results for: {goal_text[:60]}...")

    # Calculate verification metrics
    total_steps = len(plan_steps)
    completed_count = len(completed_steps)
    has_errors = bool(errors)
    confidence = state.get("confidence", Confidence.MEDIUM)

    # Determine completion
    all_steps_done = step_index >= total_steps if total_steps > 0 else True
    no_errors = not has_errors
    high_confidence = confidence in {Confidence.HIGH, Confidence.VERY_HIGH}

    is_complete = all_steps_done and no_errors and high_confidence

    # Build final output if complete
    final_output = ""
    if is_complete:
        if reflection and hasattr(reflection, "summary"):
            final_output = reflection.summary
        else:
            completed_descriptions = [
                s.description for s in completed_steps
                if hasattr(s, "description")
            ]
            final_output = (
                f"Task completed successfully.\n"
                f"Goal: {goal_text}\n"
                f"Steps completed: {completed_count}/{total_steps}\n"
                f"Results: {'; '.join(completed_descriptions[:5])}"
            )
        logger.info("Verification PASSED — task complete")
    else:
        reasons = []
        if not all_steps_done:
            reasons.append(f"steps remaining ({step_index}/{total_steps})")
        if has_errors:
            reasons.append(f"{len(errors)} errors")
        if not high_confidence:
            reasons.append(f"low confidence ({confidence.value})")
        logger.info(f"Verification incomplete: {', '.join(reasons)}")

    return {
        "phase": Phase.COMPLETE if is_complete else Phase.EXECUTE,
        "is_complete": is_complete,
        "final_output": final_output,
    }
