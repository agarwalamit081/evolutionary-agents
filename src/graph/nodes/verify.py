"""Verify node — validates execution results against success criteria."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from src.graph.enums import Confidence, Phase
from src.graph.state import AgentState

if TYPE_CHECKING:
    from src.llm.gateway import LLMGateway


async def verify_node(
    state: AgentState,
    *,
    gateway: LLMGateway | None = None,
) -> dict[str, Any]:
    """Verify execution results against the goal's success criteria.

    When gateway is provided, uses LLM for semantic verification.
    Otherwise falls back to heuristic checks.

    Args:
        state: Current agent state with reflection and execution results.
        gateway: Optional LLM gateway for LLM-enhanced verification.

    Returns:
        Partial state update with verification result.
    """
    goal = state.get("current_goal")
    reflection = state.get("reflection")
    completed_steps = state.get("completed_steps", [])
    errors = state.get("errors", [])
    plan_steps = state.get("plan_steps", [])
    step_index = state.get("current_step_index", 0)
    confidence = state.get("confidence", Confidence.MEDIUM)

    goal_text = goal.text if goal else "Unknown goal"
    logger.info(f"Verifying results for: {goal_text[:60]}...")

    # Try LLM verification first, fall back to heuristics
    if gateway is not None:
        result = await _llm_verify(gateway, state)
        if result is not None:
            return result

    return _heuristic_verify(state, goal_text, reflection, completed_steps, errors, plan_steps, step_index, confidence)


def _heuristic_verify(
    _state: AgentState,  # noqa: ARG001 — kept for interface consistency
    goal_text: str,
    reflection: Any,
    completed_steps: list[Any],
    errors: list[str],
    plan_steps: list[Any],
    step_index: int,
    confidence: Confidence,
) -> dict[str, Any]:
    """Heuristic verification based on step completion and error state."""
    total_steps = len(plan_steps)
    completed_count = len(completed_steps)
    has_errors = bool(errors)

    all_steps_done = step_index >= total_steps if total_steps > 0 else True
    no_errors = not has_errors
    high_confidence = confidence in {Confidence.HIGH, Confidence.VERY_HIGH}

    is_complete = all_steps_done and no_errors and high_confidence

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


async def _llm_verify(
    gateway: LLMGateway,
    state: AgentState,
) -> dict[str, Any] | None:
    """Attempt LLM-based verification. Returns None on failure."""
    try:
        from src.graph.prompts import VERIFY_SYSTEM, VERIFY_USER
        from src.graph.schemas import VerificationResult
        from src.llm.structured_output import StructuredOutputManager

        goal = state.get("current_goal")
        completed_steps = state.get("completed_steps", [])
        errors = state.get("errors", [])
        plan_steps = state.get("plan_steps", [])
        reflection = state.get("reflection")

        goal_text = goal.text if goal else "Unknown goal"
        success_criteria = ""
        if goal and hasattr(goal, "success_criteria") and goal.success_criteria:
            success_criteria = "; ".join(goal.success_criteria)
        else:
            success_criteria = "Goal is fully achieved"

        total_steps = len(plan_steps)
        completed_count = len(completed_steps)

        completed_summary = "\n".join(
            f"- {s.description}: {getattr(s, 'result', 'done')}" for s in completed_steps[-5:]
        ) if completed_steps else "None yet"

        final_output = ""
        if reflection and hasattr(reflection, "summary"):
            final_output = reflection.summary

        user_prompt = VERIFY_USER.format(
            goal_text=goal_text,
            success_criteria=success_criteria,
            completed_summary=completed_summary,
            completed_count=completed_count,
            total_steps=total_steps,
            error_count=len(errors),
            final_output=final_output or "In progress",
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": VERIFY_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

        response = await gateway.acompletion(
            messages=messages,
            model="gpt-4o-mini-2024-07-18",
        )

        extractor = StructuredOutputManager()
        verification = await extractor.extract(response.content, VerificationResult)
        if verification is None:
            return None

        is_complete = verification.is_complete

        output = (
            f"Verification: {verification.completion_percentage:.0f}% complete. "
            f"{verification.quality_assessment}"
        )
        if verification.gaps:
            output += f" Gaps: {'; '.join(verification.gaps[:3])}"

        logger.info(
            f"LLM Verification: complete={is_complete}, "
            f"progress={verification.completion_percentage:.0f}%"
        )

        return {
            "phase": Phase.COMPLETE if is_complete else Phase.EXECUTE,
            "is_complete": is_complete,
            "final_output": output,
        }
    except Exception as e:
        logger.debug(f"LLM verification failed, using heuristics: {e}")
        return None
