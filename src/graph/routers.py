"""Conditional edge routing functions for the task execution graph.

Each router inspects state and returns the name of the next node.
"""

from __future__ import annotations

from loguru import logger

from src.graph.enums import Confidence
from src.graph.state import AgentState


def route_after_execute(state: AgentState) -> str:
    """Route after the execute node.

    Returns:
        "reflect" — plan exhausted or max iterations reached
        "execute" — more steps to execute (loop)
        "error_handler" — non-retriable error occurred
    """
    errors = state.get("errors", [])
    tool_results = state.get("tool_results", [])
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 25)
    plan_steps = state.get("plan_steps", [])
    step_index = state.get("current_step_index", 0)

    # Check for non-retriable errors
    if errors:
        last_error = errors[-1] if isinstance(errors, list) else str(errors)
        if "authentication" in last_error.lower() or "authorization" in last_error.lower():
            logger.warning("Non-retriable auth error, routing to error_handler")
            return "error_handler"

    # Check for tool execution failures that are retriable
    if tool_results:
        last_result = tool_results[-1]
        if hasattr(last_result, "success") and not last_result.success:
            error_str = getattr(last_result, "error", "") or ""
            if "timeout" in error_str.lower() or "rate" in error_str.lower():
                logger.info("Retriable tool error, continuing execute loop")
                return "execute"
            # Non-retriable tool error
            logger.warning(f"Non-retriable tool error: {error_str[:100]}")
            return "error_handler"

    # Max iterations reached → reflect on what we have
    if iteration_count >= max_iterations:
        logger.info(f"Max iterations ({max_iterations}) reached, routing to reflect")
        return "reflect"

    # Plan exhausted → reflect
    if plan_steps and step_index >= len(plan_steps):
        logger.info("All plan steps executed, routing to reflect")
        return "reflect"

    # More steps to execute → loop back
    return "execute"


def route_after_reflect(state: AgentState) -> str:
    """Route after the reflect node.

    Returns:
        "verify" — confidence is medium or higher
        "execute" — low confidence, retry execution
        "plan" — reflection suggests replanning
    """
    reflection = state.get("reflection")

    if reflection is None:
        return "verify"

    # Check for replanning trigger
    if hasattr(reflection, "should_replan") and reflection.should_replan:
        logger.info("Reflection suggests replanning")
        return "plan"

    # Check confidence level
    confidence = state.get("confidence", Confidence.MEDIUM)
    if isinstance(confidence, str):
        confidence = Confidence(confidence)

    low_confidence_levels = {Confidence.VERY_LOW, Confidence.LOW}
    if confidence in low_confidence_levels:
        logger.info(f"Low confidence ({confidence.value}), routing back to execute")
        return "execute"

    return "verify"


def route_after_verify(state: AgentState) -> str:
    """Route after the verify node.

    Returns:
        "evolve" — verification passed + should_evolve flag
        "store_memory" — verification passed, no evolution needed
        "execute" — verification failed or partial, retry
    """
    is_complete = state.get("is_complete", False)
    confidence = state.get("confidence", Confidence.MEDIUM)
    reflection = state.get("reflection")

    if is_complete:
        should_evolve = False
        if reflection and hasattr(reflection, "should_evolve"):
            should_evolve = reflection.should_evolve

        if should_evolve:
            logger.info("Verification passed, triggering evolution")
            return "evolve"
        return "store_memory"

    # Not complete — check if we should retry
    if isinstance(confidence, str):
        confidence = Confidence(confidence)

    if confidence in {Confidence.VERY_LOW, Confidence.LOW}:
        logger.info("Verification failed with low confidence, retrying execute")
        return "execute"

    # Medium confidence — try once more
    logger.info("Verification partial, retrying execute")
    return "execute"


def route_after_evolve(state: AgentState) -> str:
    """Route after the evolve node.

    Returns:
        "store_memory" — evolution succeeded
        "error_handler" — evolution failed
    """
    errors = state.get("errors", [])
    # If new errors were added during evolution, route to error handler
    if errors and "evolution" in str(errors[-1]).lower():
        logger.warning("Evolution failed, routing to error_handler")
        return "error_handler"

    return "store_memory"


def route_after_store(state: AgentState) -> str:
    """Route after the store_memory node.

    Returns:
        "hitl_gate" — HITL required (high risk or explicit flag)
        "complete" — normal completion
    """
    is_complete = state.get("is_complete", False)

    if not is_complete:
        # Not done yet, continue execution
        return "execute"

    # For now, complete directly. HITL can be enabled via config.
    return "complete"


def route_after_hitl(state: AgentState) -> str:
    """Route after the HITL gate node.

    Returns:
        "complete" — human approved
        "execute" — human requested revision
    """
    is_complete = state.get("is_complete", True)
    if is_complete:
        return "complete"
    return "execute"


def route_after_error(state: AgentState) -> str:
    """Route after the error_handler node.

    Returns:
        "execute" — retriable error, retry
        "classify" — needs reclassification
        "hitl_gate" — needs human intervention
        "complete" — fatal error, abort
    """
    errors = state.get("errors", [])
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 25)

    if not errors:
        return "complete"

    last_error = str(errors[-1]).lower()

    # Budget exhausted → human intervention
    if "budget" in last_error:
        logger.error("Budget exhausted, escalating to HITL")
        return "hitl_gate"

    # Rate limit → retry with backoff
    if "rate" in last_error or "429" in last_error:
        if iteration_count < max_iterations:
            return "execute"
        return "complete"

    # Auth errors → reclassify with different provider
    if "auth" in last_error or "401" in last_error or "403" in last_error:
        return "classify"

    # Max iterations exceeded → abort
    if iteration_count >= max_iterations:
        logger.error(f"Max iterations ({max_iterations}) exceeded with errors, aborting")
        return "complete"

    # Default: retry execution
    return "execute"
