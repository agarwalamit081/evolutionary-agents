"""Error handler node — classifies and recovers from errors."""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.graph.enums import Phase
from src.graph.state import AgentState


async def error_handler_node(state: AgentState) -> dict[str, Any]:
    """Handle errors that occur during graph execution.

    Classifies the error, determines recovery strategy, and routes
    to the appropriate next phase (retry, reclassify, escalate, or abort).

    Args:
        state: Current agent state with accumulated errors.

    Returns:
        Partial state update with recovery routing.
    """
    errors = state.get("errors", [])
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 25)

    if not errors:
        logger.warning("Error handler called with no errors")
        return {"phase": Phase.COMPLETE, "is_complete": True}

    last_error = str(errors[-1])
    logger.error(f"Handling error: {last_error[:200]}")

    # Classify the error
    error_lower = last_error.lower()

    # Budget exhausted → escalate to HITL
    if "budget" in error_lower:
        logger.error("Budget exhausted, escalating to human")
        return {
            "phase": Phase.HITL_GATE,
            "final_output": f"Budget exhausted: {last_error[:200]}",
        }

    # Max iterations exceeded → abort
    if iteration_count >= max_iterations:
        logger.error(f"Max iterations ({max_iterations}) exceeded, aborting")
        return {
            "phase": Phase.COMPLETE,
            "is_complete": True,
            "final_output": f"Task aborted after {max_iterations} iterations: {last_error[:200]}",
        }

    # Auth error → reclassify with different provider
    if any(kw in error_lower for kw in ["auth", "401", "403", "api key"]):
        logger.warning("Auth error, reclassifying with different provider")
        return {"phase": Phase.CLASSIFY}

    # Rate limit → retry after backoff
    if any(kw in error_lower for kw in ["rate", "429", "throttl"]):
        logger.info("Rate limit error, will retry")
        return {"phase": Phase.EXECUTE}

    # Default: retry execution
    logger.info("Generic error, retrying execution")
    return {"phase": Phase.EXECUTE}
