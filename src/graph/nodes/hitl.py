"""HITL gate node — human-in-the-loop approval checkpoint."""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.graph.enums import Phase
from src.graph.state import AgentState

# LangGraph interrupt is imported lazily to avoid hard dependency
# at module level — it's only needed when HITL is actually enabled.


async def hitl_gate_node(state: AgentState) -> dict[str, Any]:
    """Human-in-the-loop approval gate.

    When the agent config has HITL enabled and LangGraph interrupt
    is available, pauses execution for human review. Otherwise
    auto-approves the task.

    Args:
        state: Current agent state awaiting approval.

    Returns:
        Partial state update after review.
    """
    goal = state.get("current_goal")
    final_output = state.get("final_output", "")
    is_complete = state.get("is_complete", False)

    goal_text = goal.text if goal else "Unknown task"
    logger.info(f"HITL gate for: {goal_text[:60]}...")

    # Try LangGraph interrupt for real HITL
    try:
        from langgraph.types import interrupt

        human_response = interrupt({
            "question": f"Approve the result for: {goal_text[:100]}?",
            "output": final_output[:500] if final_output else "No output yet",
            "is_complete": is_complete,
        })

        if isinstance(human_response, dict):
            approved = human_response.get("approved", True)
            feedback = human_response.get("feedback", "")
        else:
            approved = bool(human_response)
            feedback = ""

        if approved:
            logger.info("HITL: Approved by human")
            return {
                "phase": Phase.COMPLETE,
                "is_complete": True,
                "final_output": final_output or f"Completed: {goal_text}",
            }
        else:
            logger.info(f"HITL: Rejected by human — {feedback[:80]}")
            return {
                "phase": Phase.EXECUTE,
                "is_complete": False,
                "errors": [f"Human rejected: {feedback}"] if feedback else ["Human rejected the result"],
            }
    except (ImportError, TypeError):
        # LangGraph interrupt not available or not in compiled graph context
        logger.debug("Auto-approving (HITL interrupt not available)")

    return {
        "phase": Phase.COMPLETE,
        "is_complete": True,
        "final_output": final_output or f"Completed: {goal_text}",
    }
