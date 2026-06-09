"""HITL gate node — human-in-the-loop approval checkpoint."""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.graph.enums import Phase
from src.graph.state import AgentState


async def hitl_gate_node(state: AgentState) -> dict[str, Any]:
    """Human-in-the-loop approval gate.

    Pauses execution for human review when risk is high or when
    the task involves high-stakes actions. Uses LangGraph's
    interrupt mechanism in production.

    Args:
        state: Current agent state awaiting approval.

    Returns:
        Partial state update after human review.
    """
    goal = state.get("current_goal")
    final_output = state.get("final_output", "")
    is_complete = state.get("is_complete", False)

    goal_text = goal.text if goal else "Unknown task"
    logger.info(f"HITL gate for: {goal_text[:60]}...")

    # In production with LangGraph interrupt:
    #   - This node would call interrupt() to pause
    #   - Human provides approval via Command(resume=...)
    #   - The graph resumes with the human's response
    #
    # For now, auto-approve all tasks
    logger.info("Auto-approving (HITL not yet connected)")

    return {
        "phase": Phase.COMPLETE,
        "is_complete": True,
        "final_output": final_output or f"Completed: {goal_text}",
    }
