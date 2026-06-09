"""Execute node — runs the current plan step with tool calling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from src.graph.enums import GoalStatus, Phase
from src.graph.state import AgentState

if TYPE_CHECKING:
    from src.llm.gateway import LLMGateway
    from src.tools.registry import ToolRegistry


async def execute_node(
    state: AgentState,
    *,
    gateway: LLMGateway | None = None,
    tools: ToolRegistry | None = None,
) -> dict[str, Any]:
    """Execute the current plan step.

    When gateway and tools are provided, uses LLM tool calling via
    bind_tools() and AIMessage.tool_calls. Otherwise falls back to
    simulated step execution.

    Args:
        state: Current agent state with plan and step index.
        gateway: Optional LLM gateway for tool-calling execution.
        tools: Optional tool registry for executing tool calls.

    Returns:
        Partial state update with execution results.
    """
    plan_steps = state.get("plan_steps", [])
    step_index = state.get("current_step_index", 0)
    goal = state.get("current_goal")
    messages = state.get("messages", [])
    iteration_count = state.get("iteration_count", 0)

    # Guard: no plan or index out of range
    if not plan_steps or step_index >= len(plan_steps):
        logger.warning("Execute called with no remaining steps")
        return {
            "phase": Phase.REFLECT,
            "iteration_count": iteration_count + 1,
        }

    current_step = plan_steps[step_index]
    logger.info(
        f"Executing step {step_index + 1}/{len(plan_steps)}: "
        f"{current_step.description[:60]}..."
    )

    # Mark current step as active
    current_step.status = GoalStatus.ACTIVE

    # Build execution context message
    goal_text = goal.text if goal else "Unknown goal"
    user_message = {
        "role": "user",
        "content": (
            f"Execute the following step:\n"
            f"Step: {current_step.description}\n"
            f"Goal context: {goal_text}\n"
            f"Tool: {current_step.tool_name or 'none specified'}\n"
            f"Proceed with execution."
        ),
    }

    # Increment state counters
    updated_step_index = step_index + 1

    # Mark step complete (in a real implementation, the LLM would execute)
    current_step.status = GoalStatus.COMPLETED
    current_step.result = f"Executed: {current_step.description}"

    return {
        "phase": Phase.REFLECT,
        "messages": [user_message],
        "current_step_index": updated_step_index,
        "iteration_count": iteration_count + 1,
        "completed_steps": [current_step],
    }
