"""Execute node — runs the current plan step with tool calling."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.graph.enums import GoalStatus, Phase
from src.graph.models import ToolResult
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

    When gateway and tools are provided, uses LLM tool calling.
    Otherwise falls back to simulated step execution.

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

    # Build execution context
    goal_text = goal.text if goal else "Unknown goal"

    # Try LLM + tool execution first, fall back to simulated
    if gateway is not None and tools is not None:
        result = await _llm_execute(gateway, tools, state, current_step.description, goal_text)
        if result is not None:
            return result

    # Simulated execution fallback
    return await _simulated_execute(state, current_step, goal_text, messages, iteration_count)


async def _llm_execute(
    gateway: LLMGateway,
    tools: ToolRegistry,
    state: AgentState,
    step_description: str,
    goal_text: str,
) -> dict[str, Any] | None:
    """Execute step via LLM with tool calling. Returns None on failure."""
    try:
        from src.graph.prompts import EXECUTE_SYSTEM

        plan_steps = state.get("plan_steps", [])
        step_index = state.get("current_step_index", 0)
        completed_steps = state.get("completed_steps", [])
        tool_results = state.get("tool_results", [])
        iteration_count = state.get("iteration_count", 0)
        memories = state.get("retrieved_memories", [])

        # Build context
        memory_ctx = ""
        if memories:
            memory_ctx = "\nRelevant context:\n" + "\n".join(f"- {m}" for m in memories[:3])

        tool_results_ctx = ""
        if tool_results:
            recent = tool_results[-3:]
            tool_results_ctx = "\nRecent tool results:\n" + "\n".join(
                f"- {r.tool_name}: {r.output[:100]}" for r in recent if hasattr(r, "tool_name")
            )

        system_prompt = EXECUTE_SYSTEM.format(
            goal_text=goal_text,
            completed_count=len(completed_steps),
            total_steps=len(plan_steps),
            step_description=step_description,
            memory_context=memory_ctx,
            tool_results_context=tool_results_ctx,
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Execute this step now: {step_description}"},
        ]

        # Get tool definitions for function calling
        tool_defs = tools.list_tools()

        response = await gateway.acompletion_with_tools(
            messages=messages,
            tools=tool_defs,
        )

        # Process tool calls if present
        new_tool_results: list[ToolResult] = []
        new_messages: list[dict[str, str]] = [
            {"role": "user", "content": f"Executed step: {step_description}"},
        ]

        if response.tool_calls:
            for tc in response.tool_calls:
                tool_name = tc.get("function", {}).get("name", tc.get("name", ""))
                tool_args_str = tc.get("function", {}).get("arguments", tc.get("args", "{}"))

                handler = tools.get_handler(tool_name)
                if handler is None:
                    new_tool_results.append(ToolResult(
                        tool_name=tool_name,
                        success=False,
                        output="",
                        error=f"Unknown tool: {tool_name}",
                    ))
                    continue

                try:
                    args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
                    result = await handler(**args)
                    new_tool_results.append(ToolResult(
                        tool_name=tool_name,
                        success=True,
                        output=str(result)[:2000],
                    ))
                except Exception as e:
                    new_tool_results.append(ToolResult(
                        tool_name=tool_name,
                        success=False,
                        output="",
                        error=str(e)[:500],
                    ))

        # Mark step complete with LLM result
        current_step = plan_steps[step_index]
        current_step.status = GoalStatus.COMPLETED
        result_text = response.content or step_description
        current_step.result = result_text[:500]

        return {
            "phase": Phase.REFLECT,
            "messages": new_messages,
            "current_step_index": step_index + 1,
            "iteration_count": iteration_count + 1,
            "completed_steps": [current_step],
            "tool_results": new_tool_results,
        }
    except Exception as e:
        logger.debug(f"LLM execution failed, using simulated: {e}")
        return None


async def _simulated_execute(
    state: AgentState,
    current_step: Any,
    goal_text: str,
    messages: list[Any],  # noqa: ARG001 — kept for future use
    iteration_count: int,
) -> dict[str, Any]:
    """Simulated step execution (heuristic fallback)."""
    step_index = state.get("current_step_index", 0)

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

    current_step.status = GoalStatus.COMPLETED
    current_step.result = f"Executed: {current_step.description}"

    return {
        "phase": Phase.REFLECT,
        "messages": [user_message],
        "current_step_index": step_index + 1,
        "iteration_count": iteration_count + 1,
        "completed_steps": [current_step],
    }
