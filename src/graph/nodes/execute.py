"""Execute node — runs the current plan step with tool calling."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from loguru import logger

from src.graph.enums import GoalStatus, Phase
from src.graph.models import ToolResult
from src.graph.state import AgentState

if TYPE_CHECKING:
    from src.llm.gateway import LLMGateway
    from src.tools.registry import ToolRegistry
    from src.tools.result_cache import ToolResultCache

# Maximum concurrent tool calls per execute step
MAX_CONCURRENT_TOOLS = 5


def _messages_to_openai(messages: list[Any]) -> list[dict[str, Any]]:
    """Convert LangChain messages (or OpenAI dicts) to OpenAI chat-format dicts.

    The gateway (litellm) expects OpenAI message dicts, while the graph state
    stores LangChain ``AnyMessage`` objects. This bridges the two so the
    execute node can feed the real conversation history into each LLM call —
    without it, the agent is stateless across steps and memory folding has no
    real context to compress.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, dict):
            out.append(m)
            continue
        mtype = getattr(m, "type", "")
        content = getattr(m, "content", "")
        if mtype == "human":
            out.append({"role": "user", "content": content})
        elif mtype == "system":
            out.append({"role": "system", "content": content})
        elif mtype == "ai":
            entry: dict[str, Any] = {"role": "assistant", "content": content}
            tcs = getattr(m, "tool_calls", None) or []
            if tcs:
                entry["tool_calls"] = [
                    {
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": tc.get("name", ""),
                            "arguments": json.dumps(
                                tc.get("args", {}), default=str
                            ),
                        },
                    }
                    for tc in tcs
                ]
            out.append(entry)
        elif mtype == "tool":
            entry_t: dict[str, Any] = {
                "role": "tool",
                "content": content,
                "tool_call_id": getattr(m, "tool_call_id", ""),
            }
            name = getattr(m, "name", None)
            if name:
                entry_t["name"] = name
            out.append(entry_t)
        else:
            out.append({"role": "user", "content": str(content)})
    return out


def _build_ai_message(
    content: str | None,
    tool_calls: list[dict[str, Any]],
) -> AIMessage:
    """Build an AIMessage for the thread, attaching validated tool calls.

    Falls back to a content-only message if tool-call validation fails, so a
    malformed provider response never breaks the run.
    """
    text = content or ""
    if not tool_calls:
        return AIMessage(content=text)
    try:
        normalized = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            raw_args = fn.get("arguments", "{}")
            try:
                args = (
                    json.loads(raw_args)
                    if isinstance(raw_args, str)
                    else raw_args
                )
            except (json.JSONDecodeError, TypeError):
                args = {}
            normalized.append(
                {
                    "name": fn.get("name", ""),
                    "args": args,
                    "id": tc.get("id", ""),
                    "type": "tool_call",
                }
            )
        return AIMessage(content=text, tool_calls=normalized)
    except Exception as exc:  # noqa: BLE001 — never break the run on validation
        logger.debug(
            f"AIMessage tool_call validation failed, storing content only: {exc}"
        )
        return AIMessage(content=text)


async def execute_node(
    state: AgentState,
    *,
    gateway: LLMGateway | None = None,
    tools: ToolRegistry | None = None,
    result_cache: ToolResultCache | None = None,
) -> dict[str, Any]:
    """Execute the current plan step.

    When gateway and tools are provided, uses LLM tool calling.
    Otherwise falls back to simulated step execution.

    Args:
        state: Current agent state with plan and step index.
        gateway: Optional LLM gateway for tool-calling execution.
        tools: Optional tool registry for executing tool calls.
        result_cache: Optional Redis cache for idempotent tool results.
            Only tools flagged ``cacheable`` in the registry are routed
            through it; the cache is best-effort and never breaks a call.

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
        result = await _llm_execute(
            gateway, tools, state, current_step.description, goal_text, result_cache
        )
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
    result_cache: ToolResultCache | None = None,
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

        # ── Stateful ReAct thread ─────────────────────────────────────────
        # state["messages"] is the canonical conversation history (seeded with
        # the goal by initial_state). Feed it to the LLM so the agent reasons
        # across steps, and append this step's user turn + assistant reply +
        # tool results so memory folding has real context to compress. Without
        # this, every step runs blind and folds save ~nothing.
        history = state.get("messages", [])
        step_label = (
            f"Execute step {step_index + 1}/{len(plan_steps)}: {step_description}"
        )

        openai_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            *_messages_to_openai(history),
            {"role": "user", "content": step_label},
        ]

        # Get tool definitions for function calling
        tool_defs = tools.list_tools()

        response = await gateway.acompletion_with_tools(
            messages=openai_messages,
            tools=tool_defs,
        )

        # Process tool calls if present. gather preserves order, so each result
        # zips 1:1 with its tool_call — needed for ToolMessage correlation.
        raw_tool_calls: list[dict[str, Any]] = response.tool_calls or []
        new_tool_results: list[ToolResult] = []
        if raw_tool_calls:
            new_tool_results = await _execute_tool_calls_parallel(
                raw_tool_calls, tools, result_cache
            )

        # Append the real turn to the conversation thread.
        new_messages: list[Any] = [HumanMessage(content=step_label)]
        new_messages.append(_build_ai_message(response.content, raw_tool_calls))
        for tc, tr in zip(raw_tool_calls, new_tool_results, strict=False):
            tr.metadata["tool_call_id"] = tc.get("id", "")
            new_messages.append(
                ToolMessage(
                    content=tr.output or tr.error or "",
                    tool_call_id=tc.get("id", ""),
                    name=tr.tool_name,
                )
            )

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


async def _execute_tool_call(
    tc: dict[str, Any],
    tools: ToolRegistry,
    cache: ToolResultCache | None = None,
) -> ToolResult:
    """Execute a single tool call and return a ToolResult.

    For idempotent, cacheable tools (opt-in via the registry), successful
    results are served from / written to ``cache`` so repeated calls within
    and across runs skip the underlying work. Errors are never cached and any
    cache failure degrades to a transparent miss.

    Args:
        tc: Tool call dict with function.name and function.arguments.
        tools: Tool registry for handler lookup.
        cache: Optional result cache for cacheable tools.

    Returns:
        ToolResult with success/failure status.
    """
    tool_name = tc.get("function", {}).get("name", tc.get("name", ""))
    tool_args_str = tc.get("function", {}).get("arguments", tc.get("args", "{}"))

    handler = tools.get_handler(tool_name)
    if handler is None:
        return ToolResult(
            tool_name=tool_name,
            success=False,
            output="",
            error=f"Unknown tool: {tool_name}",
        )

    # Parse args up front: the cache key needs canonical args, and a clean
    # parse error is more useful than a generic handler exception.
    try:
        args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
    except (json.JSONDecodeError, TypeError) as exc:
        return ToolResult(
            tool_name=tool_name,
            success=False,
            output="",
            error=f"Invalid arguments for {tool_name}: {exc}",
        )

    # Cache lookup — only for opt-in cacheable tools.
    if cache is not None and tools.is_cacheable(tool_name):
        cached = await cache.get(tool_name, args)
        if cached is not None:
            logger.debug(f"Tool cache HIT: {tool_name}")
            return ToolResult(
                tool_name=tool_name,
                success=bool(cached.get("success", True)),
                output=str(cached.get("output", "")),
                error=cached.get("error"),
                metadata={"cached": True},
            )

    try:
        result = await handler(**args)
        tr = ToolResult(
            tool_name=tool_name,
            success=True,
            output=str(result)[:2000],
        )
        # Cache only successful results of cacheable tools (never errors).
        if cache is not None and tools.is_cacheable(tool_name):
            await cache.set(
                tool_name,
                args,
                {
                    "success": tr.success,
                    "output": tr.output,
                    "error": tr.error,
                },
            )
        return tr
    except Exception as e:
        return ToolResult(
            tool_name=tool_name,
            success=False,
            output="",
            error=str(e)[:500],
        )


async def _execute_tool_calls_parallel(
    tool_calls: list[dict[str, Any]],
    tools: ToolRegistry,
    cache: ToolResultCache | None = None,
) -> list[ToolResult]:
    """Execute multiple tool calls concurrently with semaphore limiting.

    Uses asyncio.gather so independent tool calls run in parallel.
    A semaphore caps concurrency at MAX_CONCURRENT_TOOLS to avoid
    overwhelming external services.

    Args:
        tool_calls: List of tool call dicts from LLM response.
        tools: Tool registry for handler lookup.
        cache: Optional result cache forwarded to each call.

    Returns:
        List of ToolResult in the same order as tool_calls.
    """
    if not tool_calls:
        return []

    if len(tool_calls) == 1:
        # Single call — skip gather overhead
        return [await _execute_tool_call(tool_calls[0], tools, cache)]

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TOOLS)

    async def _limited(tc: dict[str, Any]) -> ToolResult:
        async with semaphore:
            return await _execute_tool_call(tc, tools, cache)

    results = await asyncio.gather(*[_limited(tc) for tc in tool_calls])
    return list(results)
