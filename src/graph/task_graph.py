"""Main LangGraph StateGraph definition for the task execution pipeline.

Graph topology:
    START → classify → plan → retrieve_memory → execute ↔ reflect
        → verify → evolve? → store_memory → hitl? → END

With error_handler reachable from all nodes via conditional edges.

Dependencies (LLMGateway, MemoryManager, ToolRegistry) are injected via
closure wrappers in build_task_graph(). When deps are None, nodes fall
back to heuristic behavior.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph

from src.graph.nodes import (
    agent_spawn_node,
    classify_node,
    delegate_node,
    error_handler_node,
    evolve_node,
    execute_node,
    hitl_gate_node,
    plan_node,
    reflect_node,
    retrieve_memory_node,
    store_memory_node,
    tool_create_node,
    verify_node,
)
from src.graph.routers import (
    route_after_agent_spawn,
    route_after_delegate,
    route_after_error,
    route_after_evolve,
    route_after_execute,
    route_after_hitl,
    route_after_reflect,
    route_after_store,
    route_after_tool_create,
    route_after_verify,
)
from src.graph.state import AgentState

if TYPE_CHECKING:
    from src.agents.registry import SubAgentRegistry
    from src.llm.gateway import LLMGateway
    from src.memory.manager import MemoryManager
    from src.tools.registry import ToolRegistry


# ─── Closure Wrappers ───────────────────────────────────────────────


def _wrap(
    node_fn: Callable[..., Awaitable[dict[str, Any]]],
    **deps: Any,
) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    """Wrap a node function, injecting keyword dependencies via closure.

    The returned function matches LangGraph's expected signature:
        (state: AgentState) -> dict[str, Any]
    """

    async def _wrapped(state: AgentState) -> dict[str, Any]:
        return await node_fn(state, **deps)

    _wrapped.__name__ = node_fn.__name__
    _wrapped.__doc__ = node_fn.__doc__
    return _wrapped


# ─── Graph Builder ──────────────────────────────────────────────────


def build_task_graph(
    gateway: LLMGateway | None = None,
    memory: MemoryManager | None = None,
    tools: ToolRegistry | None = None,
    sub_agent_registry: SubAgentRegistry | None = None,
) -> StateGraph:
    """Build the task execution StateGraph with injected dependencies.

    Args:
        gateway: LLMGateway for LLM calls. None = heuristic fallback.
        memory: MemoryManager for 3-tier memory. None = stub behavior.
        tools: ToolRegistry for tool execution. None = no tool calls.
        sub_agent_registry: SubAgentRegistry for sub-agent delegation.

    Returns:
        StateGraph ready for compilation.
    """
    graph = StateGraph(AgentState)

    # ─── Add Nodes (with dependency injection) ────────────────────────
    # LangGraph's StateNode type is strict about signatures; our closure
    # wrappers match at runtime but Pyright can't verify that statically.
    graph.add_node("classify", _wrap(classify_node, gateway=gateway))  # type: ignore[arg-type]
    graph.add_node("plan", _wrap(plan_node, gateway=gateway, tools=tools))  # type: ignore[arg-type]
    graph.add_node("retrieve_memory", _wrap(retrieve_memory_node, memory=memory))  # type: ignore[arg-type]
    graph.add_node("execute", _wrap(execute_node, gateway=gateway, tools=tools))  # type: ignore[arg-type]
    graph.add_node("reflect", _wrap(reflect_node, gateway=gateway, tools=tools))  # type: ignore[arg-type]
    graph.add_node("verify", _wrap(verify_node, gateway=gateway))  # type: ignore[arg-type]
    graph.add_node("evolve", _wrap(evolve_node, gateway=gateway))  # type: ignore[arg-type]
    graph.add_node("store_memory", _wrap(store_memory_node, memory=memory))  # type: ignore[arg-type]
    graph.add_node("tool_create", _wrap(tool_create_node, gateway=gateway, tools=tools))  # type: ignore[arg-type]
    graph.add_node("agent_spawn", _wrap(agent_spawn_node, gateway=gateway, tools=tools, sub_agent_registry=sub_agent_registry))  # type: ignore[arg-type]
    graph.add_node("delegate", _wrap(delegate_node, gateway=gateway, tools=tools, sub_agent_registry=sub_agent_registry, memory=memory))  # type: ignore[arg-type]
    # No deps needed for HITL and error handler
    graph.add_node("hitl_gate", hitl_gate_node)  # type: ignore[arg-type]
    graph.add_node("error_handler", error_handler_node)  # type: ignore[arg-type]

    # ─── Linear Edges (START → execute) ────────────────────────────────
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "plan")
    graph.add_edge("plan", "retrieve_memory")
    graph.add_edge("retrieve_memory", "execute")

    # ─── Conditional Edges ─────────────────────────────────────────────
    graph.add_conditional_edges("execute", route_after_execute, {
        "reflect": "reflect",
        "execute": "execute",
        "error_handler": "error_handler",
    })

    graph.add_conditional_edges("reflect", route_after_reflect, {
        "agent_spawn": "agent_spawn",
        "tool_create": "tool_create",
        "verify": "verify",
        "execute": "execute",
        "plan": "plan",
    })

    graph.add_conditional_edges("tool_create", route_after_tool_create, {
        "plan": "plan",
        "execute": "execute",
    })

    graph.add_conditional_edges("agent_spawn", route_after_agent_spawn, {
        "delegate": "delegate",
        "plan": "plan",
        "tool_create": "tool_create",
    })

    graph.add_conditional_edges("delegate", route_after_delegate, {
        "verify": "verify",
        "execute": "execute",
    })

    graph.add_conditional_edges("verify", route_after_verify, {
        "evolve": "evolve",
        "store_memory": "store_memory",
        "execute": "execute",
    })

    graph.add_conditional_edges("evolve", route_after_evolve, {
        "store_memory": "store_memory",
        "error_handler": "error_handler",
    })

    graph.add_conditional_edges("store_memory", route_after_store, {
        "hitl_gate": "hitl_gate",
        "complete": END,
        "execute": "execute",
    })

    graph.add_conditional_edges("hitl_gate", route_after_hitl, {
        "complete": END,
        "execute": "execute",
    })

    graph.add_conditional_edges("error_handler", route_after_error, {
        "execute": "execute",
        "classify": "classify",
        "hitl_gate": "hitl_gate",
        "complete": END,
    })

    return graph


def compile_task_graph(
    gateway: LLMGateway | None = None,
    memory: MemoryManager | None = None,
    tools: ToolRegistry | None = None,
    checkpointer: Any = None,
    interrupt_before: list[str] | None = None,
    sub_agent_registry: Any = None,
) -> Any:
    """Build and compile the task graph with dependencies and optional checkpointing.

    Args:
        gateway: LLMGateway for LLM calls.
        memory: MemoryManager for 3-tier memory.
        tools: ToolRegistry for tool execution.
        checkpointer: AsyncPostgresSaver or similar for state persistence.
        interrupt_before: Node names to pause before execution (e.g., ["hitl_gate"]).
        sub_agent_registry: SubAgentRegistry for sub-agent delegation.

    Returns:
        Compiled StateGraph ready for invocation.
    """
    graph = build_task_graph(
        gateway=gateway,
        memory=memory,
        tools=tools,
        sub_agent_registry=sub_agent_registry,
    )

    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    if interrupt_before:
        compile_kwargs["interrupt_before"] = interrupt_before

    return graph.compile(**compile_kwargs)
