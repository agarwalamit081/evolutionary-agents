"""Main LangGraph StateGraph definition for the task execution pipeline.

Graph topology:
    START → classify → plan → retrieve_memory → execute ↔ reflect
        → verify → evolve? → store_memory → hitl? → END

With error_handler reachable from all nodes via conditional edges.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from src.graph.nodes import (
    classify_node,
    error_handler_node,
    evolve_node,
    execute_node,
    hitl_gate_node,
    plan_node,
    reflect_node,
    retrieve_memory_node,
    store_memory_node,
    verify_node,
)
from src.graph.routers import (
    route_after_error,
    route_after_evolve,
    route_after_execute,
    route_after_hitl,
    route_after_reflect,
    route_after_store,
    route_after_verify,
)
from src.graph.state import AgentState


def build_task_graph() -> StateGraph:
    """Build and return the task execution StateGraph.

    The graph is not compiled — callers should compile with optional
    checkpointer and interrupt_before settings.

    Returns:
        StateGraph ready for compilation.
    """
    graph = StateGraph(AgentState)

    # ─── Add Nodes ──────────────────────────────────────────────────────
    graph.add_node("classify", classify_node)
    graph.add_node("plan", plan_node)
    graph.add_node("retrieve_memory", retrieve_memory_node)
    graph.add_node("execute", execute_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("verify", verify_node)
    graph.add_node("evolve", evolve_node)
    graph.add_node("store_memory", store_memory_node)
    graph.add_node("hitl_gate", hitl_gate_node)
    graph.add_node("error_handler", error_handler_node)

    # ─── Linear Edges (START → execute) ─────────────────────────────────
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "plan")
    graph.add_edge("plan", "retrieve_memory")
    graph.add_edge("retrieve_memory", "execute")

    # ─── Conditional Edges ──────────────────────────────────────────────
    graph.add_conditional_edges("execute", route_after_execute, {
        "reflect": "reflect",
        "execute": "execute",
        "error_handler": "error_handler",
    })

    graph.add_conditional_edges("reflect", route_after_reflect, {
        "verify": "verify",
        "execute": "execute",
        "plan": "plan",
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
    checkpointer: Any = None,
    interrupt_before: list[str] | None = None,
) -> Any:
    """Build and compile the task graph with optional checkpointing.

    Args:
        checkpointer: AsyncPostgresSaver or similar for state persistence.
        interrupt_before: Node names to pause before execution (e.g., ["hitl_gate"]).

    Returns:
        Compiled StateGraph ready for invocation.
    """
    graph = build_task_graph()

    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    if interrupt_before:
        compile_kwargs["interrupt_before"] = interrupt_before

    return graph.compile(**compile_kwargs)
