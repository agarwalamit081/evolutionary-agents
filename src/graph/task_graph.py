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

import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph
from loguru import logger

from src.config.settings import get_settings
from src.graph.nodes import (
    agent_spawn_node,
    classify_node,
    delegate_node,
    disambiguate_node,
    error_handler_node,
    evolve_node,
    execute_node,
    hitl_gate_node,
    lats_search_node,
    plan_node,
    reflect_node,
    retrieve_memory_node,
    structure_analysis_node,
    store_memory_node,
    tool_create_node,
    verify_node,
)
from src.graph.routers import (
    route_after_agent_spawn,
    route_after_classify,
    route_after_delegate,
    route_after_error,
    route_after_evolve,
    route_after_execute,
    route_after_hitl,
    route_after_reflect,
    route_after_store,
    route_after_structure_analysis,
    route_after_tool_create,
    route_after_verify,
)
from src.graph.state import AgentState

if TYPE_CHECKING:
    from src.agents.registry import SubAgentRegistry
    from src.llm.gateway import LLMGateway
    from src.memory.manager import MemoryManager
    from src.tools.registry import ToolRegistry

from src.tools.result_cache import ToolResultCache


# ─── Closure Wrappers ───────────────────────────────────────────────


def _wrap(
    node_fn: Callable[..., Awaitable[dict[str, Any]]],
    **deps: Any,
) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    """Wrap a node function, injecting keyword dependencies via closure.

    The returned function matches LangGraph's expected signature:
        (state: AgentState) -> dict[str, Any]

    Each wrapped node is timed for attribution (Track-1): the Prometheus
    histogram is always observed, and when ``tool_metrics_enabled`` is on a
    lightweight per-node timing row is persisted to ``execution_steps`` (keyed
    by the active ``run_id``) so ``run_metrics`` can read per-node wall-clock
    per run. Both are non-fatal — a timing/DB hiccup can never break a node.
    """
    node_name = node_fn.__name__

    async def _wrapped(state: AgentState) -> dict[str, Any]:
        start = time.perf_counter()
        status = "completed"
        try:
            return await node_fn(state, **deps)
        except BaseException:
            status = "failed"
            raise
        finally:
            elapsed_s = time.perf_counter() - start
            try:
                from src.observability.metrics import record_node_duration

                record_node_duration(node_name, elapsed_s)
            except Exception:  # noqa: BLE001 — metrics must never break a node
                pass
            try:
                if get_settings().agent.tool_metrics_enabled:
                    await _persist_node_step(node_name, status, int(elapsed_s * 1000))
            except Exception:  # noqa: BLE001 — timing must never break a node
                pass

    _wrapped.__name__ = node_fn.__name__
    _wrapped.__doc__ = node_fn.__doc__
    return _wrapped


async def _persist_node_step(node_name: str, status: str, duration_ms: int) -> None:
    """Persist one lightweight per-node timing row (Track-1 attribution).

    Revives the dormant ``execution_steps`` table as a per-node wall-clock log
    keyed by ``run_id`` so ``run_metrics`` can read persisted per-node timing
    per run (replacing the ``llm_span_seconds`` proxy). ``task_id`` is NULL for
    these timing-only rows (no ``task_executions`` parent); ``step_number=0``
    satisfies the NOT-NULL column. Observability-only: any DB error is logged
    at DEBUG and swallowed (CostTracker-resilience pattern) so timing can never
    break a node.
    """
    try:
        from src.db.models import ExecutionStep
        from src.db.session import get_session
        from src.tools._paths import get_active_run_id

        async with get_session() as session:
            session.add(
                ExecutionStep(
                    step_number=0,
                    phase=node_name,
                    duration_ms=duration_ms,
                    status=status,
                    run_id=get_active_run_id(),
                )
            )
    except Exception as exc:  # noqa: BLE001 — timing must never break a node
        logger.debug("Node-step timing persist skipped for '{}': {}", node_name, exc)


def _folding_cfg_from_settings() -> dict[str, Any]:
    """Build the memory-folding config dict from ``AgentSettings``.

    Maps the ``memory_folding_*`` settings (previously dead — never passed to
    ``MemoryFolder``) into the constructor kwargs the folder expects, plus the
    ``enabled`` flag consumed by ``_check_and_fold``.

    Returns:
        Folding configuration dict.
    """
    agent = get_settings().agent
    return {
        "enabled": agent.memory_folding_enabled,
        "fold_interval": agent.memory_folding_interval,
        "token_threshold": agent.memory_folding_token_threshold,
        "message_count_floor": agent.memory_folding_message_floor,
        "message_count_threshold": agent.memory_folding_message_threshold,
        "message_token_estimate": agent.memory_folding_message_token_estimate,
        "max_folds": agent.memory_folding_max_folds,
    }


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

    # Tool-result cache (Redis, best-effort). Lazily connects on first use,
    # so constructing it here is free and safe even when Redis is down — a
    # miss degrades to a no-op and never breaks a tool call. Only opt-in
    # cacheable tools (web_search, file_reader) are routed through it.
    result_cache = ToolResultCache.from_settings(get_settings())

    # ─── Add Nodes (with dependency injection) ────────────────────────
    # LangGraph's StateNode type is strict about signatures; our closure
    # wrappers match at runtime but Pyright can't verify that statically.
    graph.add_node("classify", _wrap(classify_node, gateway=gateway))  # type: ignore[arg-type]
    # Feature B: ambiguity-resolution cascade between classify and plan.
    # Default-off (clarifying_gate_enabled); route_after_classify returns
    # "plan" unless the gate is on + the goal is ambiguous, so the topology is
    # byte-identical to today until toggled.
    graph.add_node("disambiguate", _wrap(disambiguate_node, gateway=gateway, tools=tools))  # type: ignore[arg-type]
    graph.add_node("plan", _wrap(plan_node, gateway=gateway, tools=tools))  # type: ignore[arg-type]
    graph.add_node("retrieve_memory", _wrap(retrieve_memory_node, memory=memory))  # type: ignore[arg-type]
    graph.add_node("structure_analysis", _wrap(structure_analysis_node, tools=tools, sub_agent_registry=sub_agent_registry, gateway=gateway))  # type: ignore[arg-type]
    graph.add_node("execute", _wrap(execute_node, gateway=gateway, tools=tools, result_cache=result_cache))  # type: ignore[arg-type]
    graph.add_node("reflect", _wrap(reflect_node, gateway=gateway, tools=tools, memory=memory, folding_cfg=_folding_cfg_from_settings()))  # type: ignore[arg-type]
    graph.add_node("verify", _wrap(verify_node, gateway=gateway))  # type: ignore[arg-type]
    graph.add_node("evolve", _wrap(evolve_node, gateway=gateway, tools=tools, sub_agent_registry=sub_agent_registry))  # type: ignore[arg-type]
    graph.add_node("store_memory", _wrap(store_memory_node, memory=memory, gateway=gateway))  # type: ignore[arg-type]
    graph.add_node("tool_create", _wrap(tool_create_node, gateway=gateway, tools=tools))  # type: ignore[arg-type]
    graph.add_node("agent_spawn", _wrap(agent_spawn_node, gateway=gateway, tools=tools, sub_agent_registry=sub_agent_registry))  # type: ignore[arg-type]
    graph.add_node("delegate", _wrap(delegate_node, gateway=gateway, tools=tools, sub_agent_registry=sub_agent_registry, memory=memory))  # type: ignore[arg-type]
    # No deps needed for HITL and error handler
    graph.add_node("hitl_gate", hitl_gate_node)  # type: ignore[arg-type]
    graph.add_node("error_handler", error_handler_node)  # type: ignore[arg-type]
    # G3a LATS/MCTS tree-search (default-off). Explores alternative next-steps in
    # reasoning space on a stalled CRITICAL retry, then hands the chosen branch to
    # execute. Unreachable unless route_after_reflect returns "lats_search"
    # (LATS_ENABLED + CRITICAL + remaining-steps), so the topology is byte-identical
    # until toggled on. lats_search always → execute (single-trajectory execution
    # preserved; LATS only selects the step).
    graph.add_node("lats_search", _wrap(lats_search_node, gateway=gateway, tools=tools))  # type: ignore[arg-type]

    # ─── Linear Edges (START → execute) ────────────────────────────────
    graph.add_edge(START, "classify")
    # Feature B ambiguity gate (default-off). route_after_classify returns
    # "plan" unless clarifying_gate_enabled + an ambiguous goal — in which case
    # classify -> disambiguate -> plan. disambiguate_node is single-shot
    # (disambiguation_done), so no classify↔disambiguate cycle is possible.
    graph.add_conditional_edges("classify", route_after_classify, {
        "plan": "plan",
        "disambiguate": "disambiguate",
    })
    graph.add_edge("disambiguate", "plan")
    graph.add_edge("plan", "retrieve_memory")
    graph.add_edge("retrieve_memory", "structure_analysis")

    # Proactive capability detection before the execute loop: routes to
    # tool_create / agent_spawn when the goal states that intent up front,
    # otherwise proceeds to execute. Runs at most once (structure_analysis_done).
    graph.add_conditional_edges("structure_analysis", route_after_structure_analysis, {
        "agent_spawn": "agent_spawn",
        "tool_create": "tool_create",
        "execute": "execute",
    })

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
        "lats_search": "lats_search",
    })
    # G3a: lats_search commits the best branch (or a no-op pass-through) then
    # always runs execute on the (possibly swapped) current step.
    graph.add_edge("lats_search", "execute")

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
        "plan": "plan",
    })

    graph.add_conditional_edges("evolve", route_after_evolve, {
        "store_memory": "store_memory",
        "error_handler": "error_handler",
        "execute": "execute",
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
        "verify": "verify",
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
