"""Dynamic LangGraph subgraph construction for sub-agents.

Builds a StateGraph from a SubAgentSpec definition. Supports two modes:
  - Fixed template: classify → plan → execute ↔ reflect → END
  - Custom template: nodes and edges defined by node_config

Reuses existing node functions from src/graph/nodes/ with the same
_wrap() closure injection pattern as build_task_graph().
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph
from loguru import logger

from src.agents.state import SubAgentState
from src.graph.enums import Confidence
from src.graph.models import SubAgentSpec
from src.graph.routers import route_after_execute

if TYPE_CHECKING:
    from src.llm.gateway import LLMGateway
    from src.llm.models import LLMResponse
    from src.memory.manager import MemoryManager
    from src.tools.registry import ToolRegistry


# ── Model Override Proxy ─────────────────────────────────────────────────


class _ModelOverrideProxy:
    """Thin wrapper around LLMGateway that forces a specific model.

    Used when spawning parallel sub-agents with diverse model assignments.
    All ``acompletion`` and ``acompletion_with_tools`` calls are routed
    through the real gateway but with the ``model`` parameter forced.

    Other attributes (cost_tracker, cache, etc.) are delegated to the
    underlying gateway via ``__getattr__``.
    """

    def __init__(self, gateway: LLMGateway, model: str) -> None:
        self._gateway = gateway
        self._model = model

    async def acompletion(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        """Forward to gateway with forced model."""
        return await self._gateway.acompletion(messages=messages, model=self._model, **kwargs)

    async def acompletion_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], **kwargs: Any) -> Any:
        """Forward to gateway with forced model."""
        return await self._gateway.acompletion_with_tools(
            messages=messages, tools=tools, model=self._model, **kwargs
        )

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attributes to the real gateway."""
        return getattr(self._gateway, name)


# ── Closure Wrapper (mirrors src/graph/task_graph.py:_wrap) ─────────────


def _wrap(
    node_fn: Callable[..., Awaitable[dict[str, Any]]],
    **deps: Any,
) -> Callable[[SubAgentState], Awaitable[dict[str, Any]]]:
    """Wrap a node function, injecting keyword dependencies via closure."""

    async def _wrapped(state: SubAgentState) -> dict[str, Any]:
        return await node_fn(state, **deps)  # type: ignore[arg-type]

    _wrapped.__name__ = node_fn.__name__
    _wrapped.__doc__ = node_fn.__doc__
    return _wrapped


# ── Tool Scoping ────────────────────────────────────────────────────────


def scope_tools(
    spec: SubAgentSpec,
    parent_tools: ToolRegistry,
) -> ToolRegistry:
    """Create a scoped ToolRegistry for the sub-agent.

    Args:
        spec: Sub-agent specification with tool_scope and tool_subset.
        parent_tools: Parent agent's full ToolRegistry.

    Returns:
        Scoped ToolRegistry with the appropriate subset of tools.
    """
    from src.tools.registry import ToolRegistry

    scoped = ToolRegistry()

    if spec.tool_scope == "inherit_all":
        # Copy all tools from parent
        for name in parent_tools.list_names():
            tool = parent_tools.get(name)
            if tool:
                scoped.register(
                    name, tool["handler"], tool["description"], tool["parameters"]
                )

    elif spec.tool_scope == "inherit_subset":
        # Only copy named tools
        for name in spec.tool_subset:
            tool = parent_tools.get(name)
            if tool:
                scoped.register(
                    name, tool["handler"], tool["description"], tool["parameters"]
                )
            else:
                logger.warning(
                    f"Sub-agent '{spec.name}': requested tool '{name}' "
                    f"not found in parent registry, skipping"
                )

    # "self_create": empty registry — tool_create node will populate it
    return scoped


# ── Subgraph Routing ────────────────────────────────────────────────────


def _route_after_execute_sub(state: SubAgentState) -> str:
    """Route after execute in sub-agent subgraph.

    Simplified from the main graph: no error_handler, just loop or reflect.
    """
    errors = state.get("errors", [])
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 10)

    # Max iterations → reflect
    if iteration_count >= max_iterations:
        return "reflect"

    # Errors → reflect (let it assess)
    if errors:
        return "reflect"

    # Use main router for normal flow
    return route_after_execute(state)  # type: ignore[arg-type]


def _route_after_reflect_sub(state: SubAgentState) -> str:
    """Route after reflect in sub-agent subgraph.

    Checks for tool gaps first (if tool_create is available),
    then decides based on confidence.
    """
    # Tool gaps → tool_create (only if available in subgraph)
    pending_gaps = state.get("pending_tool_gaps", [])  # type: ignore[typeddict-item]
    if pending_gaps:
        return "tool_create"

    confidence = state.get("confidence", Confidence.MEDIUM)
    if isinstance(confidence, str):
        confidence = Confidence(confidence)

    # Low confidence → retry execute
    if confidence in {Confidence.VERY_LOW, Confidence.LOW}:
        return "execute"

    # Done → END
    return "__end__"


def _route_after_tool_create_sub(state: SubAgentState) -> str:
    """Route after tool_create in sub-agent subgraph."""
    tools_created = state.get("tools_created", [])  # type: ignore[typeddict-item]
    if tools_created:
        return "plan"
    return "execute"


# ── Fixed Template Builder ──────────────────────────────────────────────


def _build_fixed_subgraph(
    spec: SubAgentSpec,
    gateway: LLMGateway | _ModelOverrideProxy,
    tools: ToolRegistry,
) -> StateGraph:
    """Build fixed template: classify → plan → execute ↔ reflect → END.

    If tool_scope is "self_create", also adds tool_create node.
    """
    # Lazy imports to avoid circular dependency (src.graph.nodes → src.agents → src.agents.subgraph)
    from src.graph.nodes import classify_node, execute_node, plan_node, reflect_node

    graph = StateGraph(SubAgentState)

    # Add core nodes
    graph.add_node("classify", _wrap(classify_node, gateway=gateway))  # type: ignore[arg-type]
    graph.add_node("plan", _wrap(plan_node, gateway=gateway, tools=tools))  # type: ignore[arg-type]
    graph.add_node("execute", _wrap(execute_node, gateway=gateway, tools=tools))  # type: ignore[arg-type]
    graph.add_node("reflect", _wrap(reflect_node, gateway=gateway, tools=tools))  # type: ignore[arg-type]

    # Linear edges
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "plan")
    graph.add_edge("plan", "execute")

    # Conditional: execute loop
    # Always enable tool creation for sub-agents — they can create tools
    # regardless of whether they inherit parent tools or start empty
    has_tool_create = True
    execute_targets = {
        "reflect": "reflect",
        "execute": "execute",
    }

    graph.add_conditional_edges("execute", _route_after_execute_sub, execute_targets)  # type: ignore[arg-type]

    # Conditional: reflect → END or tool_create or execute
    reflect_targets = {
        "__end__": END,
        "execute": "execute",
    }

    if has_tool_create:
        from src.graph.nodes import tool_create_node

        graph.add_node(
            "tool_create",
            _wrap(tool_create_node, gateway=gateway, tools=tools),  # type: ignore[arg-type]
        )
        reflect_targets["tool_create"] = "tool_create"
        graph.add_conditional_edges(
            "tool_create",
            _route_after_tool_create_sub,
            {"plan": "plan", "execute": "execute"},
        )  # type: ignore[arg-type]

    graph.add_conditional_edges("reflect", _route_after_reflect_sub, reflect_targets)  # type: ignore[arg-type]

    # Guard: sub-agents never self-evolve — the fixed subgraph terminates at
    # reflect → END with no evolve node (F13 §4). Evolving a sub-agent would
    # mutate the parent's evolution repo from a delegated context, which is
    # explicitly out of scope (max 3 sub-agents, no sub-agent self-evolution).
    assert "evolve" not in graph.nodes, "sub-agent subgraph must not contain evolve"
    return graph


# ── Custom Template Builder ─────────────────────────────────────────────

# Available node names for custom templates (lazy-loaded to avoid circular imports).
# NOTE: "evolve" is deliberately absent — sub-agents never self-evolve (F13 §4).
# A custom node_config requesting "evolve" is rejected at _get_node_function /
# the unknown-node skip in _build_custom_subgraph, and both builders assert the
# invariant after construction.
_AVAILABLE_NODE_NAMES: list[str] = [
    "classify", "plan", "execute", "reflect", "tool_create",
]


def _get_node_function(name: str) -> Callable[..., Any]:
    """Lazy-load a node function to avoid circular imports."""
    from src.graph.nodes import (
        classify_node,
        execute_node,
        plan_node,
        reflect_node,
        tool_create_node,
    )
    nodes = {
        "classify": classify_node,
        "plan": plan_node,
        "execute": execute_node,
        "reflect": reflect_node,
        "tool_create": tool_create_node,
    }
    return nodes[name]


def _build_custom_subgraph(
    spec: SubAgentSpec,
    gateway: LLMGateway | _ModelOverrideProxy,
    tools: ToolRegistry,
    _memory: MemoryManager | None = None,
) -> StateGraph:
    """Build custom template from node_config.

    note: ``_memory`` is accepted for API symmetry but not yet wired into
    custom node composition.  Reserved for a future enhancement.

    node_config format:
        {
            "nodes": ["classify", "plan", "execute", "reflect"],
            "edges": [
                ["START", "classify"],
                ["classify", "plan"],
                ["plan", "execute"],
                ...
            ],
            "conditional_edges": {
                "execute": {"router": "execute_loop", "targets": {"reflect": "reflect", "execute": "execute"}},
                ...
            }
        }

    Falls back to fixed template if config is invalid.
    """
    config = spec.node_config
    if not config or "nodes" not in config:
        logger.warning(
            f"Sub-agent '{spec.name}': invalid node_config, "
            f"falling back to fixed template"
        )
        return _build_fixed_subgraph(spec, gateway, tools)

    graph = StateGraph(SubAgentState)

    # Add requested nodes
    node_names: list[str] = []
    for node_name in config["nodes"]:
        if node_name not in _AVAILABLE_NODE_NAMES:
            logger.warning(
                f"Sub-agent '{spec.name}': unknown node '{node_name}', skipping"
            )
            continue

        deps: dict[str, Any] = {"gateway": gateway}
        if node_name in ("plan", "execute", "reflect", "tool_create"):
            deps["tools"] = tools

        node_fn = _get_node_function(node_name)
        graph.add_node(node_name, _wrap(node_fn, **deps))  # type: ignore[arg-type]
        node_names.append(node_name)

    if not node_names:
        logger.warning(
            f"Sub-agent '{spec.name}': no valid nodes in config, "
            f"falling back to fixed template"
        )
        return _build_fixed_subgraph(spec, gateway, tools)

    # Add static edges
    for edge in config.get("edges", []):
        if len(edge) != 2:
            continue
        src, dst = edge
        if src == "START":
            graph.add_edge(START, dst)
        elif dst == "END":
            graph.add_edge(src, END)
        else:
            graph.add_edge(src, dst)

    # Add conditional edges
    for src, cfg in config.get("conditional_edges", {}).items():
        targets = cfg.get("targets", {})
        # Map string router names to actual functions
        router_name = cfg.get("router", "")
        router_fn = _get_subgraph_router(router_name)
        graph.add_conditional_edges(src, router_fn, targets)

    # Validate graph has at least START → something
    if not node_names:
        return _build_fixed_subgraph(spec, gateway, tools)

    # Guard: sub-agents never self-evolve (F13 §4). _AVAILABLE_NODE_NAMES
    # excludes "evolve", so a well-formed custom config can't add it — this
    # assert makes the invariant explicit and fails loudly if that ever changes.
    assert "evolve" not in graph.nodes, "sub-agent subgraph must not contain evolve"
    return graph


def _get_subgraph_router(name: str) -> Callable[..., str]:
    """Get a router function by name for custom subgraphs."""
    routers = {
        "execute_loop": _route_after_execute_sub,
        "reflect": _route_after_reflect_sub,
        "tool_create": _route_after_tool_create_sub,
    }
    return routers.get(name, _route_after_reflect_sub)


# ── Public API ──────────────────────────────────────────────────────────


def build_subgraph(
    spec: SubAgentSpec,
    gateway: LLMGateway,
    tools: ToolRegistry,
    memory: MemoryManager | None = None,
    budget_remaining: float | None = None,
    preferred_model: str = "",
) -> StateGraph:
    """Build a LangGraph subgraph from a SubAgentSpec definition.

    Args:
        spec: Sub-agent specification with template_type, tool_scope, etc.
        gateway: LLMGateway for LLM calls within sub-agent.
        tools: Parent's ToolRegistry (will be scoped via scope_tools()).
        memory: Optional MemoryManager (isolated for sub-agent).
        budget_remaining: Remaining budget for shared mode.
        preferred_model: When set, wraps gateway in a proxy that forces
            this model for all LLM calls (used for diverse sub-agent routing).

    Returns:
        StateGraph ready for compilation and execution.
    """
    # Scope tools
    scoped_tools = scope_tools(spec, tools)

    # Wrap gateway with model override when a preferred model is specified
    effective_gateway: LLMGateway | _ModelOverrideProxy = gateway
    if preferred_model:
        effective_gateway = _ModelOverrideProxy(gateway, preferred_model)
        logger.debug(
            f"Sub-agent '{spec.name}' using model override: {preferred_model}"
        )

    if spec.template_type == "custom" and spec.node_config:
        return _build_custom_subgraph(spec, effective_gateway, scoped_tools, memory)

    return _build_fixed_subgraph(spec, effective_gateway, scoped_tools)
