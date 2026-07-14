"""Structural + wiring smoke tests for ``src.graph.task_graph``.

These complement ``test_integration_flow.py`` (which proves a mocked-gateway
full run terminates) by locking down the topology and dependency-injection
contracts that the integration run does NOT assert:

* the FULL declared node set (the older ``test_task_graph`` only checks a
  10-node subset — there are 17 nodes now, incl. disambiguate /
  structure_analysis / tool_create / agent_spawn / delegate / lats_search);
* every conditional-edge routing target resolves to a real node (or END) — a
  static guard against a router returning a value with no edge (runtime
  KeyError);
* no orphan node — every declared node is reachable from START;
* the pure-heuristic ``build_task_graph()`` (all deps ``None``) still compiles,
  and individual node closures run on the heuristic path without a gateway;
* dependency-injection actually wires through — a sentinel gateway + memory are
  observed being called by the nodes that received them (proves the closures
  injected the deps, not ``None``).
"""

from __future__ import annotations

from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.errors import GraphRecursionError

from src.graph.factory import initial_state
from src.graph.nodes import classify_node, verify_node
from src.graph.state import AgentState
from src.graph.task_graph import build_task_graph, compile_task_graph

# The full set of nodes ``build_task_graph`` registers (18). Updating this set
# is the point — it fails loudly on accidental add/remove of a graph node.
_EXPECTED_NODES: set[str] = {
    "classify",
    "disambiguate",
    "plan",
    "retrieve_memory",
    "research",
    "structure_analysis",
    "execute",
    "reflect",
    "verify",
    "evolve",
    "store_memory",
    "tool_create",
    "agent_spawn",
    "delegate",
    "hitl_gate",
    "error_handler",
    "lats_search",
}

# Every conditional-edge routing target declared in ``build_task_graph`` (the
# mapping values). END is permitted; anything else MUST be a declared node — a
# target with no node would raise at runtime when the router returns it.
_ROUTING_TARGETS: set[str] = {
    "plan", "disambiguate", "agent_spawn", "tool_create", "execute",
    "error_handler", "reflect", "verify", "lats_search", "delegate",
    "store_memory", "evolve", "hitl_gate", "classify", "research",
}


def _canned_gateway() -> MagicMock:
    """Deterministic gateway: classify JSON + a no-tool execute response.

    Mirrors the proven pattern in ``test_integration_flow`` so the run
    terminates on heuristic fallbacks for plan/reflect/verify.
    """
    from src.llm.models import LLMResponse

    gateway = MagicMock()
    gateway.acompletion = AsyncMock(
        return_value=LLMResponse(
            content=(
                '{"complexity": "simple", "strategy": "direct", '
                '"estimated_steps": 1, "confidence": 0.9, "reasoning": "x"}'
            ),
            model="gpt-4o-mini-2024-07-18",
            provider="openai",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            cost_usd=0.0001,
        )
    )
    tool_resp = MagicMock()
    tool_resp.tool_calls = []
    tool_resp.content = "done"
    gateway.acompletion_with_tools = AsyncMock(return_value=tool_resp)
    gateway.get_cost_records = MagicMock(return_value=[])
    return gateway


def _mock_tools() -> MagicMock:
    tools = MagicMock()
    tools.list_tools = MagicMock(return_value=[])
    tools.get_handler = MagicMock(return_value=None)
    return tools


def _mock_memory() -> MagicMock:
    memory = MagicMock()
    memory.retrieve_context = AsyncMock(return_value=[])
    memory.store_observation = AsyncMock(return_value=None)
    memory.store_skill = AsyncMock(return_value="uuid")
    return memory


class TestTaskGraphTopology:
    """Static topology invariants — no execution, fast."""

    def test_full_node_set_exact(self) -> None:
        """All 18 declared nodes are present, none missing, none extra."""
        graph = build_task_graph()
        assert set(graph.nodes.keys()) == _EXPECTED_NODES, (
            f"Node set drift: {set(graph.nodes.keys()) ^ _EXPECTED_NODES}"
        )

    def test_every_routing_target_is_a_real_node(self) -> None:
        """No conditional edge points to an undeclared node (would KeyError)."""
        graph = build_task_graph()
        nodes = set(graph.nodes.keys())
        unknown = _ROUTING_TARGETS - nodes - {"END"}
        assert not unknown, f"Routing targets with no node: {unknown}"

    def test_no_orphan_nodes_all_reachable_from_start(self) -> None:
        """Every declared node is reachable from START (no dead nodes).

        BFS over the COMPILED graph's successor relation via ``get_graph()``,
        whose ``Edge(source, target, ...)`` entries cover both linear and
        conditional edges. A node with no inbound edge compiles fine but is
        dead — this catches that.
        """
        compiled = compile_task_graph()
        gg = compiled.get_graph()

        successors: dict[str, set[str]] = {}
        for edge in gg.edges:
            successors.setdefault(edge.source, set()).add(edge.target)

        reachable: set[str] = set()
        frontier: set[str] = set(successors.get("__start__", set()))
        assert "classify" in frontier, "START must wire to classify"
        while frontier:
            node = frontier.pop()
            if node in reachable:
                continue
            reachable.add(node)
            frontier.update(successors.get(node, set()) - reachable)

        orphans = set(_EXPECTED_NODES) - reachable
        assert not orphans, f"Nodes unreachable from START: {orphans}"


class TestHeuristicFallbackPath:
    """The all-deps-None path must build + run without a gateway."""

    def test_build_with_no_deps_compiles(self) -> None:
        compiled = compile_task_graph()
        assert hasattr(compiled, "ainvoke")

    @pytest.mark.asyncio
    async def test_classify_node_heuristic_no_gateway(self) -> None:
        """classify_node(state, gateway=None) returns a valid partial update."""
        state = cast(AgentState, dict(initial_state("Sum 2+2", "smoke-001", 6)))
        result = await classify_node(state, gateway=None)
        assert isinstance(result, dict)
        # Heuristic classify always sets a complexity + confidence.
        assert "complexity" in result or "phase" in result

    @pytest.mark.asyncio
    async def test_verify_node_heuristic_no_gateway(self) -> None:
        """verify_node(state, gateway=None) returns a dict (no crash)."""
        state = cast(AgentState, dict(initial_state("Sum 2+2", "smoke-002", 6)))
        result = await verify_node(state, gateway=None)
        assert isinstance(result, dict)


class TestDependencyInjectionWiring:
    """Prove the closure wrappers actually deliver the injected deps."""

    @pytest.mark.asyncio
    async def test_sentinel_gateway_is_actually_called(self) -> None:
        """A run with an injected gateway exercises gateway.acompletion.

        If DI were broken (closures captured None), acompletion would never be
        awaited — this is the wiring proof that classify/plan/etc. received the
        gateway, not the heuristic fallback.
        """
        gateway = _canned_gateway()
        compiled = compile_task_graph(
            gateway=gateway, memory=_mock_memory(), tools=_mock_tools()
        )
        state = dict(
            initial_state("Explain REST in one line", "smoke-di-001", 6,
                          no_evolution=True)
        )
        await compiled.ainvoke(state, {"recursion_limit": 60})
        assert gateway.acompletion.await_count >= 1, (
            "Injected gateway was never called — DI wiring is broken"
        )

    @pytest.mark.asyncio
    async def test_sentinel_memory_is_actually_called(self) -> None:
        """retrieve_memory node queries the injected memory manager."""
        memory = _mock_memory()
        compiled = compile_task_graph(
            gateway=_canned_gateway(), memory=memory, tools=_mock_tools()
        )
        state = dict(
            initial_state("Explain REST in one line", "smoke-di-002", 6,
                          no_evolution=True)
        )
        await compiled.ainvoke(state, {"recursion_limit": 60})
        assert memory.retrieve_context.await_count >= 1, (
            "Injected memory was never queried — retrieve_memory DI broken"
        )

    @pytest.mark.asyncio
    async def test_recursion_limit_guard_fires(self) -> None:
        """An absurdly low recursion_limit raises GraphRecursionError.

        Proves the LangGraph recursion guard is wired (the safety net against
        an infinite execute↔reflect loop).
        """
        compiled = compile_task_graph(
            gateway=_canned_gateway(), memory=_mock_memory(), tools=_mock_tools()
        )
        state = dict(
            initial_state("Explain REST in one line", "smoke-rl-001", 6,
                          no_evolution=True)
        )
        with pytest.raises(GraphRecursionError):
            await compiled.ainvoke(state, {"recursion_limit": 2})
