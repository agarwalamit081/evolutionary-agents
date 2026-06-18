"""Sub-agent read-only long-term memory recall (2d).

When a ``MemoryManager`` is wired into ``build_subgraph``, a leading
``retrieve`` node recalls warm+cold context for the sub-agent's goal and seeds
``retrieved_memories`` into state (plan/execute inject it as context). Recall is
READ-ONLY — a delegated task never persists to long-term memory, so it cannot
mutate shared memory as a side effect (the isolation contract).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.state import initial_sub_agent_state
from src.agents.subgraph import build_subgraph
from src.graph.models import SubAgentSpec
from src.tools.registry import ToolRegistry


def _parent_tools() -> ToolRegistry:
    tools = ToolRegistry()
    tools.register(
        "search_tool", lambda: "x", "Search the web", {"query": {"type": "string"}}
    )
    return tools


def _gateway() -> MagicMock:
    gw = MagicMock()
    gw.acompletion = MagicMock()
    return gw


def _seeded_memory(content: str = "seeded warm skill: normalize csv") -> MagicMock:
    """MemoryManager mock whose retrieve_context returns one seeded memory."""
    memory = MagicMock()
    memory.retrieve_context = AsyncMock(
        return_value=[{"content": content, "tier": "warm", "score": 0.9}]
    )
    memory.warm = MagicMock()
    memory.warm.retrieve = AsyncMock(return_value=[])
    return memory


def _fixed_spec(name: str = "fixed_agent") -> SubAgentSpec:
    return SubAgentSpec(
        name=name,
        description="Fixed template sub-agent",
        goal="normalize a csv",
        parent_thread_id="thread-001",
        tool_scope="inherit_all",
    )


class TestSubgraphMemoryRecall:
    """Wiring + recall for the read-only sub-agent memory retrieve node (2d)."""

    def test_fixed_subgraph_adds_retrieve_when_memory_wired(self) -> None:
        """A fixed subgraph built WITH memory has a leading retrieve node."""
        graph = build_subgraph(
            _fixed_spec(), _gateway(), _parent_tools(), memory=_seeded_memory()
        )
        assert "retrieve" in graph.nodes
        # The core nodes are still present.
        assert "classify" in graph.nodes and "plan" in graph.nodes

    def test_fixed_subgraph_has_no_retrieve_when_memory_absent(self) -> None:
        """Without memory the topology is unchanged (no retrieve node)."""
        graph = build_subgraph(_fixed_spec(), _gateway(), _parent_tools())
        assert "retrieve" not in graph.nodes
        assert "classify" in graph.nodes  # original entrypoint intact

    def test_custom_subgraph_wires_retrieve_with_memory(self) -> None:
        """A custom config listing 'retrieve' wires it when memory is supplied."""
        spec = SubAgentSpec(
            name="custom_agent",
            description="Custom template sub-agent",
            goal="normalize a csv",
            parent_thread_id="thread-002",
            tool_scope="inherit_all",
            template_type="custom",
            node_config={
                "nodes": ["retrieve", "classify", "plan", "execute", "reflect"],
                "edges": [
                    ["START", "retrieve"],
                    ["retrieve", "classify"],
                    ["classify", "plan"],
                    ["plan", "execute"],
                    ["execute", "reflect"],
                ],
                "conditional_edges": {},
            },
        )
        graph = build_subgraph(
            spec, _gateway(), _parent_tools(), memory=_seeded_memory()
        )
        assert "retrieve" in graph.nodes

    def test_custom_subgraph_retrieve_without_memory_does_not_raise(self) -> None:
        """A 'retrieve' node with no memory degrades to a no-op recall."""
        spec = SubAgentSpec(
            name="memoryless",
            description="No memory wired",
            goal="normalize a csv",
            parent_thread_id="thread-003",
            tool_scope="inherit_all",
            template_type="custom",
            node_config={
                "nodes": ["retrieve", "classify", "plan"],
                "edges": [["START", "retrieve"], ["retrieve", "classify"]],
                "conditional_edges": {},
            },
        )
        # Must not raise even though memory is None.
        graph = build_subgraph(spec, _gateway(), _parent_tools())
        assert "retrieve" in graph.nodes

    @pytest.mark.asyncio
    async def test_wired_retrieve_node_recalls_seeded_skill(self) -> None:
        """The wired retrieve node recalls a seeded warm skill into state.

        This is the node-level 'a sub-agent run retrieves a seeded warm skill':
        the retrieve node produced by ``build_subgraph`` is invoked against a
        seeded memory and surfaces the skill in ``retrieved_memories``.
        """
        memory = _seeded_memory("seeded warm skill: normalize csv")
        graph = build_subgraph(_fixed_spec("recaller"), _gateway(), _parent_tools(), memory=memory)
        assert "retrieve" in graph.nodes

        state = initial_sub_agent_state("normalize a csv", "thread-001")
        # langgraph's runnable is a RunnableCallable at runtime; the typed union
        # doesn't expose .ainvoke, so cast to Any.
        retrieve_runnable: Any = graph.nodes["retrieve"].runnable
        result = await retrieve_runnable.ainvoke(state)

        memories = result["retrieved_memories"]
        assert memories, "retrieve node seeded no memories"
        assert "seeded warm skill" in memories[0]["content"]
        memory.retrieve_context.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retrieve_is_read_only_no_writes(self) -> None:
        """The wired retrieve node never calls a write method on memory."""
        memory = _seeded_memory()
        # Surface any write-like method the manager might expose.
        memory.store = MagicMock()
        memory.add = MagicMock()
        memory.warm.store = MagicMock()
        graph = build_subgraph(_fixed_spec("readonly"), _gateway(), _parent_tools(), memory=memory)

        state = initial_sub_agent_state("normalize a csv", "thread-001")
        retrieve_runnable: Any = graph.nodes["retrieve"].runnable
        await retrieve_runnable.ainvoke(state)

        memory.store.assert_not_called()
        memory.add.assert_not_called()
        memory.warm.store.assert_not_called()
