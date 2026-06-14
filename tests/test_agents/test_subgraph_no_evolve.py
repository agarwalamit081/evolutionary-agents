"""Sub-agent subgraphs must never contain an evolve node (F13 §4 guard).

Sub-agents are delegated work units; they must not self-evolve (mutating the
parent's evolution repo from a delegated context is out of scope). Both the
fixed and custom subgraph builders assert this invariant after construction,
and the custom builder's allowlist excludes ``evolve`` so a rogue node_config
cannot inject it.
"""

from __future__ import annotations

from unittest.mock import MagicMock

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
    gateway = MagicMock()
    gateway.acompletion = MagicMock()
    return gateway


class TestSubgraphNoEvolve:
    """Both fixed and custom sub-agent subgraphs omit the evolve node."""

    def test_fixed_subgraph_has_no_evolve_node(self) -> None:
        spec = SubAgentSpec(
            name="fixed_agent",
            description="Fixed template sub-agent",
            goal="do the thing",
            parent_thread_id="thread-001",
            tool_scope="inherit_all",
        )
        graph = build_subgraph(spec, _gateway(), _parent_tools())
        assert "evolve" not in graph.nodes
        # Sanity: the core fixed-template nodes ARE present.
        assert "classify" in graph.nodes
        assert "reflect" in graph.nodes

    def test_custom_subgraph_has_no_evolve_node(self) -> None:
        spec = SubAgentSpec(
            name="custom_agent",
            description="Custom template sub-agent",
            goal="do the thing",
            parent_thread_id="thread-002",
            tool_scope="inherit_all",
            template_type="custom",
            node_config={
                "nodes": ["classify", "plan", "execute", "reflect"],
                "edges": [
                    ["START", "classify"],
                    ["classify", "plan"],
                    ["plan", "execute"],
                    ["execute", "reflect"],
                ],
                "conditional_edges": {},
            },
        )
        graph = build_subgraph(spec, _gateway(), _parent_tools())
        assert "evolve" not in graph.nodes
        assert "reflect" in graph.nodes

    def test_custom_subgraph_rejects_evolve_in_node_config(self) -> None:
        """A custom node_config naming 'evolve' is skipped (not in the allowlist),
        so the invariant still holds."""
        spec = SubAgentSpec(
            name="rogue_agent",
            description="Tries to add evolve",
            goal="self-evolve",
            parent_thread_id="thread-003",
            tool_scope="inherit_all",
            template_type="custom",
            node_config={
                "nodes": ["classify", "plan", "execute", "reflect", "evolve"],
                "edges": [["START", "classify"]],
                "conditional_edges": {},
            },
        )
        graph = build_subgraph(spec, _gateway(), _parent_tools())
        assert "evolve" not in graph.nodes
