"""Tests for build_subgraph and scope_tools from src.agents.subgraph."""

from __future__ import annotations


import pytest
from unittest.mock import MagicMock

from src.agents.subgraph import build_subgraph, scope_tools
from src.graph.models import SubAgentSpec
from src.tools.registry import ToolRegistry


@pytest.fixture
def parent_tools() -> ToolRegistry:
    """Create a parent ToolRegistry with test tools."""
    tools = ToolRegistry()
    tools.register(
        "search_tool",
        lambda: "search result",
        "Search the web",
        {"query": {"type": "string"}}
    )
    tools.register(
        "calculate_tool",
        lambda: "42",
        "Perform calculations",
        {"expression": {"type": "string"}}
    )
    tools.register(
        "format_tool",
        lambda: "formatted",
        "Format output",
        {"text": {"type": "string"}}
    )
    return tools


@pytest.fixture
def mock_gateway() -> MagicMock:
    """Create a mock LLMGateway."""
    gateway = MagicMock()
    gateway.acompletion = MagicMock()
    return gateway


class TestScopeTools:
    """Tests for scope_tools() function."""

    def test_scope_tools_inherit_all(self, parent_tools: ToolRegistry) -> None:
        """scope_tools with inherit_all copies all tools from parent."""
        spec = SubAgentSpec(
            name="all_inheriting_agent",
            description="Inherits all tools",
            goal="inherit all",
            parent_thread_id="thread-001",
            tool_scope="inherit_all",
        )

        scoped = scope_tools(spec, parent_tools)

        assert scoped.count == 3
        assert scoped.has("search_tool") is True
        assert scoped.has("calculate_tool") is True
        assert scoped.has("format_tool") is True

    def test_scope_tools_inherit_subset(self, parent_tools: ToolRegistry) -> None:
        """scope_tools with inherit_subset copies only named tools."""
        spec = SubAgentSpec(
            name="subset_inheriting_agent",
            description="Inherits subset of tools",
            goal="inherit subset",
            parent_thread_id="thread-002",
            tool_scope="inherit_subset",
            tool_subset=["search_tool", "format_tool"],
        )

        scoped = scope_tools(spec, parent_tools)

        assert scoped.count == 2
        assert scoped.has("search_tool") is True
        assert scoped.has("format_tool") is True
        assert scoped.has("calculate_tool") is False

    def test_scope_tools_self_create(self, parent_tools: ToolRegistry) -> None:
        """scope_tools with self_create returns empty registry."""
        spec = SubAgentSpec(
            name="self_creating_agent",
            description="Creates its own tools",
            goal="self create",
            parent_thread_id="thread-003",
            tool_scope="self_create",
        )

        scoped = scope_tools(spec, parent_tools)

        assert scoped.count == 0
        assert scoped.list_names() == []

    def test_scope_tools_missing_subset_warns(self, parent_tools: ToolRegistry) -> None:
        """scope_tools with inherit_subset warns and skips missing tools."""
        spec = SubAgentSpec(
            name="subset_with_missing_agent",
            description="Requests non-existent tool",
            goal="subset with missing",
            parent_thread_id="thread-004",
            tool_scope="inherit_subset",
            tool_subset=["search_tool", "nonexistent_tool"],
        )

        scoped = scope_tools(spec, parent_tools)

        # Should only have the existing tool
        assert scoped.count == 1
        assert scoped.has("search_tool") is True
        assert scoped.has("nonexistent_tool") is False
        # Note: loguru warnings go to stderr/file, not pytest caplog

    def test_scope_tools_preserves_tool_metadata(self, parent_tools: ToolRegistry) -> None:
        """scope_tools preserves handler, description, and parameters."""
        spec = SubAgentSpec(
            name="metadata_preserving_agent",
            description="Preserves metadata",
            goal="preserve metadata",
            parent_thread_id="thread-005",
            tool_scope="inherit_subset",
            tool_subset=["search_tool"],
        )

        scoped = scope_tools(spec, parent_tools)

        tool = scoped.get("search_tool")
        assert tool is not None
        assert tool["handler"] is not None
        assert tool["description"] == "Search the web"
        assert tool["parameters"] == {"query": {"type": "string"}}


class TestBuildFixedSubgraph:
    """Tests for _build_fixed_subgraph() function."""

    def test_build_fixed_subgraph_creates_four_nodes(self, mock_gateway: MagicMock, parent_tools: ToolRegistry) -> None:
        """build_subgraph with fixed template creates 4-node graph."""
        spec = SubAgentSpec(
            name="fixed_template_agent",
            description="Uses fixed template",
            goal="fixed template",
            parent_thread_id="thread-001",
            tool_scope="inherit_all",
        )

        graph = build_subgraph(spec, mock_gateway, parent_tools)

        # Check that nodes exist
        nodes = graph.nodes
        assert "classify" in nodes
        assert "plan" in nodes
        assert "execute" in nodes
        assert "reflect" in nodes

    def test_build_fixed_subgraph_with_tool_create(self, mock_gateway: MagicMock, parent_tools: ToolRegistry) -> None:
        """build_subgraph adds tool_create when tool_scope is self_create."""
        spec = SubAgentSpec(
            name="self_creating_agent",
            description="Creates its own tools",
            goal="self create",
            parent_thread_id="thread-002",
            tool_scope="self_create",
        )

        graph = build_subgraph(spec, mock_gateway, parent_tools)

        # Should have tool_create node
        nodes = graph.nodes
        assert "tool_create" in nodes

        # Should have all core nodes plus tool_create
        assert "classify" in nodes
        assert "plan" in nodes
        assert "execute" in nodes
        assert "reflect" in nodes


class TestBuildCustomSubgraph:
    """Tests for _build_custom_subgraph() function."""

    def test_build_custom_subgraph_with_valid_config(self, mock_gateway: MagicMock, parent_tools: ToolRegistry) -> None:
        """build_subgraph with custom template uses node_config."""
        spec = SubAgentSpec(
            name="custom_template_agent",
            description="Uses custom template",
            goal="custom template",
            parent_thread_id="thread-001",
            template_type="custom",
            node_config={
                "nodes": ["plan", "execute", "reflect"],
                "edges": [
                    ["START", "plan"],
                    ["plan", "execute"],
                    ["execute", "reflect"],
                    ["reflect", "END"],
                ],
            },
        )

        graph = build_subgraph(spec, mock_gateway, parent_tools)

        # Should have only the specified nodes
        nodes = graph.nodes
        assert "plan" in nodes
        assert "execute" in nodes
        assert "reflect" in nodes
        assert "classify" not in nodes  # Not in config

    def test_build_custom_subgraph_falls_back_on_invalid_config(self, mock_gateway: MagicMock, parent_tools: ToolRegistry) -> None:
        """build_subgraph falls back to fixed template for invalid config."""
        spec = SubAgentSpec(
            name="invalid_custom_agent",
            description="Has invalid config",
            goal="invalid config",
            parent_thread_id="thread-002",
            template_type="custom",
            node_config={},  # Empty config - invalid
        )

        graph = build_subgraph(spec, mock_gateway, parent_tools)

        # Should fall back to fixed template
        nodes = graph.nodes
        assert "classify" in nodes
        assert "plan" in nodes
        assert "execute" in nodes
        assert "reflect" in nodes

    def test_build_custom_subgraph_skips_unknown_nodes(self, mock_gateway: MagicMock, parent_tools: ToolRegistry) -> None:
        """build_subgraph skips unknown node names in config."""
        spec = SubAgentSpec(
            name="unknown_nodes_agent",
            description="References unknown nodes",
            goal="unknown nodes",
            parent_thread_id="thread-003",
            template_type="custom",
            node_config={
                "nodes": ["plan", "unknown_node", "execute"],
                "edges": [
                    ["START", "plan"],
                    ["plan", "execute"],
                    ["execute", "END"],
                ],
            },
        )

        graph = build_subgraph(spec, mock_gateway, parent_tools)

        # Should have only the valid nodes
        nodes = graph.nodes
        assert "plan" in nodes
        assert "execute" in nodes
        assert "unknown_node" not in nodes


class TestBuildSubgraphIntegration:
    """Integration tests for build_subgraph."""

    def test_build_subgraph_scopes_tools_before_building(self, mock_gateway: MagicMock, parent_tools: ToolRegistry) -> None:
        """build_subgraph scopes tools before passing to nodes."""
        spec = SubAgentSpec(
            name="scoping_agent",
            description="Tests tool scoping",
            goal="test scoping",
            parent_thread_id="thread-001",
            tool_scope="inherit_subset",
            tool_subset=["search_tool"],
        )

        graph = build_subgraph(spec, mock_gateway, parent_tools)

        # Graph should be built successfully
        assert graph is not None
        assert "plan" in graph.nodes
        assert "execute" in graph.nodes

    def test_build_subgraph_compiles_successfully(self, mock_gateway: MagicMock, parent_tools: ToolRegistry) -> None:
        """build_subgraph produces a compilable StateGraph."""
        spec = SubAgentSpec(
            name="compilable_agent",
            description="Compiles successfully",
            goal="test compilation",
            parent_thread_id="thread-001",
            tool_scope="inherit_all",
        )

        graph = build_subgraph(spec, mock_gateway, parent_tools)

        # Should be compilable without errors
        compiled = graph.compile()
        assert compiled is not None

    def test_build_subgraph_with_memory(self, mock_gateway: MagicMock, parent_tools: ToolRegistry) -> None:
        """build_subgraph accepts memory parameter."""
        from unittest.mock import MagicMock

        spec = SubAgentSpec(
            name="memory_agent",
            description="Uses memory",
            goal="test memory",
            parent_thread_id="thread-001",
            tool_scope="inherit_all",
        )

        memory = MagicMock()
        graph = build_subgraph(spec, mock_gateway, parent_tools, memory)

        assert graph is not None
        assert "plan" in graph.nodes

    def test_build_subgraph_with_budget(self, mock_gateway: MagicMock, parent_tools: ToolRegistry) -> None:
        """build_subgraph accepts budget_remaining parameter."""
        spec = SubAgentSpec(
            name="budget_agent",
            description="Uses budget",
            goal="test budget",
            parent_thread_id="thread-001",
            tool_scope="inherit_all",
        )

        graph = build_subgraph(spec, mock_gateway, parent_tools, budget_remaining=1.0)

        assert graph is not None
        assert "plan" in graph.nodes


class TestRoutingFunctions:
    """Tests for internal routing functions."""

    def test_route_after_execute_sub_max_iterations(self) -> None:
        """_route_after_execute_sub routes to reflect on max iterations."""
        from src.agents.subgraph import _route_after_execute_sub

        state = {
            "errors": [],
            "iteration_count": 10,
            "max_iterations": 10,
        }
        result = _route_after_execute_sub(state)
        assert result == "reflect"

    def test_route_after_execute_sub_on_errors(self) -> None:
        """_route_after_execute_sub routes to reflect on errors."""
        from src.agents.subgraph import _route_after_execute_sub

        state = {
            "errors": ["some error"],
            "iteration_count": 5,
            "max_iterations": 10,
        }
        result = _route_after_execute_sub(state)
        assert result == "reflect"

    def test_route_after_reflect_sub_tool_gaps(self) -> None:
        """_route_after_reflect_sub routes to tool_create on gaps."""
        from src.agents.subgraph import _route_after_reflect_sub

        state = {
            "pending_tool_gaps": ["missing_tool"],
            "confidence": "high",
        }
        result = _route_after_reflect_sub(state)
        assert result == "tool_create"

    def test_route_after_reflect_sub_low_confidence(self) -> None:
        """_route_after_reflect_sub routes to execute on low confidence."""
        from src.agents.subgraph import _route_after_reflect_sub

        state = {
            "pending_tool_gaps": [],
            "confidence": "low",
        }
        result = _route_after_reflect_sub(state)
        assert result == "execute"

    def test_route_after_reflect_sub_high_confidence(self) -> None:
        """_route_after_reflect_sub routes to END on high confidence."""
        from src.agents.subgraph import _route_after_reflect_sub

        state = {
            "pending_tool_gaps": [],
            "confidence": "high",
        }
        result = _route_after_reflect_sub(state)
        assert result == "__end__"

    def test_route_after_tool_create_sub_with_tools(self) -> None:
        """_route_after_tool_create_sub routes to plan when tools created."""
        from src.agents.subgraph import _route_after_tool_create_sub

        state = {
            "tools_created": ["new_tool"],
        }
        result = _route_after_tool_create_sub(state)
        assert result == "plan"

    def test_route_after_tool_create_sub_no_tools(self) -> None:
        """_route_after_tool_create_sub routes to execute when no tools created."""
        from src.agents.subgraph import _route_after_tool_create_sub

        state = {
            "tools_created": [],
        }
        result = _route_after_tool_create_sub(state)
        assert result == "execute"
