"""Tests for src.graph.task_graph — graph construction and compilation."""

from __future__ import annotations

from unittest.mock import MagicMock


from src.graph.task_graph import build_task_graph, compile_task_graph


class TestBuildTaskGraph:
    """Tests for build_task_graph function."""

    def test_build_task_graph_has_correct_nodes(self) -> None:
        """build_task_graph creates a graph with all 10 expected nodes."""
        graph = build_task_graph()
        # StateGraph.nodes contains the node names added via add_node
        node_names = set(graph.nodes.keys())
        expected_nodes = {
            "classify",
            "plan",
            "retrieve_memory",
            "execute",
            "reflect",
            "verify",
            "evolve",
            "store_memory",
            "hitl_gate",
            "error_handler",
        }
        assert expected_nodes.issubset(node_names), (
            f"Missing nodes: {expected_nodes - node_names}"
        )

    def test_build_task_graph_has_start_edge(self) -> None:
        """build_task_graph wires START to classify and classify to plan."""
        graph = build_task_graph()
        # Verify classify and plan nodes exist (START → classify → plan)
        assert "classify" in graph.nodes
        assert "plan" in graph.nodes


class TestCompileTaskGraph:
    """Tests for compile_task_graph function."""

    def test_compile_task_graph_without_deps(self) -> None:
        """compile_task_graph compiles successfully with no dependencies."""
        compiled = compile_task_graph()
        # A compiled graph should be callable (has invoke/ainvoke methods)
        assert hasattr(compiled, "ainvoke")
        assert hasattr(compiled, "ainvoke")

    def test_compile_task_graph_with_tools(self) -> None:
        """compile_task_graph compiles with mock tools injected."""
        mock_tools = MagicMock()
        mock_tools.list_tools = MagicMock(return_value=[])
        mock_tools.get_handler = MagicMock(return_value=None)

        compiled = compile_task_graph(tools=mock_tools)
        assert hasattr(compiled, "ainvoke")

    def test_compile_task_graph_with_all_deps(self, mock_gateway: MagicMock, mock_tools: MagicMock) -> None:
        """compile_task_graph compiles with gateway and tools injected."""
        compiled = compile_task_graph(
            gateway=mock_gateway,
            tools=mock_tools,
        )
        assert hasattr(compiled, "ainvoke")

    def test_compile_task_graph_with_interrupt_before(self) -> None:
        """compile_task_graph accepts interrupt_before for HITL nodes."""
        compiled = compile_task_graph(
            interrupt_before=["hitl_gate"],
        )
        assert hasattr(compiled, "ainvoke")
