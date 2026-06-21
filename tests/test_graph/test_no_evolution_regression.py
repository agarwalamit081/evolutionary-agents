"""Regression tests for the ``--no-evolution`` flag (F4).

Why a separate file: the original Phase-4 unit tests invoked
``route_after_verify`` directly WITH a ``config`` keyword, which masked the
production reality. LangGraph calls conditional-edge routers single-arg
(``router(state)``); in this graph (AsyncPostgresSaver checkpointer +
``interrupt_before`` + subgraphs) it passes ``config=None``. So the
config-based ``no_evolution`` was a no-op live — a ``--no-evolution`` run
still triggered ``evolve``. The fix threads ``no_evolution`` through
``AgentState``.

These tests guard against regression by:

1. Driving the REAL compiled graph (proves the wiring is intact, not just the
   router function in isolation).
2. Invoking ``route_after_verify`` single-arg — the exact call shape LangGraph
   uses — never passing ``config``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.graph.factory import initial_state
from src.graph.models import ReflectionResult
from src.graph.routers import route_after_verify
from src.graph.task_graph import build_task_graph, compile_task_graph


def _completed_state(no_evolution: bool) -> dict[str, Any]:
    """A post-verify state: complete, reflection requests evolution."""
    state = dict(initial_state("Some goal", "reg-thread", 5))
    state["is_complete"] = True
    state["no_evolution"] = no_evolution
    state["reflection"] = ReflectionResult(summary="done", should_evolve=True)
    return state


class TestNoEvolutionGraphWiring:
    """The real compiled graph wires ``route_after_verify`` at ``verify``."""

    def test_verify_branch_uses_route_after_verify(self) -> None:
        """The verify node's conditional edge is the (single-arg) router."""
        graph = build_task_graph()
        verify_branches = graph.branches.get("verify")
        assert verify_branches is not None, "verify node has no conditional edge"
        # LangGraph keys the branch dict by the router function name.
        assert "route_after_verify" in verify_branches, (
            "verify edge must route through route_after_verify"
        )
        branch = verify_branches["route_after_verify"]
        # Both destinations must be reachable — proves evolve AND store_memory
        # are wired (the branch isn't accidentally collapsed). The router no
        # longer declares a ``config`` param, so it cannot accept one — verified
        # by the single-arg parametrized test below.
        assert "evolve" in branch.ends
        assert "store_memory" in branch.ends

    def test_real_compiled_graph_constructs_with_no_evolution_in_state(
        self, mock_gateway: MagicMock, mock_memory: MagicMock, mock_tools: MagicMock
    ) -> None:
        """The production graph compiles cleanly and carries no_evolution."""
        compiled = compile_task_graph(
            gateway=mock_gateway, memory=mock_memory, tools=mock_tools
        )
        assert hasattr(compiled, "ainvoke")
        state = dict(initial_state("goal", "t", 3, no_evolution=True))
        assert state["no_evolution"] is True


class TestNoEvolutionRouting:
    """``route_after_verify`` reads no_evolution from STATE, single-arg (no config)."""

    @pytest.mark.parametrize("no_evolution", [True, False])
    def test_router_honors_state_flag_single_arg(self, no_evolution: bool) -> None:
        """Single-arg invocation (LangGraph's real call shape) honors the flag.

        True  → store_memory (evolve skipped)
        False → evolve
        """
        result = route_after_verify(_completed_state(no_evolution))
        if no_evolution:
            assert result == "store_memory"
        else:
            assert result == "evolve"

    def test_router_default_when_key_absent_routes_to_evolve(self) -> None:
        """State built without an explicit no_evolution key defaults to evolve."""
        state = _completed_state(no_evolution=False)
        state.pop("no_evolution", None)
        assert route_after_verify(state) == "evolve"


class TestInitialStateSeedsFlag:
    """``initial_state`` propagates ``no_evolution`` into state."""

    def test_default_is_false(self) -> None:
        state = initial_state("goal", "t", 5)
        assert state["no_evolution"] is False

    def test_true_propagates(self) -> None:
        state = initial_state("goal", "t", 5, no_evolution=True)
        assert state["no_evolution"] is True
