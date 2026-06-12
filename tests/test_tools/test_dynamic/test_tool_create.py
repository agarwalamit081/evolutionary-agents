"""Tests for the tool_create graph node."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.graph.enums import Phase
from src.graph.nodes.tool_create import tool_create_node


def _make_state(gaps: list[str] | None = None) -> dict[str, Any]:
    """Create a minimal state dict for tool_create_node."""
    from src.graph.factory import initial_state

    state = dict(initial_state("Test goal", "test-thread", 10))
    if gaps:
        state["pending_tool_gaps"] = gaps
    return state


class TestToolCreateNode:
    """Tests for tool_create_node."""

    @pytest.mark.asyncio
    async def test_no_gaps_skips_creation(self) -> None:
        result = await tool_create_node(_make_state(), gateway=MagicMock(), tools=MagicMock())
        assert result["phase"] == Phase.EXECUTE
        assert result["pending_tool_gaps"] == []

    @pytest.mark.asyncio
    async def test_no_deps_skips_creation(self) -> None:
        state = _make_state(gaps=["HTTP fetcher"])
        result = await tool_create_node(state, gateway=None, tools=None)
        assert result["phase"] == Phase.EXECUTE
        assert result["pending_tool_gaps"] == []

    @pytest.mark.asyncio
    async def test_no_gateway_skips_creation(self) -> None:
        state = _make_state(gaps=["HTTP fetcher"])
        result = await tool_create_node(state, gateway=None, tools=MagicMock())
        assert result["phase"] == Phase.EXECUTE

    @pytest.mark.asyncio
    async def test_no_tools_skips_creation(self) -> None:
        state = _make_state(gaps=["HTTP fetcher"])
        result = await tool_create_node(state, gateway=MagicMock(), tools=None)
        assert result["phase"] == Phase.EXECUTE

    @pytest.mark.asyncio
    @patch("src.graph.nodes.tool_create._create_single_tool")
    async def test_successful_creation_routes_to_plan(self, mock_create: AsyncMock) -> None:
        mock_create.return_value = {
            "success": True,
            "tool_name": "http_fetcher",
            "description": "Fetch URLs",
            "safety_passed": True,
            "sandbox_passed": True,
        }
        gateway = MagicMock()
        tools = MagicMock()
        tools.list_tools = MagicMock(return_value=[])
        state = _make_state(gaps=["HTTP fetcher"])

        result = await tool_create_node(state, gateway=gateway, tools=tools)
        assert result["phase"] == Phase.PLAN
        assert len(result["tools_created"]) == 1
        assert result["tools_created"][0]["tool_name"] == "http_fetcher"

    @pytest.mark.asyncio
    @patch("src.graph.nodes.tool_create._create_single_tool")
    async def test_failed_creation_routes_to_execute(self, mock_create: AsyncMock) -> None:
        mock_create.return_value = {
            "success": False,
            "reason": "LLM generation failed",
            "gap": "HTTP fetcher",
        }
        gateway = MagicMock()
        tools = MagicMock()
        tools.list_tools = MagicMock(return_value=[])
        state = _make_state(gaps=["HTTP fetcher"])

        result = await tool_create_node(state, gateway=gateway, tools=tools)
        assert result["phase"] == Phase.EXECUTE
        assert len(result["tools_created"]) == 0
        # Failed gaps are cleared (not kept) to prevent infinite loops
        assert result["pending_tool_gaps"] == []
        # But they are recorded as attempted
        assert result["attempted_tool_gaps"] == ["HTTP fetcher"]


class TestRouteAfterToolCreate:
    """Tests for route_after_tool_create router."""

    def test_routes_to_plan_when_tools_created(self) -> None:
        from src.graph.routers import route_after_tool_create

        state = {"tools_created": [{"tool_name": "fetcher"}]}
        assert route_after_tool_create(state) == "plan"

    def test_routes_to_execute_when_no_tools(self) -> None:
        from src.graph.routers import route_after_tool_create

        state = {"tools_created": []}
        assert route_after_tool_create(state) == "execute"

    def test_routes_to_execute_when_key_missing(self) -> None:
        from src.graph.routers import route_after_tool_create

        assert route_after_tool_create({}) == "execute"


class TestRouteAfterReflect:
    """Tests for updated route_after_reflect with tool gap detection."""

    def test_routes_to_tool_create_when_gaps(self) -> None:
        from src.graph.routers import route_after_reflect

        state = {
            "reflection": MagicMock(should_replan=False),
            "confidence": "high",
            "pending_tool_gaps": ["HTTP fetcher"],
        }
        assert route_after_reflect(state) == "tool_create"

    def test_normal_routing_when_no_gaps(self) -> None:
        from src.graph.routers import route_after_reflect
        from src.graph.enums import Confidence

        state = {
            "reflection": MagicMock(should_replan=False),
            "confidence": Confidence.HIGH,
        }
        result = route_after_reflect(state)
        assert result == "verify"

    def test_empty_gaps_does_not_route_to_tool_create(self) -> None:
        from src.graph.routers import route_after_reflect
        from src.graph.enums import Confidence

        state = {
            "reflection": MagicMock(should_replan=False),
            "confidence": Confidence.HIGH,
            "pending_tool_gaps": [],
        }
        result = route_after_reflect(state)
        assert result == "verify"
