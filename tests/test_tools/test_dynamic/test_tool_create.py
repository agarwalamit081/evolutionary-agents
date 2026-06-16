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


class TestCreateSingleToolRegeneration:
    """Regression tests for the generate→validate retry-with-feedback loop.

    Bug A (N5, battery-02): a truncated handler that fails AST validation used
    to abort tool creation after a single shot, so the tool was never
    registered and cross-run persistence+recall broke. The fix feeds the
    validation error back and regenerates, bounded by _MAX_GENERATION_ATTEMPTS.
    """

    @pytest.mark.asyncio
    async def test_truncated_handler_triggers_regeneration_and_succeeds(self) -> None:
        from src.graph.nodes.tool_create import _create_single_tool

        truncated = MagicMock(
            tool_name="dup_finder", description="d",
            input_schema={}, handler_code="async def", test_code="",
        )
        valid = MagicMock(
            tool_name="dup_finder", description="d",
            input_schema={}, handler_code="async def dup(): ...", test_code="",
        )
        gen_instance = MagicMock()
        gen_instance.generate = AsyncMock(side_effect=[truncated, valid])
        gen_instance.validate_and_register = AsyncMock(
            side_effect=[
                {"success": False, "reason": "Syntax error at line 6: '(' was never closed"},
                {"success": True, "sandbox_result": {"passed": True}},
            ]
        )

        registry = MagicMock()
        registry.list_names = MagicMock(return_value=[])

        with patch(
            "src.tools.dynamic.generator.ToolGenerator", return_value=gen_instance
        ), patch("src.graph.nodes.tool_create._persist_tool", new=AsyncMock()):
            result = await _create_single_tool(
                gateway=MagicMock(),
                registry=registry,
                gap_description="duplicate finder",
                goal_text="g",
                tool_results=[],
            )

        # Regeneration succeeded on attempt 2
        assert result["success"] is True
        assert result["tool_name"] == "dup_finder"
        assert gen_instance.generate.await_count == 2
        # The first attempt's validation error was fed back into attempt 2
        second_context = gen_instance.generate.await_args_list[1].args[1]
        assert "Previous generation attempt failed" in second_context["error_details"]

    @pytest.mark.asyncio
    async def test_persistent_failure_exhausts_attempts_and_reports(self) -> None:
        from src.graph.nodes.tool_create import _MAX_GENERATION_ATTEMPTS
        from src.graph.nodes.tool_create import _create_single_tool

        gen_instance = MagicMock()
        gen_instance.generate = AsyncMock(
            return_value=MagicMock(
                tool_name="t", description="d", input_schema={},
                handler_code="x", test_code="",
            )
        )
        # Every validation fails — must exhaust, not loop forever
        gen_instance.validate_and_register = AsyncMock(
            return_value={"success": False, "reason": "Syntax error"}
        )

        registry = MagicMock()
        registry.list_names = MagicMock(return_value=[])

        persist_mock = AsyncMock()
        with patch(
            "src.tools.dynamic.generator.ToolGenerator", return_value=gen_instance
        ), patch("src.graph.nodes.tool_create._persist_tool", new=persist_mock):
            result = await _create_single_tool(
                gateway=MagicMock(),
                registry=registry,
                gap_description="bogus",
                goal_text="g",
                tool_results=[],
            )

        assert result["success"] is False
        assert gen_instance.generate.await_count == _MAX_GENERATION_ATTEMPTS
        assert "All" in result["reason"]
        # Persistence must never run when every attempt failed
        assert persist_mock.await_count == 0


class TestCreateNodeSkipsAttemptedGaps:
    """Bug C backstop (battery-02 N6): tool_create_node must never re-attempt a
    gap already in ``attempted_tool_gaps``. agent_spawn's failed-spawn →
    tool-gap conversion re-seeds ``pending_tool_gaps`` directly (bypassing
    reflect's dedup), so without this consumer-side guard the bounded
    3-attempt regeneration loop fired once per re-seed (N6: 19 node entries,
    ~56 generations, 764s). ``attempted_tool_gaps`` is an operator.add list, so
    a gap recorded once stays recorded for the whole run.
    """

    @pytest.mark.asyncio
    async def test_attempted_gap_is_skipped_not_generated(self) -> None:
        state = _make_state(gaps=["duplicate_finder"])
        state["attempted_tool_gaps"] = ["duplicate_finder"]

        with patch(
            "src.graph.nodes.tool_create._create_single_tool", new=AsyncMock()
        ) as mock_create:
            result = await tool_create_node(
                state, gateway=MagicMock(), tools=MagicMock()
            )

        mock_create.assert_not_awaited()
        assert result["phase"] == Phase.EXECUTE
        assert result["pending_tool_gaps"] == []

    @pytest.mark.asyncio
    async def test_only_fresh_gaps_are_generated(self) -> None:
        state = _make_state(gaps=["duplicate_finder", "csv_parser"])
        state["attempted_tool_gaps"] = ["duplicate_finder"]  # already failed

        mock_create = AsyncMock(
            return_value={"success": True, "tool_name": "csv_parser"}
        )
        with patch(
            "src.graph.nodes.tool_create._create_single_tool", new=mock_create
        ):
            result = await tool_create_node(
                state, gateway=MagicMock(), tools=MagicMock()
            )

        # Only the fresh gap reached generation; the attempted one was skipped.
        mock_create.assert_awaited_once()
        assert mock_create.await_args is not None
        assert mock_create.await_args.kwargs["gap_description"] == "csv_parser"
        assert result["phase"] == Phase.PLAN
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
