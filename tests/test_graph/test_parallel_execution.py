"""Tests for parallel tool execution and sub-agent delegation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.graph.enums import GoalStatus, Phase
from src.graph.models import Goal
from src.graph.nodes.execute import (
    _execute_tool_call,
    _execute_tool_calls_parallel,
)
from src.tools.registry import ToolRegistry


def _mock_tool_call(name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create a mock tool call dict."""
    import json

    return {
        "id": f"call_{name}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args or {}),
        },
    }


def _mock_registry_with_handlers(
    handlers: dict[str, Callable[..., Coroutine[Any, Any, str]]],
) -> ToolRegistry:
    """Create a ToolRegistry with mocked async handlers."""
    registry = ToolRegistry()
    for name, handler in handlers.items():
        registry.register(
            name=name,
            handler=handler,
            description=f"Mock {name} tool",
        )
    return registry


class TestExecuteToolCall:
    """Tests for single tool call execution."""

    @pytest.mark.asyncio
    async def test_successful_tool_call(self) -> None:
        """Should return success ToolResult for valid tool."""
        handler = AsyncMock(return_value="result data")
        tools = _mock_registry_with_handlers({"my_tool": handler})  # type: ignore[arg-type]
        tc = _mock_tool_call("my_tool", {"key": "value"})

        result = await _execute_tool_call(tc, tools)

        assert result.success is True
        assert result.tool_name == "my_tool"
        assert result.output == "result data"
        handler.assert_called_once_with(key="value")

    @pytest.mark.asyncio
    async def test_unknown_tool(self) -> None:
        """Should return failure for unknown tool."""
        tools = ToolRegistry()
        tc = _mock_tool_call("nonexistent")

        result = await _execute_tool_call(tc, tools)

        assert result.success is False
        assert result.error is not None
        assert "Unknown tool" in result.error

    @pytest.mark.asyncio
    async def test_tool_exception(self) -> None:
        """Should return failure when tool handler raises."""
        handler = AsyncMock(side_effect=ValueError("bad input"))
        tools = _mock_registry_with_handlers({"failing_tool": handler})  # type: ignore[arg-type]
        tc = _mock_tool_call("failing_tool", {"x": 1})

        result = await _execute_tool_call(tc, tools)

        assert result.success is False
        assert result.error is not None
        assert "bad input" in result.error


class TestParallelToolExecution:
    """Tests for parallel execution of multiple tool calls."""

    @pytest.mark.asyncio
    async def test_multiple_tools_execute(self) -> None:
        """All tool calls should execute and return results."""
        call_order: list[str] = []

        async def slow_handler(**_kwargs: Any) -> str:
            await asyncio.sleep(0.01)
            call_order.append("slow")
            return "slow_done"

        async def fast_handler(**_kwargs: Any) -> str:
            call_order.append("fast")
            return "fast_done"

        tools = _mock_registry_with_handlers({
            "slow": slow_handler,
            "fast": fast_handler,
        })

        calls = [_mock_tool_call("slow"), _mock_tool_call("fast")]
        results = await _execute_tool_calls_parallel(calls, tools)

        assert len(results) == 2
        assert results[0].tool_name == "slow"
        assert results[1].tool_name == "fast"
        assert results[0].success is True
        assert results[1].success is True

    @pytest.mark.asyncio
    async def test_single_call_skips_gather(self) -> None:
        """Single tool call should execute without gather overhead."""
        handler = AsyncMock(return_value="solo_result")
        tools = _mock_registry_with_handlers({"solo": handler})  # type: ignore[arg-type]
        calls = [_mock_tool_call("solo")]

        results = await _execute_tool_calls_parallel(calls, tools)

        assert len(results) == 1
        assert results[0].output == "solo_result"

    @pytest.mark.asyncio
    async def test_empty_calls_returns_empty(self) -> None:
        """Empty tool calls list should return empty results."""
        tools = ToolRegistry()
        results = await _execute_tool_calls_parallel([], tools)
        assert results == []

    @pytest.mark.asyncio
    async def test_error_isolation(self) -> None:
        """One tool failure should not prevent others from executing."""
        good_handler = AsyncMock(return_value="good_result")
        bad_handler = AsyncMock(side_effect=RuntimeError("exploded"))

        tools = _mock_registry_with_handlers({
            "good_tool": good_handler,
            "bad_tool": bad_handler,
        })  # type: ignore[arg-type]

        calls = [_mock_tool_call("bad_tool"), _mock_tool_call("good_tool")]
        results = await _execute_tool_calls_parallel(calls, tools)

        assert len(results) == 2
        assert results[0].success is False
        assert results[0].error is not None
        assert "exploded" in results[0].error
        assert results[1].success is True
        assert results[1].output == "good_result"

    @pytest.mark.asyncio
    async def test_parallel_is_faster_than_sequential(self) -> None:
        """Parallel execution should be faster than sequential for I/O-bound tools."""
        async def delayed_handler(**_kwargs: Any) -> str:
            await asyncio.sleep(0.1)
            return "done"

        tools = _mock_registry_with_handlers({"delayed": delayed_handler})
        calls = [_mock_tool_call("delayed") for _ in range(3)]

        import time

        start = time.monotonic()
        results = await _execute_tool_calls_parallel(calls, tools)
        elapsed = time.monotonic() - start

        assert len(results) == 3
        assert all(r.success for r in results)
        # Parallel: ~0.1s. Sequential would be ~0.3s.
        assert elapsed < 0.25, f"Parallel took {elapsed:.2f}s — expected < 0.25s"


class TestParallelSubAgentDelegation:
    """Tests for parallel sub-agent delegation in delegate node."""

    @pytest.mark.asyncio
    async def test_delegate_uses_run_parallel(self) -> None:
        """Delegate node should call run_parallel for multiple agents."""
        from src.graph.nodes.delegate import delegate_node

        # Mock specs
        spec1 = MagicMock()
        spec1.name = "agent_1"
        spec1.is_active = True
        spec1.description = "First agent"
        spec1.id = "id-1"

        spec2 = MagicMock()
        spec2.name = "agent_2"
        spec2.is_active = True
        spec2.description = "Second agent"
        spec2.id = "id-2"

        # Mock registry
        mock_registry = MagicMock()
        mock_registry.get = lambda name: {"agent_1": spec1, "agent_2": spec2}.get(name)
        mock_runner_1 = MagicMock()
        mock_runner_1.definition = spec1
        mock_runner_2 = MagicMock()
        mock_runner_2.definition = spec2
        mock_registry.spawn = MagicMock(side_effect=[mock_runner_1, mock_runner_2])
        mock_registry.check_deprecation = MagicMock()

        # Mock gateway
        mock_gateway = MagicMock()

        # Mock tools
        mock_tools = MagicMock()

        state = {
            "sub_agents_spawned": [
                {"name": "agent_1", "description": "First"},
                {"name": "agent_2", "description": "Second"},
            ],
            "thread_id": "test-thread",
            "current_goal": Goal(text="test goal", status=GoalStatus.ACTIVE),
        }

        with patch("src.agents.runner.run_parallel") as mock_run_parallel:
            mock_run_parallel.return_value = [
                {
                    "success": True,
                    "result": "agent 1 done",
                    "tokens_used": 100,
                    "latency_ms": 50,
                    "errors": [],
                },
                {
                    "success": True,
                    "result": "agent 2 done",
                    "tokens_used": 200,
                    "latency_ms": 75,
                    "errors": [],
                },
            ]

            result = await delegate_node(
                state,
                gateway=mock_gateway,
                tools=mock_tools,
                sub_agent_registry=mock_registry,
                memory=None,
            )

        # Verify run_parallel was called
        mock_run_parallel.assert_called_once()
        call_args = mock_run_parallel.call_args[0][0]
        assert len(call_args) == 2

        # Verify results collected
        assert result["phase"] == Phase.VERIFY
        assert len(result["delegation_results"]) == 2
        assert len(result["tool_results"]) == 2

    @pytest.mark.asyncio
    async def test_delegate_handles_inactive_agent(self) -> None:
        """Inactive agent should be recorded as failure without blocking others."""
        from src.graph.nodes.delegate import delegate_node

        spec_active = MagicMock()
        spec_active.name = "active_agent"
        spec_active.is_active = True
        spec_active.description = "Active"
        spec_active.id = "id-active"

        spec_inactive = MagicMock()
        spec_inactive.name = "inactive_agent"
        spec_inactive.is_active = False
        spec_inactive.description = "Inactive"

        mock_registry = MagicMock()
        _specs = {"active_agent": spec_active, "inactive_agent": spec_inactive}
        mock_registry.get = lambda name: _specs.get(name)
        mock_runner = MagicMock()
        mock_runner.definition = spec_active
        mock_runner.run = AsyncMock(return_value={
            "success": True,
            "result": "done",
            "tokens_used": 50,
            "latency_ms": 30,
            "errors": [],
        })
        mock_registry.spawn = MagicMock(return_value=mock_runner)
        mock_registry.check_deprecation = MagicMock()

        state = {
            "sub_agents_spawned": [
                {"name": "inactive_agent"},
                {"name": "active_agent"},
            ],
            "thread_id": "test-thread",
            "current_goal": Goal(text="test", status=GoalStatus.ACTIVE),
        }

        result = await delegate_node(
            state,
            gateway=MagicMock(),
            tools=MagicMock(),
            sub_agent_registry=mock_registry,
            memory=None,
        )

        # Should have 2 delegation results: 1 failure (inactive) + 1 success (active)
        assert len(result["delegation_results"]) == 2
        # Phase is EXECUTE because one agent failed (all_success = False)
        assert result["phase"] == Phase.EXECUTE
