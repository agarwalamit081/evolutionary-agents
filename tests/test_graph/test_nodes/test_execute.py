"""Tests for src.graph.nodes.execute — execute node function."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.graph.enums import GoalStatus, Phase
from src.graph.models import PlanStep, ToolResult
from src.graph.nodes.execute import execute_node
from src.llm.models import ToolCallResponse


class TestExecuteNode:
    """Tests for the execute_node async function."""

    @pytest.mark.asyncio
    async def test_execute_marks_step_complete(self, state_with_plan: dict[str, Any]) -> None:
        """Execute node marks the current step as COMPLETED."""
        result = await execute_node(state_with_plan)

        # The completed_steps list should contain the finished step
        completed = result["completed_steps"]
        assert len(completed) == 1
        assert completed[0].status == GoalStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_advances_step_index(self, state_with_plan: dict[str, Any]) -> None:
        """Execute node increments current_step_index by 1."""
        initial_index = state_with_plan["current_step_index"]
        assert initial_index == 0

        result = await execute_node(state_with_plan)
        assert result["current_step_index"] == initial_index + 1

    @pytest.mark.asyncio
    async def test_execute_no_plan_returns_reflect(self, sample_state: dict[str, Any]) -> None:
        """When plan_steps is empty, execute routes to REFLECT phase."""
        sample_state["plan_steps"] = []
        sample_state["current_step_index"] = 0

        result = await execute_node(sample_state)
        assert result["phase"] == Phase.REFLECT

    @pytest.mark.asyncio
    async def test_execute_index_out_of_range_returns_reflect(self, sample_state: dict[str, Any]) -> None:
        """When step_index >= len(plan_steps), route to REFLECT."""
        sample_state["plan_steps"] = [
            PlanStep(id="s1", description="Step 1"),
        ]
        sample_state["current_step_index"] = 5  # out of range

        result = await execute_node(sample_state)
        assert result["phase"] == Phase.REFLECT

    @pytest.mark.asyncio
    async def test_execute_increments_iteration_count(self, state_with_plan: dict[str, Any]) -> None:
        """Execute node increments iteration_count."""
        initial_count = state_with_plan["iteration_count"]
        result = await execute_node(state_with_plan)
        assert result["iteration_count"] == initial_count + 1

    @pytest.mark.asyncio
    async def test_execute_adds_message(self, state_with_plan: dict[str, Any]) -> None:
        """Execute node adds a user message describing the execution."""
        result = await execute_node(state_with_plan)
        messages = result.get("messages", [])
        assert len(messages) >= 1
        # The message should reference the step description
        step_desc = state_with_plan["plan_steps"][0].description
        assert any(step_desc in str(m) for m in messages)

    @pytest.mark.asyncio
    async def test_execute_sequential_steps(self, state_with_plan: dict[str, Any]) -> None:
        """Execute processes steps sequentially, advancing index each call."""
        assert state_with_plan["current_step_index"] == 0

        result1 = await execute_node(state_with_plan)
        assert result1["current_step_index"] == 1

        # Update state for next step
        state_with_plan["current_step_index"] = result1["current_step_index"]
        result2 = await execute_node(state_with_plan)
        assert result2["current_step_index"] == 2

    @pytest.mark.asyncio
    async def test_execute_step_has_result(self, state_with_plan: dict[str, Any]) -> None:
        """After execution, the completed step has a result string."""
        result = await execute_node(state_with_plan)
        completed = result["completed_steps"][0]
        assert completed.result is not None
        assert len(completed.result) > 0


class TestExecuteNodeLLM:
    """Tests for the execute_node LLM tool-calling path via closure injection."""

    @pytest.mark.asyncio
    async def test_llm_execute_with_tool_calls(self, state_with_plan: dict[str, Any]) -> None:
        """LLM returns tool calls — handler invoked, ToolResult objects created."""
        gateway = MagicMock()
        gateway.acompletion_with_tools = AsyncMock(return_value=ToolCallResponse(
            content="Used code_executor",
            tool_calls=[{
                "id": "tc1",
                "type": "function",
                "function": {
                    "name": "code_executor",
                    "arguments": '{"code": "print(42)"}',
                },
            }],
            model="gpt-4o-mini-2024-07-18",
            provider="openai",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            cost_usd=0.0001,
        ))

        async_handler = AsyncMock(return_value="42")
        tools = MagicMock()
        tools.list_tools = MagicMock(return_value=[{
            "type": "function",
            "function": {
                "name": "code_executor",
                "description": "Execute code",
                "parameters": {"type": "object", "properties": {"code": {"type": "string"}}},
            },
        }])
        tools.get_handler = MagicMock(return_value=async_handler)

        result = await execute_node(state_with_plan, gateway=gateway, tools=tools)

        # Handler was called with parsed args
        async_handler.assert_awaited_once_with(code="print(42)")

        # ToolResult created for the successful call
        tool_results: list[ToolResult] = result["tool_results"]
        assert len(tool_results) == 1
        assert tool_results[0].tool_name == "code_executor"
        assert tool_results[0].success is True
        assert tool_results[0].output == "42"

        # Step completed and index advanced
        assert result["phase"] == Phase.REFLECT
        assert result["current_step_index"] == 1

    @pytest.mark.asyncio
    async def test_llm_execute_with_unknown_tool(self, state_with_plan: dict[str, Any]) -> None:
        """LLM returns tool call for unknown tool — error ToolResult created."""
        gateway = MagicMock()
        gateway.acompletion_with_tools = AsyncMock(return_value=ToolCallResponse(
            content="Tried unknown tool",
            tool_calls=[{
                "id": "tc2",
                "type": "function",
                "function": {
                    "name": "nonexistent_tool",
                    "arguments": "{}",
                },
            }],
            model="gpt-4o-mini-2024-07-18",
            provider="openai",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            cost_usd=0.0001,
        ))

        tools = MagicMock()
        tools.list_tools = MagicMock(return_value=[])
        # get_handler returns None → unknown tool
        tools.get_handler = MagicMock(return_value=None)

        result = await execute_node(state_with_plan, gateway=gateway, tools=tools)

        tool_results: list[ToolResult] = result["tool_results"]
        assert len(tool_results) == 1
        assert tool_results[0].tool_name == "nonexistent_tool"
        assert tool_results[0].success is False
        assert "Unknown tool" in tool_results[0].error

    @pytest.mark.asyncio
    async def test_llm_execute_with_handler_exception(self, state_with_plan: dict[str, Any]) -> None:
        """Tool handler raises exception — error captured in ToolResult."""
        gateway = MagicMock()
        gateway.acompletion_with_tools = AsyncMock(return_value=ToolCallResponse(
            content="Tool call failed",
            tool_calls=[{
                "id": "tc3",
                "type": "function",
                "function": {
                    "name": "code_executor",
                    "arguments": '{"code": "raise ValueError()"}',
                },
            }],
            model="gpt-4o-mini-2024-07-18",
            provider="openai",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            cost_usd=0.0001,
        ))

        failing_handler = AsyncMock(side_effect=RuntimeError("sandbox timeout"))
        tools = MagicMock()
        tools.list_tools = MagicMock(return_value=[])
        tools.get_handler = MagicMock(return_value=failing_handler)

        result = await execute_node(state_with_plan, gateway=gateway, tools=tools)

        tool_results: list[ToolResult] = result["tool_results"]
        assert len(tool_results) == 1
        assert tool_results[0].success is False
        assert "sandbox timeout" in tool_results[0].error

    @pytest.mark.asyncio
    async def test_llm_execute_no_tool_calls(self, state_with_plan: dict[str, Any]) -> None:
        """LLM returns no tool calls — step still completed with text content."""
        gateway = MagicMock()
        gateway.acompletion_with_tools = AsyncMock(return_value=ToolCallResponse(
            content="I analyzed the requirements. No tool needed.",
            tool_calls=[],
            model="gpt-4o-mini-2024-07-18",
            provider="openai",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            cost_usd=0.0001,
        ))

        tools = MagicMock()
        tools.list_tools = MagicMock(return_value=[])

        result = await execute_node(state_with_plan, gateway=gateway, tools=tools)

        # No tool results but step still completed
        assert result["tool_results"] == []
        assert result["current_step_index"] == 1
        assert result["phase"] == Phase.REFLECT
        # The completed step should have the LLM content as result
        completed_step = result["completed_steps"][0]
        assert completed_step.status == GoalStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_llm_execute_returns_none_falls_back(self, state_with_plan: dict[str, Any]) -> None:
        """gateway.acompletion_with_tools raises — falls back to simulated execution."""
        gateway = MagicMock()
        gateway.acompletion_with_tools = AsyncMock(side_effect=RuntimeError("API unreachable"))

        tools = MagicMock()
        tools.list_tools = MagicMock(return_value=[])

        result = await execute_node(state_with_plan, gateway=gateway, tools=tools)

        # Falls back to simulated execution
        assert result["phase"] == Phase.REFLECT
        assert result["current_step_index"] == 1
        completed_step = result["completed_steps"][0]
        assert completed_step.status == GoalStatus.COMPLETED
        assert "Executed:" in completed_step.result

    @pytest.mark.asyncio
    async def test_llm_execute_step_status_transitions(self, state_with_plan: dict[str, Any]) -> None:
        """Step transitions from PENDING → ACTIVE → COMPLETED during LLM execution."""
        gateway = MagicMock()
        gateway.acompletion_with_tools = AsyncMock(return_value=ToolCallResponse(
            content="Done",
            tool_calls=[],
            model="gpt-4o-mini-2024-07-18",
            provider="openai",
            input_tokens=5,
            output_tokens=5,
            total_tokens=10,
            cost_usd=0.0,
        ))

        tools = MagicMock()
        tools.list_tools = MagicMock(return_value=[])

        # Verify initial state
        step = state_with_plan["plan_steps"][0]
        assert step.status == GoalStatus.PENDING

        result = await execute_node(state_with_plan, gateway=gateway, tools=tools)

        # After execution, the step in completed_steps should be COMPLETED
        completed_step = result["completed_steps"][0]
        assert completed_step.status == GoalStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_llm_execute_with_multiple_tool_calls(self, state_with_plan: dict[str, Any]) -> None:
        """LLM returns multiple tool calls — all handlers invoked, all ToolResults created."""
        gateway = MagicMock()
        gateway.acompletion_with_tools = AsyncMock(return_value=ToolCallResponse(
            content="Used multiple tools",
            tool_calls=[
                {
                    "id": "tc_a",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"query": "python async"}'},
                },
                {
                    "id": "tc_b",
                    "type": "function",
                    "function": {"name": "code_executor", "arguments": '{"code": "1+1"}'},
                },
            ],
            model="gpt-4o-mini-2024-07-18",
            provider="openai",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            cost_usd=0.0001,
        ))

        handler_search = AsyncMock(return_value="search results here")
        handler_code = AsyncMock(return_value="2")

        tools = MagicMock()
        tools.list_tools = MagicMock(return_value=[])
        tools.get_handler = MagicMock(side_effect=[handler_search, handler_code])

        result = await execute_node(state_with_plan, gateway=gateway, tools=tools)

        tool_results: list[ToolResult] = result["tool_results"]
        assert len(tool_results) == 2
        assert tool_results[0].tool_name == "web_search"
        assert tool_results[0].success is True
        assert tool_results[1].tool_name == "code_executor"
        assert tool_results[1].success is True
