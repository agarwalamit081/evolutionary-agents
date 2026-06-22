"""Tests for src.graph.nodes.execute — execute node function."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config.settings import get_settings
from src.graph.enums import GoalStatus, Phase
from src.graph.factory import initial_state
from src.graph.models import PlanStep, ToolResult
from src.graph.nodes.execute import (
    _called_file_output_tool,
    _deliverable_on_disk,
    _extract_expected_file_path,
    _extract_goal_deliverable,
    _first_file_output_call,
    _is_compute_deliverable,
    _is_producing_step,
    _tool_call_args,
    _write_nudge,
    execute_node,
)
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
    async def test_write_step_forces_file_writer_tool_choice_on_nudge(
        self, state_with_plan: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A write step that the model narrates (instead of calling file_writer)
        is re-prompted with a forced named tool_choice so narration-prone models
        cannot reply with prose on the nudge turn. Turn 1 stays free; the nudge
        turn pins file_writer. Regression for the q3/q4 loop where the cheap
        model narrated through every nudge and the deliverable was never written.
        """
        # Deterministic nudge budget independent of ambient .env MAX_WRITE_NUDGES
        monkeypatch.setattr(get_settings().agent, "max_write_nudges", 2)

        # Step 0 is a write-step → expected_path derived from the description
        state_with_plan["plan_steps"] = [
            PlanStep(
                id="s1",
                description="Write the report to results/q/test_report.md",
                status="pending",
            ),
        ]

        narration = ToolCallResponse(
            content="I will write the report now.",
            tool_calls=[],
            model="gpt-4o-mini-2024-07-18",
            provider="openai",
            input_tokens=10,
            output_tokens=10,
            total_tokens=20,
            cost_usd=0.0001,
        )
        write_call = ToolCallResponse(
            content="",
            tool_calls=[{
                "id": "tc_fw",
                "type": "function",
                "function": {
                    "name": "file_writer",
                    "arguments": '{"file_path": "results/q/test_report.md", "content": "# Report"}',
                },
            }],
            model="gpt-4o-mini-2024-07-18",
            provider="openai",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            cost_usd=0.0002,
        )
        gateway = MagicMock()
        gateway.acompletion_with_tools = AsyncMock(side_effect=[narration, write_call])

        async_handler = AsyncMock(return_value="wrote results/q/test_report.md")
        tools = MagicMock()
        tools.list_tools = MagicMock(return_value=[{
            "type": "function",
            "function": {
                "name": "file_writer",
                "description": "Write a file to disk",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                },
            },
        }])
        tools.get_handler = MagicMock(return_value=async_handler)

        result = await execute_node(state_with_plan, gateway=gateway, tools=tools)

        calls = gateway.acompletion_with_tools.call_args_list
        assert len(calls) == 2
        # Turn 1: no nudge yet → free choice
        assert calls[0].kwargs.get("tool_choice") is None
        # Turn 1's front-loaded step label directs create_dirs + file_writer
        turn1_messages = calls[0].kwargs.get("messages", [])
        step_labels = " ".join(
            str(m.get("content", "")) for m in turn1_messages if m.get("role") == "user"
        ).lower()
        assert "create_dirs" in step_labels, step_labels
        assert "file_writer" in step_labels, step_labels
        # Turn 2 (nudge): forced file_writer so prose cannot win
        assert calls[1].kwargs.get("tool_choice") == {
            "type": "function",
            "function": {"name": "file_writer"},
        }
        # Step completed and advanced after the write
        assert result["phase"] == Phase.REFLECT
        assert result["current_step_index"] == 1

    def test_write_nudge_directs_create_dirs_and_file_writer(self) -> None:
        """For a TEXT deliverable the write nudge must tell the model to set
        create_dirs=true and use file_writer (not code_executor).

        Regression for battery-04 q3: even when file_writer was forced via
        tool_choice, some models omitted create_dirs (defaulting then to False)
        so nested writes silently failed on a missing parent. The nudge now
        spells out create_dirs=true and names file_writer over code_executor.
        (Compute/data deliverables like .csv have their own branch — see
        TestComputeDeliverableSteering.)
        """
        text = _write_nudge("results/q03/overview.md", compute=False)
        low = text.lower()
        assert "create_dirs" in low, text
        assert "file_writer" in low, text
        assert "code_executor" not in low, text
        assert "results/q03/overview.md" in text

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

        # F14: a failed tool call must NOT read as progress. The step stays
        # ACTIVE, the index does not advance, and the run loops back to execute
        # (so the LLM sees the failure in the thread/tool_results) instead of
        # aborting with a false is_complete=True and no deliverable.
        assert result["phase"] == Phase.EXECUTE
        assert result["current_step_index"] == 0
        assert "completed_steps" not in result

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

        # F14: a handler exception is a recoverable tool failure, not success —
        # retry execute without advancing the step, never false-complete.
        assert result["phase"] == Phase.EXECUTE
        assert result["current_step_index"] == 0
        assert "completed_steps" not in result

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


class TestExecuteReActThread:
    """WS1: execute maintains a real conversation thread (ReAct), not a stub.

    Guards the regression where execute appended a single throwaway user line
    per step — starving memory folding of real context to compress. The thread
    now carries the Human turn, the AI turn (with tool calls), and each Tool
    result, correlated by ``tool_call_id``. ``result_cache`` defaults to None
    here, so the cacheable path is not exercised (covered in test_result_cache).
    """

    @pytest.mark.asyncio
    async def test_appends_human_ai_tool_messages_in_order(
        self, state_with_plan: dict[str, Any]
    ) -> None:
        """A tool-calling step appends Human → AI(tool_calls) → Tool, in order."""
        gateway = MagicMock()
        gateway.acompletion_with_tools = AsyncMock(return_value=ToolCallResponse(
            content="Running code",
            tool_calls=[{
                "id": "tc_thread_1",
                "type": "function",
                "function": {"name": "code_executor", "arguments": '{"code": "print(1)"}'},
            }],
            model="gpt-4o-mini-2024-07-18", provider="openai",
            input_tokens=10, output_tokens=20, total_tokens=30, cost_usd=0.0001,
        ))
        handler = AsyncMock(return_value="1")
        tools = MagicMock()
        tools.list_tools = MagicMock(return_value=[])
        tools.get_handler = MagicMock(return_value=handler)

        result = await execute_node(state_with_plan, gateway=gateway, tools=tools)

        msgs = result["messages"]
        types = [getattr(m, "type", "") for m in msgs]
        # Human turn → AI turn → exactly one ToolMessage per tool call.
        assert types[0] == "human"
        assert types[1] == "ai"
        assert types[2:] == ["tool"]

        # The AI turn carries the validated tool call.
        assert msgs[1].tool_calls, "AIMessage must carry the tool call"
        # The ToolMessage is correlated to the call by id and named for the tool.
        assert msgs[2].tool_call_id == "tc_thread_1"
        assert msgs[2].name == "code_executor"
        # And the ToolResult carries the same id for folding/verification.
        assert result["tool_results"][0].metadata["tool_call_id"] == "tc_thread_1"

    @pytest.mark.asyncio
    async def test_gateway_receives_accumulated_history(
        self, state_with_plan: dict[str, Any]
    ) -> None:
        """Prior turns in state['messages'] are fed to the LLM (intra-run memory)."""
        goal_msg = state_with_plan["messages"][0]  # seeded HumanMessage(goal_text)
        gateway = MagicMock()
        gateway.acompletion_with_tools = AsyncMock(return_value=ToolCallResponse(
            content="ok", tool_calls=[],
            model="gpt-4o-mini-2024-07-18", provider="openai",
            input_tokens=5, output_tokens=5, total_tokens=10, cost_usd=0.0,
        ))
        tools = MagicMock()
        tools.list_tools = MagicMock(return_value=[])

        await execute_node(state_with_plan, gateway=gateway, tools=tools)

        sent = gateway.acompletion_with_tools.call_args.kwargs["messages"]
        # The seeded goal (a prior HumanMessage) must appear in the history sent.
        assert any(getattr(goal_msg, "content", "") in str(m) for m in sent)

    @pytest.mark.asyncio
    async def test_write_step_front_loads_file_writer_hint(self) -> None:
        """A step declaring a deliverable tells the LLM to call file_writer on
        turn 1 — front-loaded in the step label, not deferred to the post-hoc
        nudge. Without this, cheaper models narrate the deliverable as prose on
        the first attempt and burn 1-2 extra turns before writing the file."""
        from src.graph.enums import Strategy

        state = dict(initial_state("Write a report", "th-write", 10))
        state["plan_steps"] = [
            PlanStep(
                id="s1",
                description="Save the analysis to results/report.md",
                status="pending",
            ),
        ]
        state["current_step_index"] = 0
        state["strategy"] = Strategy.REACT

        gateway = MagicMock()
        gateway.acompletion_with_tools = AsyncMock(return_value=ToolCallResponse(
            content="Here is the report", tool_calls=[],
            model="gpt-4o-mini-2024-07-18", provider="openai",
            input_tokens=5, output_tokens=5, total_tokens=10, cost_usd=0.0,
        ))
        tools = MagicMock()
        tools.list_tools = MagicMock(return_value=[])

        await execute_node(state, gateway=gateway, tools=tools)

        # The FIRST LLM call must already carry the file_writer instruction
        # (front-loaded), before any nudge turn fires.
        first_msgs = gateway.acompletion_with_tools.call_args_list[0].kwargs["messages"]
        blob = " ".join(str(getattr(m, "content", m)) for m in first_msgs)
        assert "file_writer" in blob
        assert "results/report.md" in blob


class TestWriteStepHelpers:
    """Pure helpers that detect write-step intent and file-output tool calls.

    These underpin the execute-node nudge: a step declaring a file deliverable
    that the LLM narrates as text (instead of calling file_writer) gets a
    bounded re-prompt so the deliverable is actually produced. The path regex
    mirrors verify._SAVE_TO_RE so the path we nudge toward is the one verify
    later checks on disk."""

    def test_extract_path_detects_write(self) -> None:
        path = _extract_expected_file_path("Write the guide to results/q9_onboarding.md")
        assert path == "results/q9_onboarding.md"

    def test_extract_path_detects_export(self) -> None:
        path = _extract_expected_file_path("Export the report to reports/q3.pdf")
        assert path == "reports/q3.pdf"

    def test_extract_path_none_for_non_write_step(self) -> None:
        """A non-write step never triggers a nudge."""
        assert _extract_expected_file_path("Analyze the requirements") is None

    def test_extract_path_none_for_empty(self) -> None:
        assert _extract_expected_file_path("") is None

    def test_extract_path_rejects_e_g_abbreviation(self) -> None:
        """F-k: 'e.g' (a 1-char 'extension') must not be captured as a
        deliverable. q3 narrated 'write the evolved tool script ... e.g. at a
        threshold' and the old regex grabbed 'e.g', feeding a bogus missing-
        deliverable verify loop."""
        step = (
            "Generate the mutation: write the evolved tool script (v2) that "
            "emits churn flags e.g. at a configurable threshold"
        )
        assert _extract_expected_file_path(step) is None

    def test_extract_path_rejects_numeric_version_fragment(self) -> None:
        """F-k: a version/section fragment like '1.0' or '3.2' (1-char tail)
        is not a file deliverable."""
        assert _extract_expected_file_path("Write version 1.0 to the notes") is None
        assert _extract_expected_file_path("Export see section 3.2 for details") is None

    def test_extract_path_still_finds_real_extension(self) -> None:
        """F-k regression guard: real >=2-char extensions are still captured."""
        assert (
            _extract_expected_file_path("Write the matrix to results/q03/retention.csv")
            == "results/q03/retention.csv"
        )

    def test_called_file_output_true_for_file_writer(self) -> None:
        tcs = [{"id": "1", "type": "function", "function": {"name": "file_writer", "arguments": "{}"}}]
        assert _called_file_output_tool(tcs) is True

    def test_called_file_output_false_for_other_tools(self) -> None:
        tcs = [{"id": "1", "type": "function", "function": {"name": "code_executor", "arguments": "{}"}}]
        assert _called_file_output_tool(tcs) is False

    def test_called_file_output_false_for_empty(self) -> None:
        assert _called_file_output_tool([]) is False

    def test_first_file_output_call_returns_writer(self) -> None:
        tcs = [
            {"id": "1", "type": "function", "function": {"name": "web_search", "arguments": "{}"}},
            {"id": "2", "type": "function", "function": {"name": "file_writer", "arguments": '{"file_path": "x.md"}'}},
        ]
        fw = _first_file_output_call(tcs)
        assert fw is not None
        assert fw["function"]["name"] == "file_writer"

    def test_first_file_output_call_none_when_absent(self) -> None:
        assert _first_file_output_call(
            [{"id": "1", "function": {"name": "web_search"}}]
        ) is None

    def test_tool_call_args_parses_json(self) -> None:
        tc = {"function": {"name": "file_writer", "arguments": '{"file_path": "a.md", "content": "hi"}'}}
        assert _tool_call_args(tc) == {"file_path": "a.md", "content": "hi"}

    def test_tool_call_args_handles_invalid_json(self) -> None:
        tc = {"function": {"name": "file_writer", "arguments": "not json{"}}
        assert _tool_call_args(tc) == {}

    def test_write_nudge_names_path_and_tool(self) -> None:
        nudge = _write_nudge("results/x.md", compute=False)
        assert "results/x.md" in nudge
        assert "file_writer" in nudge


class TestCodeExecutorWriteSatisfiesWriteStep:
    """battery-04 q1+q3 fix: a write-step whose deliverable lands on disk via
    ``code_executor`` (not ``file_writer``) must NOT trigger the file_writer
    nudge. ``code_executor`` is not in FILE_OUTPUT_TOOLS (no ``file_path`` arg to
    record), but when the agent's generated code writes the deliverable the step
    is genuinely done. Previously the nudge false-fired 3x, execute marked the
    step complete-with-gap, verify flagged the (present) deliverable missing, and
    the run looped to MAX_ITERATIONS. The disk check
    (``_deliverable_on_disk``) recognizes the code_executor-mediated write by its
    outcome — the file now exists — rather than by the tool name."""

    @staticmethod
    def _settings(tmp_path: Any, *, deliverable_on_disk: bool) -> Any:
        from types import SimpleNamespace

        if deliverable_on_disk:
            (tmp_path / "report.md").write_text("# real deliverable\n", encoding="utf-8")
        return SimpleNamespace(
            agent=SimpleNamespace(
                results_root=str(tmp_path),
                workspace_root=str(tmp_path / "workspace"),
                results_per_run_subdir=False,
                max_write_nudges=2,
                # select_tools_for_query (findings-05) reads these off settings.agent.
                tool_retrieval_enabled=False,
                tool_retrieval_top_k=8,
            )
        )

    @pytest.mark.asyncio
    async def test_no_nudge_when_code_executor_wrote_deliverable(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deliverable present on disk + code_executor called (not file_writer)
        → nudge suppressed, step completes on the first attempt (1 gateway call).
        """
        from src.graph.enums import Strategy

        fake = self._settings(tmp_path, deliverable_on_disk=True)
        # execute.py binds get_settings at module level → patch its binding; and
        # _deliverable_on_disk → normalize → _paths reads the source-module
        # binding. Both must point at the fake so the disk check resolves under
        # tmp_path and max_attempts reads max_write_nudges=2.
        monkeypatch.setattr("src.graph.nodes.execute.get_settings", lambda: fake)
        monkeypatch.setattr("src.config.settings.get_settings", lambda: fake)

        state = dict(initial_state("Write a report to results/report.md", "th-ce", 10))
        state["plan_steps"] = [
            PlanStep(
                id="s1",
                description="Save the analysis to results/report.md",
                status="pending",
            )
        ]
        state["current_step_index"] = 0
        state["strategy"] = Strategy.REACT

        gateway = MagicMock()
        gateway.acompletion_with_tools = AsyncMock(return_value=ToolCallResponse(
            content="Wrote the report via code_executor",
            tool_calls=[{
                "id": "tc_ce",
                "type": "function",
                "function": {
                    "name": "code_executor",
                    "arguments": '{"code": "open(\'results/report.md\',\'w\').write(\'# real deliverable\')"}',
                },
            }],
            model="deepseek-v4-flash", provider="deepseek",
            input_tokens=10, output_tokens=20, total_tokens=30, cost_usd=0.0,
        ))
        handler = AsyncMock(return_value="wrote report")
        tools = MagicMock()
        tools.list_tools = MagicMock(return_value=[])
        tools.get_handler = MagicMock(return_value=handler)

        result = await execute_node(state, gateway=gateway, tools=tools)

        # Nudge did NOT fire: exactly ONE gateway call, then break → step done.
        assert gateway.acompletion_with_tools.call_count == 1
        assert result["phase"] == Phase.REFLECT
        assert result["completed_steps"][0].status == GoalStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_nudge_still_fires_when_deliverable_absent(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deliverable genuinely absent + code_executor (no file_writer) → nudge
        still fires on repeated attempts (behavior preserved). The disk check
        only SUPPRESSES the nudge for real writes; it never removes the safety
        net for a step that produced nothing."""
        from src.graph.enums import Strategy

        fake = self._settings(tmp_path, deliverable_on_disk=False)
        monkeypatch.setattr("src.graph.nodes.execute.get_settings", lambda: fake)
        monkeypatch.setattr("src.config.settings.get_settings", lambda: fake)

        state = dict(initial_state("Write a report to results/report.md", "th-ce2", 10))
        state["plan_steps"] = [
            PlanStep(
                id="s1",
                description="Save the analysis to results/report.md",
                status="pending",
            )
        ]
        state["current_step_index"] = 0
        state["strategy"] = Strategy.REACT

        gateway = MagicMock()
        gateway.acompletion_with_tools = AsyncMock(return_value=ToolCallResponse(
            content="I will write it",
            tool_calls=[{
                "id": "tc_ce2",
                "type": "function",
                "function": {
                    "name": "code_executor",
                    "arguments": '{"code": "print(1)"}',
                },
            }],
            model="deepseek-v4-flash", provider="deepseek",
            input_tokens=10, output_tokens=20, total_tokens=30, cost_usd=0.0,
        ))
        handler = AsyncMock(return_value="1")
        tools = MagicMock()
        tools.list_tools = MagicMock(return_value=[])
        tools.get_handler = MagicMock(return_value=handler)

        result = await execute_node(state, gateway=gateway, tools=tools)

        # Nudge fired on repeated attempts (>1 gateway call) then budget-spent
        # break. The step still completes; verify flags the gap downstream.
        assert gateway.acompletion_with_tools.call_count > 1
        assert result["completed_steps"][0].status == GoalStatus.COMPLETED


class TestComputeDeliverableSteering:
    """battery-04 q01 fix: a DATA deliverable (.csv/.jsonl/…) must be produced by
    ``code_executor``, not ``file_writer`` — the model cannot reliably hand-author
    normalized/transformed rows as text. The write-step steering now branches on
    ``_is_compute_deliverable(path)``: compute files steer to code_executor
    (turn-1 step label + a code_executor nudge + NO file_writer tool_choice lock);
    text files (.md/.txt/.json) stay on file_writer (create_dirs + forced
    tool_choice on the nudge turn). Regression for q01 where normalized.csv never
    materialized: the prompt said 'file_writer, NOT code_executor' AND the nudge
    pinned file_writer via tool_choice, so the agent could only narrate the data
    and looped plan→execute→reflect→verify to MAX_ITERATIONS — no eval row."""

    @pytest.mark.parametrize(
        "path",
        [
            "results/q01/normalized.csv",
            "results/q01/events.tsv",
            "results/q01/events.jsonl",
            "results/q01/events.jsonlines",
            "results/q01/data.xlsx",
            "results/q01/data.xls",
            "results/q01/data.parquet",
            "results/q01/data.feather",
        ],
    )
    def test_is_compute_deliverable_true_for_data_exts(self, path: str) -> None:
        assert _is_compute_deliverable(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "results/q01/summary.json",
            "results/q03/overview.md",
            "results/q9_onboarding.md",
            "reports/q3.pdf",
            "results/q01/notes.txt",
        ],
    )
    def test_is_compute_deliverable_false_for_text_exts(self, path: str) -> None:
        assert _is_compute_deliverable(path) is False

    def test_is_compute_deliverable_case_insensitive(self) -> None:
        assert _is_compute_deliverable("RESULTS/Q01/NORMALIZED.CSV") is True

    def test_write_nudge_compute_branch_steers_to_code_executor(self) -> None:
        text = _write_nudge("results/q01/normalized.csv", compute=True)
        low = text.lower()
        assert "code_executor" in low, text
        assert "results/q01/normalized.csv" in text
        # A compute nudge must NOT mention create_dirs/file_writer — that would
        # contradict the code_executor instruction and re-lock the wrong tool.
        assert "create_dirs" not in low, text
        assert "file_writer" not in low, text

    def test_write_nudge_text_branch_steers_to_file_writer(self) -> None:
        text = _write_nudge("results/q01/summary.md", compute=False)
        low = text.lower()
        assert "file_writer" in low, text
        assert "create_dirs" in low, text
        assert "code_executor" not in low, text

    @pytest.mark.asyncio
    async def test_compute_step_steers_to_code_executor_and_skips_file_writer_lock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Turn-1 step label for a .csv deliverable steers to code_executor, and
        a subsequent nudge turn is NOT force-locked to file_writer (so the
        instructed tool stays callable). Text-only turns force the nudge path
        deterministically via the on-disk mock."""
        from src.graph.enums import Strategy

        monkeypatch.setattr(get_settings().agent, "max_write_nudges", 2)
        # Deliverable absent → nudge fires regardless of ambient results/ state.
        monkeypatch.setattr("src.graph.nodes.execute._deliverable_on_disk", lambda _path: False)

        state = dict(initial_state(
            "Ingest+dedupe events, writing results/q01/normalized.csv",
            "th-q01-compute", 10,
        ))
        state["plan_steps"] = [
            PlanStep(
                id="s1",
                description="Write the normalized events to results/q01/normalized.csv",
                status="pending",
            ),
        ]
        state["current_step_index"] = 0
        state["strategy"] = Strategy.REACT

        narration = ToolCallResponse(
            content="I will produce the normalized CSV now.",
            tool_calls=[],
            model="deepseek-v4-flash", provider="deepseek",
            input_tokens=10, output_tokens=10, total_tokens=20, cost_usd=0.0001,
        )
        gateway = MagicMock()
        gateway.acompletion_with_tools = AsyncMock(side_effect=[narration, narration, narration])
        tools = MagicMock()
        tools.list_tools = MagicMock(return_value=[
            {"type": "function", "function": {
                "name": "file_writer",
                "description": "Write a file",
                "parameters": {"type": "object", "properties": {
                    "file_path": {"type": "string"}, "content": {"type": "string"}}},
            }},
            {"type": "function", "function": {
                "name": "code_executor",
                "description": "Run code",
                "parameters": {"type": "object", "properties": {"code": {"type": "string"}}},
            }},
        ])
        tools.get_handler = MagicMock(return_value=AsyncMock(return_value="ok"))

        await execute_node(state, gateway=gateway, tools=tools)

        calls = gateway.acompletion_with_tools.call_args_list
        assert len(calls) >= 2

        # Turn 1: compute step label steers to code_executor, not file_writer.
        turn1_blob = " ".join(str(m) for m in calls[0].kwargs.get("messages", [])).lower()
        assert "code_executor" in turn1_blob, turn1_blob
        assert "produced by running code" in turn1_blob, turn1_blob
        # create_dirs is file_writer-specific — absent from the compute label.
        assert "create_dirs" not in turn1_blob, turn1_blob

        # Turn 2 (nudge): a compute deliverable is NOT force-locked to file_writer
        # (tool_choice stays None so code_executor remains callable), and the
        # nudge text itself names code_executor.
        assert calls[1].kwargs.get("tool_choice") is None
        turn2_blob = " ".join(str(m) for m in calls[1].kwargs.get("messages", [])).lower()
        assert "code_executor" in turn2_blob, turn2_blob
        assert "create_dirs" not in turn2_blob, turn2_blob


class TestGoalDeliverableFallback:
    """A producing step that names no file of its own (e.g. "merge the results
    into a cohesive overview") falls back to the goal's canonical deliverable
    path so file_writer still gets called. This is the Q3 fix: the merge step
    narrated the overview as text and q3_overview.md was never written because
    the step description had no path the nudge could detect."""

    def test_extract_goal_deliverable_from_merge_phrase(self) -> None:
        """The key Q3 case — no save/write verb, but a results/ path exists."""
        goal = "Merge the results into results/q3_overview.md."
        assert _extract_goal_deliverable(goal) == "results/q3_overview.md"

    def test_extract_goal_deliverable_from_save_phrase(self) -> None:
        goal = "save the metrics to results/q1_text_stats.md"
        assert _extract_goal_deliverable(goal) == "results/q1_text_stats.md"

    def test_extract_goal_deliverable_none_when_no_path(self) -> None:
        assert _extract_goal_deliverable("Explain the architecture of the system.") is None

    def test_extract_goal_deliverable_none_for_empty(self) -> None:
        assert _extract_goal_deliverable("") is None

    def test_extract_goal_deliverable_skips_input_path(self) -> None:
        """F-k: a goal names its INPUT first ('Reuse q01's normalizer output at
        results/q01/normalized.csv'). The input path must NOT be returned as the
        deliverable; the LAST output path is (churn_flags.csv)."""
        goal = (
            "Reuse q01's normalizer output at results/q01/normalized.csv. Create a "
            "cohort-retention tool, writing it to results/q03/retention.csv; then "
            "evolve that tool, writing results/q03/churn_flags.csv."
        )
        assert _extract_goal_deliverable(goal) == "results/q03/churn_flags.csv"

    def test_extract_goal_deliverable_returns_last_output(self) -> None:
        """With multiple outputs and no input, the last is the primary output."""
        goal = (
            "write the matrix to results/q03/retention.csv and "
            "results/q03/churn_flags.csv"
        )
        assert _extract_goal_deliverable(goal) == "results/q03/churn_flags.csv"

    def test_extract_goal_deliverable_skips_from_context(self) -> None:
        """An input phrased with 'from' is skipped in favour of the output."""
        goal = "Build a report from results/input.csv, saving results/output.md"
        assert _extract_goal_deliverable(goal) == "results/output.md"

    # ── Bug #5: bare-filename deliverables (no results/ prefix) ──────────

    def test_extract_goal_deliverable_bare_filename_named_cue(self) -> None:
        """Bug #5: a goal naming its output without the results/ prefix
        ("write a CSV file named primes_demo.csv") must still resolve the
        deliverable. Previously the regex required results/ and returned None,
        so the evolution gate never matched a completed bare-filename run."""
        goal = "Write a CSV file named primes_demo.csv with the first 15 primes."
        assert _extract_goal_deliverable(goal) == "primes_demo.csv"

    def test_extract_goal_deliverable_bare_filename_save_cue(self) -> None:
        """A bare filename preceded by a save/write verb resolves."""
        goal = "compute the primes and save primes_demo.csv"
        assert _extract_goal_deliverable(goal) == "primes_demo.csv"

    def test_extract_goal_deliverable_bare_filename_no_cue_returns_none(self) -> None:
        """False-positive guard: a bare known-extension token NOT preceded by an
        output cue is incidental text ("the schema is in schema.md"), not a
        deliverable. Returning None here is correct — neither caller treats it
        as success (nudge runs once; evolution gate uses its other criterion)."""
        goal = "Analyze the engine latency; the schema is documented in schema.md"
        assert _extract_goal_deliverable(goal) is None

    def test_extract_goal_deliverable_ignores_version_and_decimal_tokens(self) -> None:
        """Version strings ("v0.23.0") and decimals are NOT grabbed as paths —
        the bare branch is restricted to known data/text extensions."""
        goal = "Benchmark vLLM v0.23.0 against TGI v0.19.0 and report the ratio as 2.0"
        assert _extract_goal_deliverable(goal) is None

    def test_extract_goal_deliverable_bare_output_wins_over_results_input(self) -> None:
        """A results/-prefixed INPUT (read-context) is skipped in favour of a
        bare-filename OUTPUT named later, proving the two shapes cooperate."""
        goal = "read results/input.csv and write a summary to brief.md"
        assert _extract_goal_deliverable(goal) == "brief.md"

    def test_is_producing_step_true_for_merge(self) -> None:
        assert _is_producing_step(
            "Merge all three processed results into a cohesive Q3 overview", 0, 2
        ) is True

    def test_is_producing_step_true_for_last_step(self) -> None:
        """Last step is treated as producing even without a merge verb."""
        assert _is_producing_step("Verify the overview exists", 2, 3) is True

    def test_is_producing_step_false_for_read_step(self) -> None:
        assert _is_producing_step("Read CLAUDE.md to extract architecture", 0, 3) is False

    def test_is_producing_step_false_for_list_step(self) -> None:
        assert _is_producing_step("List contents of src/tools/builtin", 1, 3) is False

    def test_deliverable_on_disk_true_when_present(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "q3_overview.md").write_text("# Overview\n")
        fake = MagicMock()
        fake.agent.results_root = str(tmp_path)
        monkeypatch.setattr("src.config.settings.get_settings", lambda: fake)
        assert _deliverable_on_disk("results/q3_overview.md") is True

    def test_deliverable_on_disk_false_when_missing(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = MagicMock()
        fake.agent.results_root = str(tmp_path)
        monkeypatch.setattr("src.config.settings.get_settings", lambda: fake)
        assert _deliverable_on_disk("results/absent.md") is False

    def test_deliverable_on_disk_false_when_empty(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "empty.md").write_text("")
        fake = MagicMock()
        fake.agent.results_root = str(tmp_path)
        monkeypatch.setattr("src.config.settings.get_settings", lambda: fake)
        assert _deliverable_on_disk("results/empty.md") is False

    def _merge_step_state(self) -> dict[str, Any]:
        """Single merge step (last step) with a path-free description but a goal
        that names results/q3_overview.md."""
        state = dict(initial_state(
            "Use sub-agents in parallel; merge the results into results/q3_overview.md",
            "q3-thread",
            25,
        ))
        state["plan_steps"] = [
            PlanStep(id="s1", description="Merge all three processed results into a cohesive Q3 overview"),
        ]
        state["current_step_index"] = 0
        return state

    def _file_writer_response(self) -> ToolCallResponse:
        return ToolCallResponse(
            content="Writing the merged overview now.",
            tool_calls=[{
                "id": "tc_fw",
                "type": "function",
                "function": {
                    "name": "file_writer",
                    "arguments": '{"file_path": "results/q3_overview.md", "content": "# Q3 Overview\\n\\nMerged."}',
                },
            }],
            model="deepseek-v4-flash",
            provider="deepseek",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            cost_usd=0.0001,
        )

    def _text_only_response(self) -> ToolCallResponse:
        return ToolCallResponse(
            content="Here is the merged Q3 overview described in text: # Q3 Overview ...",
            tool_calls=[],
            model="deepseek-v4-flash",
            provider="deepseek",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            cost_usd=0.0001,
        )

    def _tools_with_writer(self) -> Any:
        file_handler = AsyncMock(return_value="Successfully wrote 48 bytes to results/q3_overview.md")
        tools = MagicMock()
        tools.list_tools = MagicMock(return_value=[{
            "type": "function",
            "function": {
                "name": "file_writer",
                "description": "Write content to a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                },
            },
        }])
        tools.get_handler = MagicMock(return_value=file_handler)
        return tools

    @pytest.mark.asyncio
    async def test_merge_step_falls_back_to_goal_deliverable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A path-free merge step + goal naming results/q3_overview.md: text-only
        turn 1 → nudge → file_writer turn 2 writes the goal deliverable."""
        # Force the "deliverable not yet on disk" condition so the fall-back fires
        # deterministically, independent of ambient results/ filesystem state.
        monkeypatch.setattr(
            "src.graph.nodes.execute._deliverable_on_disk", lambda _path: False
        )
        gateway = MagicMock()
        gateway.acompletion_with_tools = AsyncMock(side_effect=[
            self._text_only_response(),
            self._file_writer_response(),
        ])
        tools = self._tools_with_writer()

        result = await execute_node(self._merge_step_state(), gateway=gateway, tools=tools)

        # file_writer called with the GOAL's deliverable path (the fall-back target).
        handler = tools.get_handler.return_value
        handler.assert_awaited_once()
        assert handler.call_args.kwargs.get("file_path") == "results/q3_overview.md"

        # Two LLM turns (text-only, then nudge→writer).
        assert gateway.acompletion_with_tools.await_count == 2

        completed = result["completed_steps"][0]
        assert completed.status == GoalStatus.COMPLETED
        assert completed.tool_name == "file_writer"
        assert completed.tool_input.get("file_path") == "results/q3_overview.md"

    @pytest.mark.asyncio
    async def test_no_fallback_when_deliverable_already_on_disk(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Once the deliverable exists on disk, the producing step runs once with
        no goal-deliverable nudge (avoids re-nudging after a prior step wrote it)."""
        monkeypatch.setattr(
            "src.graph.nodes.execute._deliverable_on_disk", lambda _path: True
        )
        gateway = MagicMock()
        gateway.acompletion_with_tools = AsyncMock(return_value=self._text_only_response())
        tools = self._tools_with_writer()

        result = await execute_node(self._merge_step_state(), gateway=gateway, tools=tools)

        # Single LLM call — no nudge because the file is already on disk.
        assert gateway.acompletion_with_tools.await_count == 1
        completed = result["completed_steps"][0]
        assert completed.tool_name != "file_writer"

    @pytest.mark.asyncio
    async def test_no_fallback_when_goal_has_no_deliverable(self) -> None:
        """A producing step whose goal names no file deliverable runs once (no
        nudge) — the fall-back only applies to goals that embed a results/ path."""
        state = dict(initial_state("Explain the system architecture clearly.", "explain-thread", 25))
        state["plan_steps"] = [
            PlanStep(id="s1", description="Merge the findings into a clear explanation"),
        ]
        state["current_step_index"] = 0
        gateway = MagicMock()
        gateway.acompletion_with_tools = AsyncMock(return_value=self._text_only_response())
        tools = self._tools_with_writer()

        result = await execute_node(state, gateway=gateway, tools=tools)

        assert gateway.acompletion_with_tools.await_count == 1
        completed = result["completed_steps"][0]
        assert completed.tool_name != "file_writer"


class TestWriteStepNudge:
    """Write-steps (a declared file deliverable) must produce the file via
    file_writer. When the LLM narrates the deliverable as text on turn 1,
    execute re-prompts with a bounded nudge instead of marking the step
    complete with no artifact — the Q9 failure where haiku took ~4 step
    attempts before it called file_writer, burning reflect/plan cycles."""

    def _write_step_state(self) -> dict[str, Any]:
        state = dict(initial_state("Produce the onboarding guide", "write-thread", 25))
        state["plan_steps"] = [
            PlanStep(id="s1", description="Write the onboarding guide to results/q9_onboarding.md"),
        ]
        state["current_step_index"] = 0
        return state

    def _file_writer_response(self) -> ToolCallResponse:
        return ToolCallResponse(
            content="Writing the onboarding guide now.",
            tool_calls=[{
                "id": "tc_fw",
                "type": "function",
                "function": {
                    "name": "file_writer",
                    "arguments": '{"file_path": "results/q9_onboarding.md", "content": "# Onboarding\\n\\nWelcome."}',
                },
            }],
            model="claude-haiku-4-5-20251001",
            provider="anthropic",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            cost_usd=0.0001,
        )

    def _text_only_response(self) -> ToolCallResponse:
        return ToolCallResponse(
            content="Here is the onboarding guide described in text: # Onboarding ...",
            tool_calls=[],
            model="claude-haiku-4-5-20251001",
            provider="anthropic",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            cost_usd=0.0001,
        )

    def _tools_with_writer(self) -> Any:
        file_handler = AsyncMock(return_value="Wrote results/q9_onboarding.md (4673 bytes)")
        tools = MagicMock()
        tools.list_tools = MagicMock(return_value=[{
            "type": "function",
            "function": {
                "name": "file_writer",
                "description": "Write content to a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                },
            },
        }])
        tools.get_handler = MagicMock(return_value=file_handler)
        return tools

    @pytest.mark.asyncio
    async def test_nudge_makes_llm_call_file_writer_on_second_attempt(self) -> None:
        """Text-only turn 1 → nudge → file_writer on turn 2: deliverable produced."""
        gateway = MagicMock()
        gateway.acompletion_with_tools = AsyncMock(side_effect=[
            self._text_only_response(),
            self._file_writer_response(),
        ])
        tools = self._tools_with_writer()

        result = await execute_node(self._write_step_state(), gateway=gateway, tools=tools)

        # file_writer was actually invoked with the parsed file_path.
        handler = tools.get_handler.return_value
        handler.assert_awaited_once()
        assert handler.call_args.kwargs.get("file_path") == "results/q9_onboarding.md"

        # Exactly two LLM turns (text-only, then nudge→writer).
        assert gateway.acompletion_with_tools.await_count == 2

        # Step completed with the writer's tool_input recorded for verify.
        assert result["phase"] == Phase.REFLECT
        assert result["current_step_index"] == 1
        completed = result["completed_steps"][0]
        assert completed.status == GoalStatus.COMPLETED
        assert completed.tool_name == "file_writer"
        assert completed.tool_input.get("file_path") == "results/q9_onboarding.md"

        tr = result["tool_results"][0]
        assert tr.tool_name == "file_writer"
        assert tr.success is True

    @pytest.mark.asyncio
    async def test_write_step_calls_file_writer_first_try_no_nudge(self) -> None:
        """file_writer on turn 1: a single LLM call, no nudge."""
        gateway = MagicMock()
        gateway.acompletion_with_tools = AsyncMock(return_value=self._file_writer_response())
        tools = self._tools_with_writer()

        result = await execute_node(self._write_step_state(), gateway=gateway, tools=tools)

        assert gateway.acompletion_with_tools.await_count == 1
        assert result["current_step_index"] == 1
        completed = result["completed_steps"][0]
        assert completed.tool_name == "file_writer"

    @pytest.mark.asyncio
    async def test_write_step_bounded_when_llm_never_calls_file_writer(self) -> None:
        """LLM never calls file_writer → bounded to max_write_nudges+1 calls (no
        infinite loop). Step still marked complete (graceful degrade); a later
        verify pass will flag the missing deliverable rather than hanging."""
        gateway = MagicMock()
        gateway.acompletion_with_tools = AsyncMock(return_value=self._text_only_response())
        tools = self._tools_with_writer()

        result = await execute_node(self._write_step_state(), gateway=gateway, tools=tools)

        max_nudges = get_settings().agent.max_write_nudges
        assert gateway.acompletion_with_tools.await_count == max_nudges + 1
        # Degrades gracefully: step completes (no hang), but no writer recorded.
        assert result["phase"] == Phase.REFLECT
        assert result["current_step_index"] == 1
        completed = result["completed_steps"][0]
        assert completed.status == GoalStatus.COMPLETED
        assert completed.tool_name != "file_writer"

    @pytest.mark.asyncio
    async def test_nudge_message_appears_in_second_payload(self) -> None:
        """The nudge is a user-role message appended after the text-only turn."""
        gateway = MagicMock()
        gateway.acompletion_with_tools = AsyncMock(side_effect=[
            self._text_only_response(),
            self._file_writer_response(),
        ])
        tools = self._tools_with_writer()

        await execute_node(self._write_step_state(), gateway=gateway, tools=tools)

        second_payload = gateway.acompletion_with_tools.call_args_list[1].kwargs["messages"]
        nudge_bodies = [m["content"] for m in second_payload if m.get("role") == "user"]
        assert any("file_writer" in body and "did not" in body for body in nudge_bodies)

    @pytest.mark.asyncio
    async def test_failed_file_writer_call_does_not_advance(self) -> None:
        """A failed file_writer call is a recoverable tool failure (F14 path),
        not a nudge trigger — the step stays ACTIVE and retries via execute."""
        gateway = MagicMock()
        gateway.acompletion_with_tools = AsyncMock(return_value=ToolCallResponse(
            content="Writing",
            tool_calls=[{
                "id": "tc_fw_fail",
                "type": "function",
                "function": {
                    "name": "file_writer",
                    "arguments": '{"file_path": "results/q9_onboarding.md", "content": "x"}',
                },
            }],
            model="claude-haiku-4-5-20251001",
            provider="anthropic",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            cost_usd=0.0001,
        ))
        tools = MagicMock()
        tools.list_tools = MagicMock(return_value=[])
        failing_handler = AsyncMock(side_effect=RuntimeError("disk full"))
        tools.get_handler = MagicMock(return_value=failing_handler)

        result = await execute_node(self._write_step_state(), gateway=gateway, tools=tools)

        # F14: recoverable failure → ACTIVE, no advance, loop back to execute.
        assert result["phase"] == Phase.EXECUTE
        assert result["current_step_index"] == 0
        assert "completed_steps" not in result
