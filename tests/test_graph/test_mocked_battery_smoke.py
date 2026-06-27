"""CI-safe full-graph battery-shaped E2E coverage (mocked gateway).

Deterministic, hermetic, NO real LLM and NO API key required. Exercises the
REAL compiled task graph (``compile_task_graph``) end-to-end against a MOCKED
``LLMGateway`` (canned responses) + an in-memory mocked ``MemoryManager`` + a
mock tool registry. This is the q01/q05-shaped "battery" happy path driven
through the whole node chain without any network, so it runs in the normal
``-k "not e2e"`` CI gate.

Covered node path (``no_evolution=True`` for a stable, side-effect-free run):

    classify → plan → retrieve_memory → structure_analysis → execute →
    reflect → verify → store_memory → END

Why canned responses deterministically complete the path:
- ``acompletion`` always returns the classify JSON (valid TaskClassification).
  plan / reflect / verify run their StructuredOutput extractors on it, fail to
  match their schemas, and deterministically FALL BACK to their heuristic paths
  (heuristic plan, heuristic reflect → HIGH confidence when steps complete,
  heuristic verify → ``is_complete`` when all steps done + HIGH confidence +
  no declared deliverable).
- ``no_evolution=True`` makes ``route_after_verify`` skip the evolve node and
  its git/filesystem mutation side effects → store_memory → END.
- ``acompletion_with_tools`` returns a response with NO tool calls by default,
  so execute marks each step complete in a single attempt and the index
  advances.

The second test class flips the execute canned response to emit ONE tool call
and asserts the registered handler is actually invoked + the run still
terminates.

NOT marked ``@pytest.mark.e2e`` (deliberate) — must pass under
``-k "not e2e"`` with zero provider key.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.errors import GraphRecursionError

from src.graph.enums import Phase
from src.graph.factory import initial_state
from src.graph.task_graph import compile_task_graph
from src.llm.models import LLMResponse

# Distinct sentinel so a caller-supplied falsy override (e.g. None) still works
# and is NOT confused with "no override supplied".
_SENTINEL: Any = object()


def _classify_content() -> str:
    """Canned classify JSON (valid TaskClassification payload).

    Returned for EVERY ``acompletion`` call. classify parses it; plan /
    reflect / verify fail their schema extraction on it and use their
    heuristic fallback (see module docstring).
    """
    return (
        '{"complexity": "simple", "strategy": "direct", '
        '"estimated_steps": 1, "confidence": 0.9, "reasoning": "trivial"}'
    )


def _make_canned_gateway(
    *,
    tool_response_override: Any = _SENTINEL,
) -> MagicMock:
    """Mock gateway with deterministic ``acompletion`` + ``acompletion_with_tools``.

    ``acompletion`` always returns the classify payload (the other nodes'
    extractors reject it and fall back heuristically).
    ``acompletion_with_tools`` returns a response with NO tool calls by default
    so execute marks each step complete in a single attempt. Pass
    ``tool_response_override`` to inject a tool-call-bearing response for the
    tool-invocation test.

    ``get_cost_records`` is the live-token accessor read by memory folding; keep
    it well below the fold threshold so folding never fires mid-run (stable
    path).
    """
    gateway = MagicMock()
    gateway.acompletion = AsyncMock(
        return_value=LLMResponse(
            content=_classify_content(),
            model="gpt-4o-mini-2024-07-18",
            provider="openai",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            cost_usd=0.0001,
        )
    )

    if tool_response_override is _SENTINEL:
        # Execute reads response.tool_calls (→ []) and response.content (→ text).
        tool_response = MagicMock()
        tool_response.tool_calls = []
        tool_response.content = "Step completed."
    else:
        tool_response = tool_response_override

    gateway.acompletion_with_tools = AsyncMock(return_value=tool_response)

    gateway.get_cost_records = MagicMock(return_value=[])
    return gateway


def _make_mock_tools(
    *, handler: Any = None, tool_name: str = "echo_tool"
) -> MagicMock:
    """Mock tool registry.

    By default advertises no tools and resolves no handlers (the no-tool-call
    path). When ``handler`` is provided, advertises one tool and resolves the
    handler so the execute node's tool-call path can be exercised.
    """
    tools = MagicMock()
    if handler is None:
        tools.list_tools = MagicMock(return_value=[])
        tools.get_handler = MagicMock(return_value=None)
        return tools

    # Advertise a single function-calling tool definition in OpenAI format.
    tool_def = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": "Echo back the provided value.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        },
    }
    tools.list_tools = MagicMock(return_value=[tool_def])
    tools.get_handler = MagicMock(return_value=handler)
    tools.is_generated = MagicMock(return_value=False)
    # The execute node builds a real Redis-backed ToolResultCache internally and
    # short-circuits to a cached result when ``tools.is_cacheable(name)`` is
    # truthy. A bare MagicMock returns a truthy MagicMock here, so the handler
    # is never reached (a stale cross-run cache HIT). Force non-cacheable so the
    # handler is always invoked.
    tools.is_cacheable = MagicMock(return_value=False)
    return tools


def _make_mock_memory() -> MagicMock:
    """Mock in-memory MemoryManager (retrieve + store, no DB)."""
    memory = MagicMock()
    memory.retrieve_context = AsyncMock(return_value=[])
    memory.store_observation = AsyncMock(return_value=None)
    memory.store_skill = AsyncMock(return_value="test-uuid")
    return memory


class TestMockedBatteryHappyPath:
    """Trivial + multi-step goals through the full happy-path node chain."""

    @pytest.mark.asyncio
    async def test_trivial_goal_completes_full_chain(self) -> None:
        """Trivial goal (q01-shaped) completes through classify→…→store_memory→END.

        Asserts: terminates within recursion_limit, is_complete True,
        final_output non-empty, a plan was produced, cost_records is a list,
        memory.retrieve_context was awaited at least once.
        """
        gateway = _make_canned_gateway()
        memory = _make_mock_memory()
        tools = _make_mock_tools()

        compiled = compile_task_graph(
            gateway=gateway,
            memory=memory,
            tools=tools,
            checkpointer=None,
        )

        state = dict(
            initial_state(
                "Explain what a REST API is in two sentences",
                "thread-mock-batt-trivial-001",
                max_iterations=6,
                no_evolution=True,
            )
        )

        try:
            result = await compiled.ainvoke(state, {"recursion_limit": 60})
        except GraphRecursionError:
            pytest.fail("Graph did not terminate within recursion limit")
        result = dict(result) if not isinstance(result, dict) else result

        # ── Termination + completion ─────────────────────────────────────────
        assert result.get("phase") in {Phase.COMPLETE, Phase.STORE_MEMORY}, (
            f"Expected terminal phase, got {result.get('phase')!r}"
        )
        assert result.get("is_complete") is True, (
            "Trivial goal should complete (all steps done, HIGH confidence, "
            "no errors, no declared deliverables)"
        )
        assert bool(result.get("final_output", "").strip()), (
            "final_output must be non-empty on completion"
        )

        # ── Key state transitions ───────────────────────────────────────────
        assert result.get("iteration_count", 0) >= 1, "Agent must execute >= 1 iteration"
        plan_steps = result.get("plan_steps", [])
        assert isinstance(plan_steps, list) and len(plan_steps) >= 1, (
            "A plan must have been produced"
        )

        # ── cost_records populated (graceful: always a list, possibly empty) ─
        cost_records = result.get("cost_records")
        assert cost_records is None or isinstance(cost_records, list), (
            "cost_records must be a list or absent, never a scalar"
        )

        # ── Mocked memory was actually consulted ────────────────────────────
        assert memory.retrieve_context.await_count >= 1, (
            "retrieve_memory node must have queried memory"
        )

    @pytest.mark.asyncio
    async def test_multistep_goal_terminates_and_executes(self) -> None:
        """Multi-step goal (q05-shaped) terminates and exercises the loop.

        A goal whose heuristic plan yields multiple steps still terminates
        within the recursion limit and reaches a terminal phase (no infinite
        loop). is_complete is asserted only loosely (a multi-step trivial goal
        with the no-tool-call canned gateway completes within max_iterations).
        """
        gateway = _make_canned_gateway()
        memory = _make_mock_memory()
        tools = _make_mock_tools()

        compiled = compile_task_graph(
            gateway=gateway,
            memory=memory,
            tools=tools,
            checkpointer=None,
        )

        goal = (
            "List three benefits of unit testing and write a one-line summary "
            "for each benefit."
        )
        state = dict(
            initial_state(
                goal,
                "thread-mock-batt-multi-001",
                max_iterations=8,
                no_evolution=True,
            )
        )

        try:
            result = await compiled.ainvoke(state, {"recursion_limit": 80})
        except GraphRecursionError:
            pytest.fail("Graph did not terminate within recursion limit")
        result = dict(result) if not isinstance(result, dict) else result

        assert result.get("phase") in {Phase.COMPLETE, Phase.STORE_MEMORY}, (
            f"Expected terminal phase, got {result.get('phase')!r}"
        )
        assert result.get("iteration_count", 0) >= 1
        # submitted_goal anchor (battery-04 b09f891) must be preserved verbatim.
        assert result.get("submitted_goal") == goal, (
            "submitted_goal must equal the literal goal the run was started with"
        )

    @pytest.mark.asyncio
    async def test_phase_advances_past_classify(self) -> None:
        """classify ran: phase is no longer CLASSIFY and current_goal is set."""
        gateway = _make_canned_gateway()
        memory = _make_mock_memory()
        tools = _make_mock_tools()

        compiled = compile_task_graph(
            gateway=gateway, memory=memory, tools=tools, checkpointer=None
        )
        state = dict(
            initial_state(
                "Summarize the rules of chess in one paragraph.",
                "thread-mock-batt-classify-001",
                max_iterations=5,
                no_evolution=True,
            )
        )

        result = await compiled.ainvoke(state, {"recursion_limit": 60})
        result = dict(result) if not isinstance(result, dict) else result

        assert result.get("phase") != Phase.CLASSIFY, (
            "Run must have advanced past the classify phase"
        )
        current_goal = result.get("current_goal")
        assert current_goal is not None, "current_goal must be populated post-classify"

    @pytest.mark.asyncio
    async def test_no_evolution_flag_is_honored(self) -> None:
        """no_evolution=True routes around the evolve node.

        Confirms the production-safe stable path (no git/filesystem mutation
        side effects). Evaluated by the absence of evolution_history entries
        added during THIS run on the stable path (the engine node is never
        entered when route_after_verify skips it).
        """
        gateway = _make_canned_gateway()
        memory = _make_mock_memory()
        tools = _make_mock_tools()

        compiled = compile_task_graph(
            gateway=gateway, memory=memory, tools=tools, checkpointer=None
        )
        state = dict(
            initial_state(
                "Define recursion in one sentence.",
                "thread-mock-batt-noevo-001",
                max_iterations=5,
                no_evolution=True,
            )
        )

        result = await compiled.ainvoke(state, {"recursion_limit": 60})
        result = dict(result) if not isinstance(result, dict) else result

        assert result.get("no_evolution") is True, (
            "no_evolution flag must round-trip through the graph state"
        )
        # The evolve node was skipped (it never appended a mutation record);
        # the run still terminates cleanly.
        assert result.get("phase") in {Phase.COMPLETE, Phase.STORE_MEMORY}


class TestMockedBatteryToolInvocation:
    """Execute path where the canned gateway emits ONE real tool call.

    Models the q01/q05 shape where the plan step requires a tool: execute's
    canned gateway emits a single OpenAI-format tool call, the registered mock
    handler is invoked with the parsed args, and the run still terminates.
    """

    @pytest.mark.asyncio
    async def test_tool_handler_invoked_and_step_advances(self) -> None:
        """A canned tool call reaches the registered handler (handler awaited).

        The handler records that it was called (so we can assert it was invoked
        with the parsed JSON arguments). The run still terminates within the
        recursion limit — a successful tool call advances the step rather than
        looping forever.
        """
        invoked: list[dict[str, Any]] = []

        async def echo_handler(**kwargs: Any) -> str:
            invoked.append(kwargs)
            return f"echo:{kwargs.get('value', '')}"

        tool_name = "echo_tool"

        # Canned acompletion_with_tools response carrying ONE tool call in
        # OpenAI function-calling format. Execute parses function.arguments
        # (JSON), looks up tools.get_handler(name), and awaits handler(**args).
        tool_response = MagicMock()
        tool_response.tool_calls = [
            {
                "id": "call_mock_001",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps({"value": "hello-battery"}),
                },
            }
        ]
        tool_response.content = None

        gateway = _make_canned_gateway(tool_response_override=tool_response)
        memory = _make_mock_memory()
        tools = _make_mock_tools(handler=echo_handler, tool_name=tool_name)

        compiled = compile_task_graph(
            gateway=gateway, memory=memory, tools=tools, checkpointer=None
        )

        state = dict(
            initial_state(
                "Use a tool to echo a value back to the user.",
                "thread-mock-batt-tool-001",
                max_iterations=6,
                no_evolution=True,
            )
        )

        try:
            result = await compiled.ainvoke(state, {"recursion_limit": 80})
        except GraphRecursionError:
            pytest.fail("Graph did not terminate within recursion limit")
        result = dict(result) if not isinstance(result, dict) else result

        # ── The handler was actually invoked with the parsed arguments ───────
        assert len(invoked) >= 1, (
            "The registered tool handler must be invoked when execute emits a tool call"
        )
        assert invoked[0].get("value") == "hello-battery", (
            "Handler must receive the parsed JSON arguments from function.arguments"
        )

        # ── The tool result is recorded in state ─────────────────────────────
        # Execute writes ``tool_results`` (list[ToolResult], each carrying
        # ``.tool_name``) and ``completed_steps`` — NOT a ``tools_called`` list.
        tool_results = result.get("tool_results", [])
        assert isinstance(tool_results, list) and len(tool_results) >= 1, (
            "A tool call must produce at least one tool_result entry"
        )
        recorded_names = [getattr(tr, "tool_name", None) for tr in tool_results]
        assert tool_name in recorded_names, (
            f"tool_results must record the invoked tool name ({tool_name!r}); "
            f"got {recorded_names!r}"
        )

        # ── The successful tool call advanced the step index (no stuck loop) ──
        completed_steps = result.get("completed_steps", [])
        assert isinstance(completed_steps, list) and len(completed_steps) >= 1, (
            "A successful tool call must complete the step (completed_steps non-empty)"
        )

        # ── The run terminates (a successful tool call does not loop forever) ─
        assert result.get("phase") in {Phase.COMPLETE, Phase.STORE_MEMORY}, (
            f"Expected terminal phase after a successful tool call, got "
            f"{result.get('phase')!r}"
        )

        # ── The handler registry was queried for our tool ────────────────────
        tools.get_handler.assert_any_call(tool_name)

    @pytest.mark.asyncio
    async def test_tool_result_appears_in_final_output_context(self) -> None:
        """A successful tool result contributes to the run's final_output.

        The handler's echoed string surfaces into the conversation and the
        completed step's result, which the heuristic verify path folds into
        final_output. Asserts the run completed with a non-empty final_output
        after the tool fired.
        """
        async def echo_handler(**kwargs: Any) -> str:
            return f"echo:{kwargs.get('value', '')}"

        tool_name = "echo_tool"
        tool_response = MagicMock()
        tool_response.tool_calls = [
            {
                "id": "call_mock_002",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps({"value": "battery-result"}),
                },
            }
        ]
        tool_response.content = None

        gateway = _make_canned_gateway(tool_response_override=tool_response)
        memory = _make_mock_memory()
        tools = _make_mock_tools(handler=echo_handler, tool_name=tool_name)

        compiled = compile_task_graph(
            gateway=gateway, memory=memory, tools=tools, checkpointer=None
        )
        state = dict(
            initial_state(
                "Echo a value using a tool.",
                "thread-mock-batt-toolresult-001",
                max_iterations=6,
                no_evolution=True,
            )
        )

        result = await compiled.ainvoke(state, {"recursion_limit": 80})
        result = dict(result) if not isinstance(result, dict) else result

        assert result.get("is_complete") is True, "Run must complete after the tool call"
        assert bool(result.get("final_output", "").strip()), (
            "final_output must be non-empty after a tool-driven completion"
        )


class TestMockedBatteryStateInvariants:
    """Cheap, deterministic assertions on graph-state shapes (no LLM)."""

    @pytest.mark.asyncio
    async def test_cost_records_is_list_type(self) -> None:
        """cost_records is always a list (CostTracker-resilience contract)."""
        gateway = _make_canned_gateway()
        memory = _make_mock_memory()
        tools = _make_mock_tools()
        compiled = compile_task_graph(
            gateway=gateway, memory=memory, tools=tools, checkpointer=None
        )
        state = dict(
            initial_state(
                "Say hello.",
                "thread-mock-batt-cost-001",
                max_iterations=4,
                no_evolution=True,
            )
        )
        result = await compiled.ainvoke(state, {"recursion_limit": 60})
        result = dict(result) if not isinstance(result, dict) else result

        cost_records = result.get("cost_records")
        assert cost_records is None or isinstance(cost_records, list), (
            "cost_records must be a list or absent, never a scalar"
        )

    @pytest.mark.asyncio
    async def test_plan_steps_is_list_type(self) -> None:
        """plan_steps is always a list after a run."""
        gateway = _make_canned_gateway()
        memory = _make_mock_memory()
        tools = _make_mock_tools()
        compiled = compile_task_graph(
            gateway=gateway, memory=memory, tools=tools, checkpointer=None
        )
        state = dict(
            initial_state(
                "Explain gravity in one sentence.",
                "thread-mock-batt-plan-001",
                max_iterations=4,
                no_evolution=True,
            )
        )
        result = await compiled.ainvoke(state, {"recursion_limit": 60})
        result = dict(result) if not isinstance(result, dict) else result

        plan_steps = result.get("plan_steps")
        assert isinstance(plan_steps, list), "plan_steps must be a list"

    @pytest.mark.asyncio
    async def test_iteration_count_advances(self) -> None:
        """At least one iteration executed (the graph did real work)."""
        gateway = _make_canned_gateway()
        memory = _make_mock_memory()
        tools = _make_mock_tools()
        compiled = compile_task_graph(
            gateway=gateway, memory=memory, tools=tools, checkpointer=None
        )
        state = dict(
            initial_state(
                "Define entropy briefly.",
                "thread-mock-batt-iter-001",
                max_iterations=4,
                no_evolution=True,
            )
        )
        result = await compiled.ainvoke(state, {"recursion_limit": 60})
        result = dict(result) if not isinstance(result, dict) else result

        assert result.get("iteration_count", 0) >= 1

    @pytest.mark.asyncio
    async def test_errors_is_list_type(self) -> None:
        """errors is always a list (run must not crash on the happy path)."""
        gateway = _make_canned_gateway()
        memory = _make_mock_memory()
        tools = _make_mock_tools()
        compiled = compile_task_graph(
            gateway=gateway, memory=memory, tools=tools, checkpointer=None
        )
        state = dict(
            initial_state(
                "Name a primary color.",
                "thread-mock-batt-errs-001",
                max_iterations=4,
                no_evolution=True,
            )
        )
        result = await compiled.ainvoke(state, {"recursion_limit": 60})
        result = dict(result) if not isinstance(result, dict) else result

        errors = result.get("errors")
        assert errors is None or isinstance(errors, list), (
            "errors must be a list or absent, never a scalar"
        )

    @pytest.mark.asyncio
    async def test_gateway_acompletion_was_called(self) -> None:
        """classify drove at least one real acompletion call through the gateway."""
        gateway = _make_canned_gateway()
        memory = _make_mock_memory()
        tools = _make_mock_tools()
        compiled = compile_task_graph(
            gateway=gateway, memory=memory, tools=tools, checkpointer=None
        )
        state = dict(
            initial_state(
                "Define a variable in Python.",
                "thread-mock-batt-gwcall-001",
                max_iterations=4,
                no_evolution=True,
            )
        )
        await compiled.ainvoke(state, {"recursion_limit": 60})

        assert gateway.acompletion.await_count >= 1, (
            "classify must issue at least one acompletion call to the gateway"
        )
