"""Deterministic multi-node graph integration test.

Drives the REAL compiled task graph (built by ``compile_task_graph``) through a
trivial goal with a MOCKED gateway (deterministic canned responses) + mocked
in-memory memory + a mock tool registry, and asserts the run TERMINATES within a
bounded recursion limit and that key state transitions occur.

Scoped node path (trivial goal, ``no_evolution=True`` to keep the path stable
and side-effect-free):

    classify → plan → retrieve_memory → structure_analysis → execute →
    reflect → verify → store_memory → END

Why this scope: the canned gateway returns a single classify-shaped JSON for
every ``acompletion`` call. plan / reflect / verify run their StructuredOutput
extractor on that content, fail to match their schemas, and deterministically
FALL BACK to their heuristic paths (heuristic plan, heuristic reflect → HIGH
confidence when steps complete, heuristic verify → is_complete when all steps
are done with no errors and HIGH confidence). ``no_evolution=True`` makes
``route_after_verify`` skip the evolve node (and its git/filesystem mutation
side effects) and go straight to store_memory → END. execute's LLM path is
given a canned ``acompletion_with_tools`` response with NO tool calls, so each
step is marked complete in a single attempt and the index advances. This yields
a fully deterministic completion with ≥3 real nodes exercised.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.errors import GraphRecursionError

from src.graph.enums import Phase
from src.graph.factory import initial_state
from src.graph.task_graph import compile_task_graph


def _classify_content() -> str:
    """Canned classify JSON (valid TaskClassification payload).

    Returned for EVERY ``acompletion`` call. classify parses it; plan / reflect
    / verify fail their schema extraction on it and use their heuristic fallback.
    """
    return (
        '{"complexity": "simple", "strategy": "direct", '
        '"estimated_steps": 1, "confidence": 0.9, "reasoning": "trivial"}'
    )


def _make_canned_gateway() -> MagicMock:
    """Mock gateway with deterministic ``acompletion`` + ``acompletion_with_tools``.

    ``acompletion`` always returns the classify payload (see module docstring for
    why the other nodes' extractors reject it and fall back heuristically).
    ``acompletion_with_tools`` returns a response with NO tool calls so execute
    marks each step complete in a single attempt (no deliverable nudges for a
    trivial, non-file-producing goal).
    """
    from src.llm.models import LLMResponse

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

    # Execute reads response.tool_calls (→ []) and response.content (→ text).
    tool_response = MagicMock()
    tool_response.tool_calls = []
    tool_response.content = "Step completed."
    gateway.acompletion_with_tools = AsyncMock(return_value=tool_response)

    # Live-token accessor read by memory folding; keep it well below the fold
    # threshold so folding does not fire mid-run (keeps the path stable).
    gateway.get_cost_records = MagicMock(return_value=[])
    return gateway


def _make_mock_tools() -> MagicMock:
    """Mock tool registry advertising one tool but resolving no handlers.

    Execute needs ``tools`` non-None to take its LLM path; the canned gateway
    never emits tool calls, so handlers are never invoked.
    """
    tools = MagicMock()
    tools.list_tools = MagicMock(return_value=[])
    tools.get_handler = MagicMock(return_value=None)
    return tools


def _make_mock_memory() -> MagicMock:
    """Mock in-memory MemoryManager (retrieve + store, no DB)."""
    memory = MagicMock()
    memory.retrieve_context = AsyncMock(return_value=[])
    memory.store_observation = AsyncMock(return_value=None)
    memory.store_skill = AsyncMock(return_value="test-uuid")
    return memory


class TestGraphIntegrationFlow:
    """Compile + run the real graph on a trivial goal with mocked deps."""

    @pytest.mark.asyncio
    async def test_trivial_goal_terminates_and_transitions(self) -> None:
        """Trivial goal completes through classify→…→store_memory→END.

        Asserts:
        - the run terminates (no GraphRecursionError) within the recursion limit,
        - is_complete ends True with a non-empty final_output,
        - iteration_count advanced (>= 1) and a plan was produced,
        - a terminal phase was reached (not stuck in CLASSIFY/EXECUTE).
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
                "thread-int-flow-001",
                max_iterations=6,
                no_evolution=True,
            )
        )

        # Bounded execution — a non-terminating graph raises GraphRecursionError.
        try:
            result = await compiled.ainvoke(state, {"recursion_limit": 60})
        except GraphRecursionError:
            pytest.fail("Graph did not terminate within recursion limit")
        result = dict(result) if not isinstance(result, dict) else result

        # ── Termination + completion ────────────────────────────────────────
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

        # ── Mocked memory was actually consulted (retrieve + store path ran) ─
        assert memory.retrieve_context.await_count >= 1, (
            "retrieve_memory node must have queried memory"
        )
