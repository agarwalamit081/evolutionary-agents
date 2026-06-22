"""Tests for the Feature B ambiguity-resolution cascade.

Covers three surfaces:
  - ``route_after_classify`` — the default-off gate that keeps the topology
    byte-identical (classify -> plan) until toggled on.
  - ``disambiguate_node`` — the graduated ladder (LLM self-resolve -> web
    grounding -> re-score -> HITL last resort), carried forward as ADVISORY
    context. The literal goal is NEVER rewritten.
  - graph wiring — the disambiguate node is registered and the graph compiles.

All deterministic: the gateway, the ``web_search`` handler, and
``langgraph.types.interrupt`` are mocked. Zero real-LLM spend.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import get_settings
from src.graph.enums import Phase
from src.graph.factory import initial_state
from src.graph.nodes.disambiguate import disambiguate_node
from src.graph.routers import route_after_classify
from src.graph.task_graph import build_task_graph, compile_task_graph
from src.llm.models import LLMResponse


def _mock_gateway(content: str) -> MagicMock:
    """Build a gateway mock whose acompletion returns the given JSON string.

    Mirrors the helper in test_classify.py — valid JSON parses on the first
    StructuredOutputManager.extract attempt, so no gateway retry is exercised.
    """
    gw = MagicMock()
    gw.acompletion = AsyncMock(return_value=LLMResponse(
        content=content,
        model="gpt-4o-mini-2024-07-18",
        provider="openai",
        input_tokens=10,
        output_tokens=50,
        total_tokens=60,
        cost_usd=0.0001,
    ))
    return gw


def _mock_tools(web_return: str = "Acme Corp — a widgets manufacturer (src)") -> MagicMock:
    """A ToolRegistry mock whose web_search handler returns the given text."""
    tools = MagicMock()
    handler = AsyncMock(return_value=web_return)
    tools.get_handler = MagicMock(return_value=handler)
    return tools


# A clean DisambiguationResolution payload: a proposed interpretation, two
# grounding queries (so the web-grounding step fires), resolved, low severity.
_RESOLVE_JSON = (
    '{"proposed_interpretation": "summarise the Acme Corp document", '
    '"assumptions": ["the user has one document in mind"], '
    '"grounding_queries": ["Acme Corp about", "Acme Corp company"], '
    '"resolved": true, "remaining_severity": 0.1, '
    '"notes": []}'
)


class TestRouteAfterClassify:
    """The default-off gate — the key backward-compat assertion lives here."""

    def test_defaults_to_plan_when_gate_off(self) -> None:
        """Gate off (the default) -> always plan, regardless of ambiguity."""
        state = initial_state("summarise this", "thread-router-1")
        state["ambiguity_severity"] = 0.9
        state["ambiguity_notes"] = ["'this' has no referent"]
        # Default config has clarifying_gate_enabled=False.
        assert route_after_classify(state) == "plan"

    def test_routes_to_disambiguate_when_enabled_and_ambiguous(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Gate on + severity >= threshold + non-empty notes -> disambiguate."""
        monkeypatch.setattr(get_settings().agent, "clarifying_gate_enabled", True)
        state = initial_state("summarise this", "thread-router-2")
        state["ambiguity_severity"] = 0.8
        state["ambiguity_notes"] = ["'this' has no referent"]
        assert route_after_classify(state) == "disambiguate"

    def test_below_threshold_routes_to_plan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Gate on but severity below threshold -> plan (not ambiguous enough)."""
        monkeypatch.setattr(get_settings().agent, "clarifying_gate_enabled", True)
        state = initial_state("summarise this", "thread-router-3")
        state["ambiguity_severity"] = 0.2  # < default 0.5 threshold
        state["ambiguity_notes"] = ["minor ambiguity"]
        assert route_after_classify(state) == "plan"

    def test_empty_notes_routes_to_plan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Gate on + high severity but NO notes -> plan (nothing to resolve)."""
        monkeypatch.setattr(get_settings().agent, "clarifying_gate_enabled", True)
        state = initial_state("summarise this", "thread-router-4")
        state["ambiguity_severity"] = 0.9
        state["ambiguity_notes"] = []
        assert route_after_classify(state) == "plan"

    def test_single_shot_guard_routes_to_plan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """disambiguation_done set -> plan (no classify<->disambiguate cycle)."""
        monkeypatch.setattr(get_settings().agent, "clarifying_gate_enabled", True)
        state = initial_state("summarise this", "thread-router-5")
        state["ambiguity_severity"] = 0.9
        state["ambiguity_notes"] = ["'this' has no referent"]
        state["disambiguation_done"] = True
        assert route_after_classify(state) == "plan"


class TestDisambiguateNodeGuards:
    """Loop-safety and no-deps fallback paths."""

    @pytest.mark.asyncio
    async def test_single_shot_no_recall(self) -> None:
        """disambiguation_done set -> straight to PLAN, no gateway call."""
        state = initial_state("summarise this", "thread-shot")
        state["disambiguation_done"] = True
        gw = _mock_gateway(_RESOLVE_JSON)
        result = await disambiguate_node(state, gateway=gw)
        assert result["phase"] == Phase.PLAN
        assert result["disambiguation_done"] is True
        gw.acompletion.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_gateway_carries_notes_forward(self) -> None:
        """No gateway -> the classify notes are carried forward as advisory."""
        state = initial_state("summarise this", "thread-nogw")
        state["ambiguity_notes"] = ["'this' has no referent", "no output format"]
        result = await disambiguate_node(state, gateway=None)
        assert result["phase"] == Phase.PLAN
        assert result["disambiguation_done"] is True
        # The unresolved notes surface in the advisory context.
        ctx = result["disambiguation_context"]
        assert "'this' has no referent" in ctx
        assert "no output format" in ctx

    @pytest.mark.asyncio
    async def test_no_goal_carries_notes_forward(self) -> None:
        """Missing goal -> graceful carry-forward, never an exception."""
        state = initial_state("summarise this", "thread-nogoal")
        state["current_goal"] = None
        state["ambiguity_notes"] = ["'this' has no referent"]
        result = await disambiguate_node(state, gateway=_mock_gateway(_RESOLVE_JSON))
        assert result["phase"] == Phase.PLAN
        assert result["disambiguation_done"] is True


class TestDisambiguateCascade:
    """The full graduated ladder with mocked gateway + web_search."""

    @pytest.mark.asyncio
    async def test_cascade_resolves_and_grounds(self) -> None:
        """self-resolve -> web grounding -> re-score; advisory carried forward."""
        state = initial_state("summarise this", "thread-cascade")
        state["refined_intent"] = "summarise the referenced document"
        state["ambiguity_type"] = "referential"
        state["ambiguity_severity"] = 0.8
        state["ambiguity_notes"] = ["'this' has no referent"]
        tools = _mock_tools()
        result = await disambiguate_node(
            state, gateway=_mock_gateway(_RESOLVE_JSON), tools=tools,
        )

        assert result["phase"] == Phase.PLAN
        assert result["disambiguation_done"] is True
        # The web_search handler was invoked with the emitted queries (batched).
        tools.get_handler.assert_called_with("web_search")
        handler = tools.get_handler.return_value
        handler.assert_awaited()
        # Evidence collected from the grounding step.
        assert result["disambiguation_evidence"], "grounding evidence should be populated"
        # The resolution + assumptions carry forward.
        assert "Acme Corp" in result["disambiguation_resolution"]
        assert result["disambiguation_assumptions"]
        # Advisory context is populated and framed as hypotheses.
        ctx = result["disambiguation_context"]
        assert "DISAMBIGUATION CONTEXT" in ctx
        assert "Proposed interpretation" in ctx

    @pytest.mark.asyncio
    async def test_does_not_rewrite_literal_goal(self) -> None:
        """Drift guard: current_goal.text is NEVER replaced by the resolution."""
        goal_text = "summarise this"
        state = initial_state(goal_text, "thread-drift")
        state["ambiguity_severity"] = 0.8
        state["ambiguity_notes"] = ["'this' has no referent"]
        result = await disambiguate_node(
            state, gateway=_mock_gateway(_RESOLVE_JSON), tools=_mock_tools(),
        )
        # The node returns advisory fields but never a new current_goal.
        assert "current_goal" not in result
        assert state["current_goal"].text == goal_text

    @pytest.mark.asyncio
    async def test_no_grounding_queries_skips_web_search(self) -> None:
        """When the LLM emits no grounding queries, web_search is not called."""
        no_query_json = (
            '{"proposed_interpretation": "summarise the document", '
            '"assumptions": [], "grounding_queries": [], '
            '"resolved": true, "remaining_severity": 0.1, "notes": []}'
        )
        state = initial_state("write a poem", "thread-noq")
        state["ambiguity_severity"] = 0.6
        state["ambiguity_notes"] = ["no length given"]
        tools = _mock_tools()
        result = await disambiguate_node(
            state, gateway=_mock_gateway(no_query_json), tools=tools,
        )
        assert result["phase"] == Phase.PLAN
        handler = tools.get_handler.return_value
        handler.assert_not_called()
        assert result["disambiguation_evidence"] == []

    @pytest.mark.asyncio
    async def test_grounding_failure_is_isolated(self) -> None:
        """A web_search exception yields empty evidence, never an abort."""
        tools = MagicMock()
        tools.get_handler = MagicMock(return_value=AsyncMock(side_effect=RuntimeError("boom")))
        state = initial_state("summarise this", "thread-groundfail")
        state["ambiguity_severity"] = 0.8
        state["ambiguity_notes"] = ["'this' has no referent"]
        result = await disambiguate_node(
            state, gateway=_mock_gateway(_RESOLVE_JSON), tools=tools,
        )
        assert result["phase"] == Phase.PLAN
        assert result["disambiguation_done"] is True
        assert result["disambiguation_evidence"] == []

    @pytest.mark.asyncio
    async def test_hitl_degrades_gracefully_when_unavailable(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """HITL enabled but interrupt() raises (no resume surface) -> carry on.

        Mirrors hitl_gate_node: there is no Command(resume=) path in the
        worker/CLI today, so a RuntimeError from interrupt() degrades to
        advisory (hitl_requested=False) instead of stalling the run.
        """
        import langgraph.types as lgt

        def _no_resume(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("no resume surface")

        monkeypatch.setattr(get_settings().agent, "clarifying_hitl_enabled", True)
        monkeypatch.setattr(lgt, "interrupt", _no_resume)
        severe_json = (
            '{"proposed_interpretation": "unclear", "assumptions": [], '
            '"grounding_queries": [], "resolved": false, '
            '"remaining_severity": 0.95, "notes": ["cannot resolve"]}'
        )
        state = initial_state("do the thing", "thread-hitl")
        state["ambiguity_severity"] = 0.9
        state["ambiguity_notes"] = ["'the thing' undefined"]
        result = await disambiguate_node(state, gateway=_mock_gateway(severe_json))

        assert result["phase"] == Phase.PLAN
        assert result["disambiguation_done"] is True
        # HITL could not fire (no resume surface) -> did not stall.
        assert result["hitl_requested"] is False
        # The proposed interpretation still reaches the planner as advisory.
        ctx = result["disambiguation_context"]
        assert "DISAMBIGUATION CONTEXT" in ctx
        assert "unclear" in ctx


class TestDisambiguateGraphWiring:
    """The node is registered and the graph still compiles (topology intact)."""

    def test_disambiguate_node_registered(self) -> None:
        """build_task_graph registers the disambiguate node."""
        graph = build_task_graph()
        assert "disambiguate" in set(graph.nodes.keys())

    def test_graph_compiles_with_disambiguate(self) -> None:
        """The graph still compiles (classify<->disambiguate edge is valid)."""
        compiled = compile_task_graph()
        assert hasattr(compiled, "ainvoke")

    def test_disambiguate_node_wired_to_plan(self) -> None:
        """A direct edge disambiguate -> plan exists (no dead-end node)."""
        graph = build_task_graph()
        # langgraph StateGraph.edges is a set of (src, dst) tuples for static
        # edges (the classify->disambiguate hop is conditional, in .branches;
        # the disambiguate->plan hop is the static one we assert here).
        edges = getattr(graph, "edges", set()) or set()
        assert ("disambiguate", "plan") in edges, (
            f"no disambiguate->plan static edge; edges={edges}"
        )
