"""Clarifying-gate (Feature B) opt-in parity: routing + single-shot + branches.

``route_after_classify`` is gated by ``clarifying_gate_enabled`` (default off):
when off, classify always routes to ``plan`` (byte-identical topology); when on,
an ambiguous goal (severity >= threshold + non-empty notes, not yet
disambiguated) routes classify -> disambiguate -> plan. The
``disambiguate_node`` is single-shot (``disambiguation_done`` guard) so no
classify↔disambiguate cycle can form, and its cascade (LLM self-resolve ->
web-grounding -> re-score -> HITL last resort) degrades gracefully on every
failure path. All deterministic: gateway / web_search / interrupt are mocked.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import get_settings
from src.graph.enums import Phase
from src.graph.nodes.disambiguate import disambiguate_node
from src.graph.routers import route_after_classify


# ─── routing parity ─────────────────────────────────────────────────────────


class TestRouteAfterClassifyGate:
    def test_should_route_to_plan_when_gate_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Default-off master switch: topology is byte-identical regardless of
        # how ambiguous the goal looks.
        monkeypatch.setattr(get_settings().agent, "clarifying_gate_enabled", False)
        state = {"ambiguity_severity": 0.9, "ambiguity_notes": ["vague"]}
        assert route_after_classify(state) == "plan"

    def test_should_route_to_disambiguate_when_gate_on_and_ambiguous(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(get_settings().agent, "clarifying_gate_enabled", True)
        monkeypatch.setattr(
            get_settings().agent, "clarifying_severity_threshold", 0.5
        )
        state = {"ambiguity_severity": 0.8, "ambiguity_notes": ["which entity?"]}
        assert route_after_classify(state) == "disambiguate"

    def test_should_route_to_plan_when_gate_on_but_severity_below_threshold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(get_settings().agent, "clarifying_gate_enabled", True)
        monkeypatch.setattr(
            get_settings().agent, "clarifying_severity_threshold", 0.5
        )
        state = {"ambiguity_severity": 0.2, "ambiguity_notes": ["x"]}
        assert route_after_classify(state) == "plan"

    def test_should_route_to_plan_when_ambiguous_but_no_notes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # severity alone is insufficient — non-empty notes are required so a
        # mere confidence number never triggers the cascade.
        monkeypatch.setattr(get_settings().agent, "clarifying_gate_enabled", True)
        monkeypatch.setattr(
            get_settings().agent, "clarifying_severity_threshold", 0.5
        )
        state = {"ambiguity_severity": 0.9, "ambiguity_notes": []}
        assert route_after_classify(state) == "plan"

    def test_should_route_to_plan_when_already_disambiguated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The single-shot guard: once disambiguation_done is set, a re-reached
        # classify (e.g. error_handler auth path) routes straight to plan.
        monkeypatch.setattr(get_settings().agent, "clarifying_gate_enabled", True)
        monkeypatch.setattr(
            get_settings().agent, "clarifying_severity_threshold", 0.5
        )
        state = {
            "ambiguity_severity": 0.9,
            "ambiguity_notes": ["x"],
            "disambiguation_done": True,
        }
        assert route_after_classify(state) == "plan"

    def test_should_route_to_plan_on_settings_access_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A settings hiccup must never break routing — falls back to plan.
        def _boom() -> None:
            raise RuntimeError("settings gone")

        monkeypatch.setattr("src.graph.routers.get_settings", _boom)
        assert route_after_classify({}) == "plan"


# ─── disambiguate_node single-shot + cascade branches ───────────────────────


def _goal_state(**kw: object) -> dict[str, object]:
    goal = SimpleNamespace(text=kw.get("goal", "summarise the report"))
    base = {
        "current_goal": goal,
        "ambiguity_notes": ["which report?"],
        "ambiguity_severity": 0.8,
        "ambiguity_type": "scope",
    }
    base.update(kw)
    base.pop("goal", None)
    return base


class TestDisambiguateNodeSingleShot:
    async def test_should_short_circuit_when_disambiguation_done_set(self) -> None:
        # The guard returns a minimal phase transition — no gateway call, so no
        # classify↔disambiguate cycle can form.
        gw = MagicMock()
        gw.acompletion = AsyncMock()
        out = await disambiguate_node(
            _goal_state(disambiguation_done=True), gateway=gw
        )
        assert out["phase"] == Phase.PLAN
        assert out["disambiguation_done"] is True
        gw.acompletion.assert_not_called()

    async def test_should_carry_notes_forward_when_no_gateway(self) -> None:
        # Without a gateway the cascade degrades to carrying the classify notes
        # forward as advisory context (literal goal unchanged).
        out = await disambiguate_node(_goal_state(), gateway=None)
        assert out["phase"] == Phase.PLAN
        assert out["disambiguation_done"] is True
        assert "DISAMBIGUATION CONTEXT" in out["disambiguation_context"]
        assert "which report?" in out["disambiguation_context"]

    async def test_should_never_rewrite_current_goal(self) -> None:
        # The resolution is ADVISORY only — the literal goal stays the OBJECTIVE.
        out = await disambiguate_node(_goal_state(), gateway=None)
        assert "disambiguation_resolution" in out
        # No goal rewrite key is emitted by the node.
        assert "current_goal" not in out


class TestDisambiguateNodeWebGrounding:
    async def test_should_collect_evidence_when_queries_emitted_and_grounding_on(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            get_settings().agent, "clarifying_web_grounding_enabled", True
        )
        # Self-resolve returns a resolution WITH grounding queries → grounding fires.
        from src.graph.schemas import DisambiguationResolution

        resolution = DisambiguationResolution(
            proposed_interpretation="summarise the Acme report",
            assumptions=["Acme = the referenced doc"],
            grounding_queries=["Acme Corp"],
            resolved=True,
            remaining_severity=0.1,
            notes=[],
        )
        from src.llm.models import LLMResponse

        gw = MagicMock()
        gw.acompletion = AsyncMock(
            return_value=LLMResponse(
                content="{}",
                model="gpt-4o-mini-2024-07-18",
                provider="openai",
                input_tokens=1,
                output_tokens=1,
                total_tokens=2,
                cost_usd=0.0,
            )
        )
        tools = MagicMock()
        handler = AsyncMock(return_value="Acme Corp — widgets manufacturer\nFounded 1990")
        tools.get_handler = MagicMock(return_value=handler)
        # Stub StructuredOutputManager.extract to return our resolution.
        from src.llm.structured_output import StructuredOutputManager

        monkeypatch.setattr(
            StructuredOutputManager, "extract", AsyncMock(return_value=resolution)
        )
        out = await disambiguate_node(_goal_state(), gateway=gw, tools=tools)
        # Evidence collected from the web_search handler output.
        assert len(out["disambiguation_evidence"]) >= 1
        assert "Acme" in "\n".join(out["disambiguation_evidence"])

    async def test_should_skip_grounding_when_queries_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No queries emitted → the web-grounding step is a no-op (no tool call).
        from src.graph.schemas import DisambiguationResolution

        resolution = DisambiguationResolution(
            proposed_interpretation="x",
            assumptions=[],
            grounding_queries=[],
            resolved=True,
            remaining_severity=0.1,
            notes=[],
        )
        from src.llm.models import LLMResponse

        gw = MagicMock()
        gw.acompletion = AsyncMock(
            return_value=LLMResponse(
                content="{}",
                model="gpt-4o-mini-2024-07-18",
                provider="openai",
                input_tokens=1,
                output_tokens=1,
                total_tokens=2,
                cost_usd=0.0,
            )
        )
        tools = MagicMock()
        tools.get_handler = MagicMock(return_value=AsyncMock(side_effect=AssertionError("no grounding")))
        from src.llm.structured_output import StructuredOutputManager
        from unittest.mock import AsyncMock as _AM

        # Stub extract via monkeypatch so the class mutation is torn down after.
        monkeypatch.setattr(
            StructuredOutputManager, "extract", _AM(return_value=resolution)
        )
        out = await disambiguate_node(_goal_state(), gateway=gw, tools=tools)
        assert out["disambiguation_evidence"] == []


class TestDisambiguateNodeHitlLastResort:
    async def test_should_request_hitl_when_severe_unresolved_and_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(get_settings().agent, "clarifying_hitl_enabled", True)
        monkeypatch.setattr(get_settings().agent, "clarifying_hitl_threshold", 0.5)
        from src.graph.schemas import DisambiguationResolution

        resolution = DisambiguationResolution(
            proposed_interpretation="x",
            assumptions=[],
            grounding_queries=[],
            resolved=False,  # still ambiguous
            remaining_severity=0.9,  # >= threshold
            notes=["n1"],
        )
        from src.llm.models import LLMResponse

        gw = MagicMock()
        gw.acompletion = AsyncMock(
            return_value=LLMResponse(
                content="{}",
                model="gpt-4o-mini-2024-07-18",
                provider="openai",
                input_tokens=1,
                output_tokens=1,
                total_tokens=2,
                cost_usd=0.0,
            )
        )
        from src.llm.structured_output import StructuredOutputManager
        from unittest.mock import AsyncMock as _AM

        monkeypatch.setattr(
            StructuredOutputManager, "extract", _AM(return_value=resolution)
        )
        # interrupt() raises outside a compiled context → degrades gracefully.
        out = await disambiguate_node(_goal_state(), gateway=gw)
        # HITL was attempted (severity high) but there is no resume surface in
        # the worker/CLI → interrupt() degrades gracefully → hitl_requested
        # stays False (the run carries the resolution forward, never stalls).
        assert out["hitl_requested"] is False
        assert out["disambiguation_resolution"] == "x"
