"""Tests for the multi-hop research loop node + its retrieve_memory router (Phase 5a).

The loop is opt-in (``research_loop_enabled``, default off). These tests pin its
contract with deterministic fakes (no live LLM): the loop is bounded by
``research_max_hops``, accumulates distilled findings into ``research_context``,
stops early on ``sufficient`` / empty / duplicate next-query, degrades to a no-op
when the gateway is missing or a refine call fails, and is idempotent. The router
returns ``"structure_analysis"`` unless the flag is on AND research hasn't run.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.graph.enums import Phase
from src.graph.nodes.research import research_node
from src.graph.routers import route_after_retrieve_memory
from src.graph.state import AgentState


# ── Fakes ───────────────────────────────────────────────────────────────


class _FakeGateway:
    """Returns canned refine-JSON responses in order; records call count."""

    def __init__(self, contents: list[str]) -> None:
        self._contents = list(contents)
        self.calls = 0

    async def acompletion(self, messages: Any = None, **kwargs: Any) -> Any:
        content = (
            self._contents[self.calls]
            if self.calls < len(self._contents)
            else "{}"
        )
        self.calls += 1
        return SimpleNamespace(content=content)


def _refine_json(
    *,
    sufficient: bool = False,
    next_query: str = "",
    findings: list[str] | None = None,
) -> str:
    """Build a clean ResearchRefine JSON blob (no markdown fences)."""
    import json

    return json.dumps(
        {
            "sufficient": sufficient,
            "next_query": next_query,
            "findings": findings or [],
        }
    )


class _FakeTools:
    """ToolRegistry stub: only ``web_search`` is loaded; returns a per-query snippet."""

    def __init__(self) -> None:
        self.retrieved_queries: list[str] = []

    def get_handler(self, name: str) -> Any:
        if name != "web_search":
            return None

        async def _handler(queries: list[str], max_results: int) -> str:
            self.retrieved_queries.append(queries[0])
            return f"evidence about {queries[0]}"

        return _handler


def _state(*, goal: str = "Summarize the state of fusion energy", refined_intent: str = "") -> AgentState:
    # AgentState is total=False, so a partial dict is a valid state. The node +
    # router only read these keys (objective_goal_text prefers submitted_goal).
    state: AgentState = {"submitted_goal": goal}
    if refined_intent:
        state["refined_intent"] = refined_intent
    return state


@pytest.fixture
def _research_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable the research loop with fast, deterministic bounds."""
    from src.config import get_settings

    agent = get_settings().agent
    monkeypatch.setattr(agent, "research_loop_enabled", True)
    monkeypatch.setattr(agent, "research_max_hops", 2)
    monkeypatch.setattr(agent, "research_top_k", 3)
    monkeypatch.setattr(agent, "research_max_tokens", 256)


# ── Node: loop control ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_loop_runs_all_hops_when_never_sufficient(_research_on: None) -> None:
    """Never-sufficient + always-fresh next query → bounded to research_max_hops."""
    gateway = _FakeGateway(
        [
            _refine_json(sufficient=False, next_query="follow-up beta", findings=["fact-1"]),
            _refine_json(sufficient=False, next_query="follow-up gamma", findings=["fact-2"]),
            # a third would exceed max_hops=2; included only to prove it's never reached
            _refine_json(sufficient=False, next_query="follow-up delta", findings=["fact-3"]),
        ]
    )
    tools = _FakeTools()

    result = await research_node(_state(), gateway=gateway, tools=tools)

    assert result["research_done"] is True
    assert result["phase"] == Phase.RESEARCH
    assert gateway.calls == 2  # bounded at max_hops=2, not 3
    assert tools.retrieved_queries == ["Summarize the state of fusion energy", "follow-up beta"]
    ctx = result["research_context"]
    assert "fact-1" in ctx and "fact-2" in ctx
    assert "fact-3" not in ctx  # third hop never ran


@pytest.mark.asyncio
async def test_loop_stops_when_sufficient(_research_on: None) -> None:
    """A ``sufficient`` refine decision stops the loop after one hop."""
    gateway = _FakeGateway([_refine_json(sufficient=True, findings=["enough"])])
    tools = _FakeTools()

    result = await research_node(_state(), gateway=gateway, tools=tools)

    assert gateway.calls == 1
    assert "enough" in result["research_context"]


@pytest.mark.asyncio
async def test_loop_stops_on_empty_next_query(_research_on: None) -> None:
    """No next query (but not sufficient) still stops the loop."""
    gateway = _FakeGateway(
        [_refine_json(sufficient=False, next_query="", findings=["only-hop"])]
    )

    result = await research_node(_state(), gateway=gateway, tools=_FakeTools())

    assert gateway.calls == 1
    assert "only-hop" in result["research_context"]


@pytest.mark.asyncio
async def test_loop_stops_on_duplicate_next_query(_research_on: None) -> None:
    """A next_query that was already run stops the loop (no re-querying)."""
    # refined_intent seeds the first query as "alpha"; refine echoes it back.
    gateway = _FakeGateway(
        [_refine_json(sufficient=False, next_query="alpha", findings=["one"])]
    )
    tools = _FakeTools()

    result = await research_node(
        _state(refined_intent="alpha"), gateway=gateway, tools=tools
    )

    assert gateway.calls == 1
    assert tools.retrieved_queries == ["alpha"]  # never re-ran "alpha"
    assert "one" in result["research_context"]


# ── Node: graceful degradation ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_gateway_is_noop(_research_on: None) -> None:
    """A missing gateway degrades to an empty research_context (never breaks)."""
    result = await research_node(_state(), gateway=None, tools=_FakeTools())

    assert result["research_done"] is True
    assert result["research_context"] == ""


@pytest.mark.asyncio
async def test_refine_failure_stops_gracefully(_research_on: None) -> None:
    """A refine LLM failure stops the loop with whatever was gathered (run survives)."""

    class _BoomGateway(_FakeGateway):
        async def acompletion(self, messages: Any = None, **kwargs: Any) -> Any:
            self.calls += 1
            raise RuntimeError("gateway down")

    result = await research_node(
        _state(), gateway=_BoomGateway([]), tools=_FakeTools()
    )

    assert result["research_done"] is True
    assert result["research_context"] == ""  # nothing distilled before the failure


@pytest.mark.asyncio
async def test_idempotent_guard_skips_re_run(_research_on: None) -> None:
    """An already-done research step short-circuits without calling the gateway."""
    gateway = _FakeGateway([_refine_json(sufficient=True, findings=["x"])])

    state = _state()
    state["research_done"] = True
    result = await research_node(state, gateway=gateway, tools=_FakeTools())

    assert gateway.calls == 0
    assert result == {"phase": Phase.RESEARCH, "research_done": True}


# ── Router ──────────────────────────────────────────────────────────────


def test_router_off_returns_structure_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.config import get_settings

    monkeypatch.setattr(get_settings().agent, "research_loop_enabled", False)
    assert route_after_retrieve_memory(_state()) == "structure_analysis"


def test_router_on_not_done_returns_research(_research_on: None) -> None:
    assert route_after_retrieve_memory(_state()) == "research"


def test_router_on_done_returns_structure_analysis(_research_on: None) -> None:
    state = _state()
    state["research_done"] = True
    assert route_after_retrieve_memory(state) == "structure_analysis"
