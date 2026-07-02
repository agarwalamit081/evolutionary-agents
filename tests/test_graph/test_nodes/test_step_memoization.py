"""Tests for C1 step-output memoization (default-off).

Covers the three integration points:
  * ``execute_node`` skip path — a step whose description-hash is in
    ``step_outputs`` is not re-run when ``step_memoization_enabled`` is on.
  * ``_llm_execute`` record path — a successful LLM step records its result
    under the description-hash, merging prior outputs (overwrite semantics).
  * ``verify._stamp_verify_cycle`` clear path — the memo is cleared only on a
    goal-gap re-plan (verify rejects + plan exhausted), the one path where prior
    outputs are suspect; it is preserved mid-plan and on completion.

All deterministic: ``step_memoization_enabled`` is toggled via ``monkeypatch`` on
the cached settings singleton (default-off, so the rest of the suite is
unaffected); the LLM path mocks ``acompletion_with_tools``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config.settings import get_settings
from src.graph.enums import GoalStatus, Phase
from src.graph.models import PlanStep
from src.graph.nodes.execute import _step_cache_key, execute_node
from src.graph.nodes.verify import _stamp_verify_cycle
from src.llm.models import ToolCallResponse


# ── helpers ────────────────────────────────────────────────────────────────


def _enable_memo(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    """Toggle AgentSettings.step_memoization_enabled on the cached singleton."""
    monkeypatch.setattr(get_settings().agent, "step_memoization_enabled", enabled)


def _llm_gateway_tools() -> tuple[MagicMock, MagicMock]:
    """A gateway + tools pair whose single ``code_executor`` call completes
    cleanly, so ``execute_node`` reaches the LLM success-return (where the
    memo record fires). Mirrors the established execute-test fixture."""
    gateway = MagicMock()
    gateway.acompletion_with_tools = AsyncMock(return_value=ToolCallResponse(
        content="ran code",
        tool_calls=[{
            "id": "tc1",
            "type": "function",
            "function": {
                "name": "code_executor",
                "arguments": '{"code": "print(42)"}',
            },
        }],
        model="glm-4.7",
        provider="zai",
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        cost_usd=0.0001,
    ))
    tools = MagicMock()
    tools.list_tools = MagicMock(return_value=[{
        "type": "function",
        "function": {
            "name": "code_executor",
            "description": "Execute code",
            "parameters": {"type": "object", "properties": {"code": {"type": "string"}}},
        },
    }])
    tools.get_handler = MagicMock(return_value=AsyncMock(return_value="42"))
    return gateway, tools


# ── _step_cache_key ────────────────────────────────────────────────────────


class TestStepCacheKey:
    def test_stable_for_same_description(self) -> None:
        assert _step_cache_key("Fetch the data") == _step_cache_key("Fetch the data")

    def test_differs_for_different_description(self) -> None:
        assert _step_cache_key("Fetch the data") != _step_cache_key("Fetch other data")

    def test_is_short_hex(self) -> None:
        key = _step_cache_key("x")
        assert len(key) == 16
        assert all(c in "0123456789abcdef" for c in key)


# ── execute_node skip path ─────────────────────────────────────────────────


class TestExecuteSkip:
    @pytest.mark.asyncio
    async def test_skip_when_enabled_and_cached(
        self, state_with_plan: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Enabled + cache hit → step is skipped: COMPLETED with the cached
        result, index advanced, and the gateway is never called."""
        _enable_memo(monkeypatch, True)
        desc = state_with_plan["plan_steps"][0].description
        key = _step_cache_key(desc)
        state_with_plan["step_outputs"] = {key: "CACHED_RESULT_TEXT"}

        gateway = MagicMock()
        result = await execute_node(state_with_plan, gateway=gateway, tools=MagicMock())

        # Gateway never reached (the skip short-circuits before the LLM call).
        gateway.acompletion_with_tools.assert_not_called()
        assert result["phase"] is Phase.REFLECT
        assert result["current_step_index"] == 1
        completed = result["completed_steps"]
        assert len(completed) == 1
        assert completed[0].status is GoalStatus.COMPLETED
        assert completed[0].result == "CACHED_RESULT_TEXT"
        # A synthesized turn carries the cached result so dependent steps see it.
        assert any("CACHED_RESULT_TEXT" in str(m) for m in result["messages"])

    @pytest.mark.asyncio
    async def test_disabled_does_not_skip(
        self, state_with_plan: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default-off: a populated cache is ignored; the step executes
        (simulated fallback here, since no gateway)."""
        _enable_memo(monkeypatch, False)
        desc = state_with_plan["plan_steps"][0].description
        state_with_plan["step_outputs"] = {_step_cache_key(desc): "CACHED"}

        result = await execute_node(state_with_plan)

        assert result["phase"] is Phase.REFLECT
        completed = result["completed_steps"]
        assert completed[0].result != "CACHED"  # simulated result, not the memo
        # The simulated fallback does not record into step_outputs.
        assert "step_outputs" not in result

    @pytest.mark.asyncio
    async def test_no_skip_when_enabled_but_not_cached(
        self, state_with_plan: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Enabled + cache miss (empty memo) → the step executes normally."""
        _enable_memo(monkeypatch, True)
        state_with_plan["step_outputs"] = {}

        result = await execute_node(state_with_plan)

        assert result["phase"] is Phase.REFLECT
        assert result["current_step_index"] == 1


# ── _llm_execute record path ───────────────────────────────────────────────


class TestExecuteRecord:
    @pytest.mark.asyncio
    async def test_record_on_success(
        self, state_with_plan: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A successful LLM step records its result under the description-hash."""
        _enable_memo(monkeypatch, True)
        # Neutral description so the write-step nudge does not engage.
        state_with_plan["plan_steps"][0] = PlanStep(id="s1", description="Compute the sum")
        desc = state_with_plan["plan_steps"][0].description

        gateway, tools = _llm_gateway_tools()
        result = await execute_node(state_with_plan, gateway=gateway, tools=tools)

        assert "step_outputs" in result
        key = _step_cache_key(desc)
        assert key in result["step_outputs"]
        # result_text = response_content ("ran code"); truncated to [:500].
        assert result["step_outputs"][key] == "ran code"

    @pytest.mark.asyncio
    async def test_record_merges_prior_outputs(
        self, state_with_plan: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Recording merges into the prior memo (overwrite semantics) — an
        unrelated cached entry is preserved, not dropped."""
        _enable_memo(monkeypatch, True)
        state_with_plan["plan_steps"][0] = PlanStep(id="s1", description="Compute the sum")
        state_with_plan["step_outputs"] = {"unrelated_key": "prior_output"}

        gateway, tools = _llm_gateway_tools()
        result = await execute_node(state_with_plan, gateway=gateway, tools=tools)

        memo = result["step_outputs"]
        assert memo["unrelated_key"] == "prior_output"  # preserved
        assert "ran code" in memo.values()  # new entry recorded


# ── verify._stamp_verify_cycle clear path ──────────────────────────────────


class TestVerifyClear:
    def test_clears_on_gap_replan(self) -> None:
        """verify rejects (not complete) AND plan exhausted → memo cleared.
        This mirrors route_after_verify's "plan" (gap re-plan) branch."""
        state: dict[str, Any] = {
            "plan_steps": [PlanStep(id="s1", description="only step")],
            "current_step_index": 1,  # 1 >= len(1) → no remaining steps
            "errors": [],
            "verify_cycle": 0,
        }
        state_update: dict[str, Any] = {"is_complete": False}

        out = _stamp_verify_cycle(state_update, state)

        assert out["step_outputs"] == {}

    def test_preserves_when_steps_remain(self) -> None:
        """verify rejects BUT steps remain (mid-plan retry) → memo NOT cleared
        (the cache must survive for the tool_create/agent_spawn savings)."""
        state: dict[str, Any] = {
            "plan_steps": [
                PlanStep(id="s1", description="a"),
                PlanStep(id="s2", description="b"),
            ],
            "current_step_index": 0,  # 0 < 2 → remaining steps
            "errors": [],
            "verify_cycle": 0,
        }
        state_update: dict[str, Any] = {"is_complete": False}

        out = _stamp_verify_cycle(state_update, state)

        assert "step_outputs" not in out  # preserved (untouched)

    def test_preserves_when_complete(self) -> None:
        """verify passes (is_complete) → early return; memo never cleared."""
        state: dict[str, Any] = {
            "plan_steps": [PlanStep(id="s1", description="only step")],
            "current_step_index": 1,
            "errors": [],
            "verify_cycle": 0,
        }
        state_update: dict[str, Any] = {"is_complete": True}

        out = _stamp_verify_cycle(state_update, state)

        assert "step_outputs" not in out
        assert out["verify_cycle"] == 1
