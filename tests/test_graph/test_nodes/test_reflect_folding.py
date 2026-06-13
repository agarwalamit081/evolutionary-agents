"""Tests for memory-folding integration in the reflect node (_check_and_fold).

Validates Bug A/B/C fixes:
- A: folding triggers on live gateway token usage
- B: existing messages are wrapped in RemoveMessage (history actually shrinks)
- C: the three structured summaries persist to warm memory via store_skill
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage, RemoveMessage

from src.graph.enums import GoalStatus, Phase
from src.graph.models import Goal
from src.graph.nodes.reflect import _check_and_fold


def _mock_response(content: str) -> MagicMock:
    """Build a mock LLMResponse with the given content."""
    resp = MagicMock()
    resp.content = content
    return resp


def _make_messages(count: int) -> list[HumanMessage]:
    """Build id'd HumanMessages (so RemoveMessage reduction targets them)."""
    return [HumanMessage(content=f"message {i}", id=f"m-{i}") for i in range(count)]


def _make_state(
    iteration_count: int = 10,
    messages: list[Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a minimal agent state for fold testing."""
    state: dict[str, Any] = {
        "current_goal": Goal(text="test goal", status=GoalStatus.ACTIVE),
        "iteration_count": iteration_count,
        "messages": messages or [],
        "fold_history": [],
        "tools_called": [],
        "tool_results": [],
        "completed_steps": [],
        "errors": [],
        "plan_steps": [],
    }
    state.update(kwargs)
    return state


def _make_gateway(trigger_tokens: int = 60_000) -> MagicMock:
    """Mock gateway: high live-token usage + 3 canned fold responses."""
    gateway = MagicMock()
    gateway.acompletion = AsyncMock(side_effect=[
        _mock_response(json.dumps({"task_description": "t", "key_events": []})),
        _mock_response(json.dumps({"immediate_goal": "g", "next_actions": []})),
        _mock_response(json.dumps({"tools_used": [], "derived_rules": []})),
    ])
    record = MagicMock()
    record.input_tokens = trigger_tokens
    record.output_tokens = 0
    gateway.get_cost_records = MagicMock(return_value=[record])
    return gateway


def _folding_cfg(**overrides: Any) -> dict[str, Any]:
    """Build a folding config dict mirroring AgentSettings defaults."""
    cfg: dict[str, Any] = {
        "enabled": True,
        "fold_interval": 10,
        "token_threshold": 50_000,
        "message_count_floor": 10,
        "message_count_threshold": 14,
        "message_token_estimate": 8_000,
        "max_folds": 3,
    }
    cfg.update(overrides)
    return cfg


class TestCheckAndFold:
    """Tests for the _check_and_fold helper in reflect_node."""

    @pytest.mark.asyncio
    async def test_returns_none_when_disabled(self) -> None:
        """Folding skipped entirely when enabled=False."""
        gateway = _make_gateway()
        state = _make_state(messages=_make_messages(10))
        result = await _check_and_fold(state, gateway, None, {"enabled": False})
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_trigger(self) -> None:
        """No fold when no trigger fires (low tokens, count, and context size)."""
        gateway = _make_gateway(trigger_tokens=0)
        state = _make_state(iteration_count=5, messages=_make_messages(12))
        result = await _check_and_fold(
            state, gateway, None,
            _folding_cfg(message_count_threshold=999, message_token_estimate=10**6),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_folds_and_reduces_messages(self) -> None:
        """On fold, existing messages become RemoveMessage + a summary appends."""
        gateway = _make_gateway()  # 60k tokens → live-token trigger
        msgs = _make_messages(10)
        state = _make_state(iteration_count=10, messages=msgs)
        result = await _check_and_fold(state, gateway, None, _folding_cfg())

        assert result is not None
        assert result["phase"] == Phase.REFLECT

        out_messages = result["messages"]
        # 10 RemoveMessage entries + 1 summary message
        assert len(out_messages) == len(msgs) + 1
        assert all(isinstance(m, RemoveMessage) for m in out_messages[:-1])
        assert isinstance(out_messages[-1], HumanMessage)
        assert "Memory Fold" in out_messages[-1].content

        assert len(result["fold_history"]) == 1
        assert result["last_fold_iteration"] == 10
        # Each fold record carries the iteration it fired at (surfaced for the
        # e2e report's fold details, replacing the old "Iteration 0: unknown").
        fold_record = result["fold_history"][0]
        assert fold_record["iteration"] == 10
        assert "fold_number" in fold_record

    @pytest.mark.asyncio
    async def test_persists_three_folded_summaries(self) -> None:
        """The episode/working/tool summaries persist as folded_memory skills."""
        gateway = _make_gateway()
        memory = MagicMock()
        memory.store_skill = AsyncMock(return_value="uuid")
        state = _make_state(iteration_count=10, messages=_make_messages(10))
        result = await _check_and_fold(state, gateway, memory, _folding_cfg())

        assert result is not None
        assert memory.store_skill.await_count == 3
        for call in memory.store_skill.await_args_list:
            kwargs = call.kwargs
            assert kwargs["skill_type"] == "folded_memory"
            assert "folded_memory" in kwargs["tags"]

    @pytest.mark.asyncio
    async def test_graceful_without_memory(self) -> None:
        """Folding still reduces context when memory=None (persistence skipped)."""
        gateway = _make_gateway()
        state = _make_state(iteration_count=10, messages=_make_messages(10))
        result = await _check_and_fold(state, gateway, None, _folding_cfg())

        assert result is not None
        assert result["phase"] == Phase.REFLECT
        assert isinstance(result["messages"][-1], HumanMessage)
