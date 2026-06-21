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

# Direct ``should_fold`` tests construct a MemoryFolder against a fake gateway.
# The live-token trigger reads the gateway's per-run accumulator via
# ``get_cost_records()`` (the exact accessor these tests pin). Importing here
# keeps the folding-node imports (above) cleanly separated from the unit imports.
from src.memory.folding import MemoryFolder


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


def _token_gateway(total_tokens: int) -> MagicMock:
    """Build a fake gateway whose get_cost_records() returns one record.

    The live-token branch of ``should_fold`` reads the gateway's per-run
    accumulator via ``get_cost_records()`` (not state), summing each record's
    ``input_tokens + output_tokens``. This helper fakes that accessor.
    """
    gateway = MagicMock()
    record = MagicMock()
    record.input_tokens = total_tokens
    record.output_tokens = 0
    gateway.get_cost_records = MagicMock(return_value=[record])
    return gateway


class TestShouldFoldBranches:
    """Pin each of the SIX trigger branches in MemoryFolder.should_fold.

    The trigger ladder is evaluated in order, first match wins:
      1. Cap            (fold_history >= max_folds)         → False
      2. Min guard      (iteration_count < 2 OR msgs < floor) → False
      3. Cooldown       (iteration - last_fold < fold_interval) → False
      4. Live-token     (sum(get_cost_records() tokens) >= threshold) → True
      5. Message-count  (len(messages) >= threshold)         → True
      6. Context-size   (chars // 4 >= message_token_estimate) → True

    Each test constructs state to isolate EXACTLY one branch by disabling all
    earlier triggers (cap not hit, min guard satisfied, cooldown elapsed) and,
    for the False-branches, also disabling the later True-triggers.
    """

    def test_should_fold_false_at_cap(self) -> None:
        """Branch 1: max_folds already reached → False (short-circuits first).

        State passes the min guard, cooldown, AND the live-token/message
        triggers, yet still returns False because the cap is checked first.
        """
        folder = MemoryFolder(
            _token_gateway(total_tokens=999_999),  # would trigger branch 4
            max_folds=2,
            token_threshold=50_000,
            message_count_floor=10,
            message_count_threshold=14,
        )
        state = _make_state(
            iteration_count=20,
            messages=_make_messages(20),  # would trigger branch 5 too
            fold_history=[{"fold_number": 1}, {"fold_number": 2}],  # == max_folds
            last_fold_iteration=0,  # no cooldown
        )
        assert folder.should_fold(state) is False

    def test_should_fold_false_min_guard_low_iteration(self) -> None:
        """Branch 2a: iteration_count < 2 → False (too early).

        Cap is NOT hit; the live-token trigger WOULD fire, but the min guard
        precedes it.
        """
        folder = MemoryFolder(
            _token_gateway(total_tokens=999_999),
            max_folds=3,
            token_threshold=50_000,
            message_count_floor=10,
        )
        state = _make_state(
            iteration_count=1,  # < 2 → min guard
            messages=_make_messages(20),  # >= floor
        )
        assert folder.should_fold(state) is False

    def test_should_fold_false_min_guard_few_messages(self) -> None:
        """Branch 2b: len(messages) < message_count_floor → False (too few).

        Cap NOT hit, iteration >= 2, live-token WOULD fire, but too few
        messages trips the min guard first.
        """
        folder = MemoryFolder(
            _token_gateway(total_tokens=999_999),
            max_folds=3,
            token_threshold=50_000,
            message_count_floor=10,
        )
        state = _make_state(
            iteration_count=5,
            messages=_make_messages(5),  # < floor (10)
        )
        assert folder.should_fold(state) is False

    def test_should_fold_false_cooldown(self) -> None:
        """Branch 3: last fold too recent (within fold_interval) → False.

        Cap NOT hit, min guard satisfied, live-token WOULD fire, but the
        cooldown check (iteration - last_fold < fold_interval) rejects it.
        """
        folder = MemoryFolder(
            _token_gateway(total_tokens=999_999),
            max_folds=3,
            fold_interval=6,
            token_threshold=50_000,
            message_count_floor=10,
        )
        state = _make_state(
            iteration_count=10,
            messages=_make_messages(20),
            fold_history=[{"fold_number": 1}],
            last_fold_iteration=8,  # 10 - 8 = 2 < 6 → cooldown active
        )
        assert folder.should_fold(state) is False

    def test_should_fold_true_live_token(self) -> None:
        """Branch 4: gateway token usage >= token_threshold → True.

        Cap NOT hit, min guard satisfied, cooldown elapsed, message count and
        context size BELOW their thresholds — so ONLY the live-token trigger
        fires. Proves the accessor is get_cost_records().
        """
        folder = MemoryFolder(
            _token_gateway(total_tokens=60_000),  # >= 50_000 threshold
            max_folds=3,
            fold_interval=6,
            token_threshold=50_000,
            message_count_floor=10,
            message_count_threshold=999,   # disable branch 5
            message_token_estimate=10**6,  # disable branch 6
        )
        state = _make_state(
            iteration_count=10,
            messages=_make_messages(12),  # >= floor, < threshold
            last_fold_iteration=0,        # no cooldown
        )
        assert folder.should_fold(state) is True

    def test_should_fold_true_message_count(self) -> None:
        """Branch 5: len(messages) >= message_count_threshold → True.

        Live-token BELOW threshold (so branch 4 does not fire); context size
        BELOW threshold (branch 6 disabled) so ONLY message-count triggers.
        """
        folder = MemoryFolder(
            _token_gateway(total_tokens=0),  # disable branch 4
            max_folds=3,
            fold_interval=6,
            token_threshold=50_000,
            message_count_floor=10,
            message_count_threshold=14,
            message_token_estimate=10**6,  # disable branch 6
        )
        state = _make_state(
            iteration_count=10,
            messages=_make_messages(14),  # == threshold
            last_fold_iteration=0,
        )
        assert folder.should_fold(state) is True

    def test_should_fold_true_context_size(self) -> None:
        """Branch 6: estimated tokens (chars // 4) >= estimate → True.

        Live-token BELOW threshold, message count BELOW threshold, so ONLY
        the context-size trigger fires. Uses long single messages so the char
        count (//4) crosses the estimate while the message count stays low.
        """
        # One message of 40_000 chars → 10_000 est tokens >= 8_000 estimate.
        big_msg = HumanMessage(content="x" * 40_000, id="big-0")
        # Pad up to >= message_count_floor (10) with tiny messages so the min
        # guard passes, while staying below message_count_threshold (14).
        msgs = [big_msg] + [
            HumanMessage(content="y", id=f"pad-{i}") for i in range(11)
        ]
        folder = MemoryFolder(
            _token_gateway(total_tokens=0),  # disable branch 4
            max_folds=3,
            fold_interval=6,
            token_threshold=50_000,
            message_count_floor=10,
            message_count_threshold=999,    # disable branch 5
            message_token_estimate=8_000,
        )
        state = _make_state(
            iteration_count=10,
            messages=msgs,
            last_fold_iteration=0,
        )
        assert folder.should_fold(state) is True
