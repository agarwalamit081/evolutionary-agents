"""Tests for src.memory.folding — autonomous memory compression."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.memory.folding import MemoryFolder, MemoryFoldResult, _serialize_messages, _serialize_tool_history


def _make_mock_gateway() -> MagicMock:
    """Create a mock LLMGateway with async acompletion."""
    gateway = MagicMock()
    gateway.acompletion = AsyncMock()
    return gateway


def _make_mock_response(content: str) -> MagicMock:
    """Create a mock LLMResponse."""
    resp = MagicMock()
    resp.content = content
    return resp


def _make_state(
    iteration_count: int = 5,
    total_tokens: int = 0,
    messages: list[Any] | None = None,
    fold_history: list[Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create a minimal agent state for testing."""
    from src.graph.enums import GoalStatus
    from src.graph.models import Goal

    if messages is None:
        messages = []
    if fold_history is None:
        fold_history = []

    state: dict[str, Any] = {
        "current_goal": Goal(text="test goal", status=GoalStatus.ACTIVE),
        "iteration_count": iteration_count,
        "total_tokens_used": total_tokens,
        "messages": messages,
        "fold_history": fold_history,
        "tools_called": [],
        "tool_results": [],
        "completed_steps": [],
        "errors": [],
        "plan_steps": [],
    }
    state.update(kwargs)
    return state


def _make_messages(count: int) -> list[Any]:
    """Create simple mock messages."""
    msgs = []
    for i in range(count):
        msg = MagicMock()
        msg.type = "user" if i % 2 == 0 else "assistant"
        msg.content = f"Message {i}: " + "x" * 200
        msgs.append(msg)
    return msgs


class TestShouldFold:
    """Tests for should_fold trigger conditions."""

    def test_no_fold_when_few_iterations(self) -> None:
        """Should not fold when iteration_count < 2."""
        gateway = _make_mock_gateway()
        folder = MemoryFolder(gateway)
        state = _make_state(iteration_count=1, messages=_make_messages(15))
        assert folder.should_fold(state) is False

    def test_no_fold_when_few_messages(self) -> None:
        """Should not fold when messages < 10."""
        gateway = _make_mock_gateway()
        folder = MemoryFolder(gateway)
        state = _make_state(iteration_count=5, messages=_make_messages(5))
        assert folder.should_fold(state) is False

    def test_no_fold_when_max_folds_reached(self) -> None:
        """Should not fold when fold_history exceeds max_folds."""
        gateway = _make_mock_gateway()
        folder = MemoryFolder(gateway, max_folds=2)
        state = _make_state(
            iteration_count=20,
            messages=_make_messages(15),
            fold_history=[{"fold": 1}, {"fold": 2}],
        )
        assert folder.should_fold(state) is False

    def test_fold_on_interval(self) -> None:
        """Should fold when iteration_count hits the interval."""
        gateway = _make_mock_gateway()
        folder = MemoryFolder(gateway, fold_interval=10)
        state = _make_state(iteration_count=10, messages=_make_messages(15))
        assert folder.should_fold(state) is True

    def test_fold_on_token_threshold(self) -> None:
        """Should fold when total_tokens_used exceeds threshold."""
        gateway = _make_mock_gateway()
        folder = MemoryFolder(gateway, token_threshold=1000)
        state = _make_state(iteration_count=5, total_tokens=5000, messages=_make_messages(15))
        assert folder.should_fold(state) is True

    def test_fold_on_message_length(self) -> None:
        """Should fold when estimated message tokens exceed threshold."""
        gateway = _make_mock_gateway()
        folder = MemoryFolder(gateway, message_token_estimate=100)
        # 20 messages * 200 chars each = 4000 chars / 4 = 1000 tokens > 100
        state = _make_state(iteration_count=5, messages=_make_messages(20))
        assert folder.should_fold(state) is True

    def test_no_fold_when_below_thresholds(self) -> None:
        """Should not fold when all thresholds are unmet."""
        gateway = _make_mock_gateway()
        folder = MemoryFolder(gateway, fold_interval=100, token_threshold=100000)
        state = _make_state(iteration_count=5, messages=_make_messages(12))
        assert folder.should_fold(state) is False


class TestMemoryFoldResult:
    """Tests for MemoryFoldResult serialization."""

    def test_to_dict_round_trip(self) -> None:
        """to_dict preserves all fields."""
        result = MemoryFoldResult(
            episode_memory={"task_description": "test"},
            working_memory={"immediate_goal": "do stuff"},
            tool_memory={"tools_used": []},
            fold_number=1,
            tokens_saved_estimate=5000,
        )
        d = result.to_dict()
        assert d["episode_memory"]["task_description"] == "test"
        assert d["fold_number"] == 1
        assert d["tokens_saved_estimate"] == 5000
        assert "folded_at" in d


class TestSerializeMessages:
    """Tests for _serialize_messages helper."""

    def test_serializes_message_types(self) -> None:
        """Serializes messages with their type and content."""
        msg = MagicMock()
        msg.type = "user"
        msg.content = "hello"
        result = _serialize_messages([msg])
        assert "[user] hello" in result

    def test_truncates_long_content(self) -> None:
        """Long content is truncated to 500 chars."""
        msg = MagicMock()
        msg.type = "assistant"
        msg.content = "x" * 1000
        result = _serialize_messages([msg])
        assert len(result.split("] ", 1)[1]) <= 500


class TestSerializeToolHistory:
    """Tests for _serialize_tool_history helper."""

    def test_empty_history(self) -> None:
        """Empty state returns 'No tool calls recorded.'"""
        result = _serialize_tool_history({})
        assert "No tool calls recorded" in result

    def test_serializes_tool_results(self) -> None:
        """ToolResult objects are serialized."""
        from src.graph.models import ToolResult

        tr = ToolResult(tool_name="web_search", success=True, output="found stuff")
        result = _serialize_tool_history({"tool_results": [tr]})
        assert "web_search" in result
        assert "success" in result


class TestFold:
    """Tests for the fold() method."""

    @pytest.mark.asyncio
    async def test_fold_returns_result(self) -> None:
        """fold() returns a MemoryFoldResult with 3 memory types."""
        gateway = _make_mock_gateway()
        # Mock 3 LLM calls (episode, working, tool)
        gateway.acompletion.side_effect = [
            _make_mock_response(json.dumps({"task_description": "test", "key_events": [], "current_progress": "ok"})),
            _make_mock_response(json.dumps({"immediate_goal": "goal", "current_challenges": "none", "next_actions": []})),
            _make_mock_response(json.dumps({"tools_used": [], "derived_rules": []})),
        ]

        folder = MemoryFolder(gateway)
        state = _make_state(messages=_make_messages(12))
        result = await folder.fold(state)

        assert isinstance(result, MemoryFoldResult)
        assert result.fold_number == 1
        assert "task_description" in result.episode_memory
        assert "immediate_goal" in result.working_memory
        assert "tools_used" in result.tool_memory

    @pytest.mark.asyncio
    async def test_fold_handles_llm_failure_gracefully(self) -> None:
        """fold() handles LLM errors and returns fallback dicts."""
        gateway = _make_mock_gateway()
        gateway.acompletion.side_effect = Exception("API error")

        folder = MemoryFolder(gateway)
        state = _make_state(messages=_make_messages(12))
        result = await folder.fold(state)

        assert isinstance(result, MemoryFoldResult)
        assert "error" in result.episode_memory
        assert "error" in result.working_memory
        assert "error" in result.tool_memory


class TestBuildSummaryMessage:
    """Tests for build_summary_message."""

    def test_creates_human_message(self) -> None:
        """build_summary_message returns a HumanMessage."""
        from langchain_core.messages import HumanMessage

        gateway = _make_mock_gateway()
        folder = MemoryFolder(gateway)
        result = MemoryFoldResult(
            episode_memory={"task_description": "test"},
            working_memory={"immediate_goal": "goal"},
            tool_memory={"tools_used": []},
            fold_number=1,
        )
        msg = folder.build_summary_message(result)
        assert isinstance(msg, HumanMessage)
        assert "Memory Fold #1" in msg.content
        assert "Episode Memory" in msg.content
        assert "Working Memory" in msg.content
        assert "Tool Memory" in msg.content
