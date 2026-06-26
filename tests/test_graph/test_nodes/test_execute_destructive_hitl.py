"""F3 — destructive-tool HITL gate in _execute_tool_call (opt-in knob).

The gate sits at the tool-invocation chokepoint: when
``DESTRUCTIVE_TOOL_HITL_ENABLED`` is on and the tool is flagged
``destructiveHint``, the call routes through a LangGraph ``interrupt()``. These
tests pin the three observable behaviors without spinning up a full graph:

* knob off (default) -> gate skipped, handler runs;
* knob on + destructive, called OUTSIDE a graph -> ``interrupt()`` raises
  ``RuntimeError`` -> blocked ToolResult, handler never invoked;
* knob on + destructive + human approves (``interrupt`` stubbed) -> handler runs.

The out-of-graph ``RuntimeError`` is the real langgraph behavior (verified:
``interrupt()`` with no runnable context raises ``RuntimeError``), so these
direct calls exercise the same safe-block fallback a headless worker hits.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.graph.nodes.execute import _execute_tool_call
from src.tools.registry import ToolRegistry


def _knob_settings(enabled: bool) -> MagicMock:
    """Stand-in Settings exposing only the knob _execute_tool_call reads."""
    settings = MagicMock()
    settings.agent.destructive_tool_hitl_enabled = enabled
    return settings


def _registry(handler: AsyncMock) -> ToolRegistry:
    """Registry with one destructive + one safe tool sharing a handler mock."""
    registry = ToolRegistry()
    registry.register("destructive_tool", handler, mcp_hints={"destructiveHint": True})
    registry.register("safe_tool", handler)
    return registry


class TestDestructiveToolHitlGate:
    """F3 — the execute-node HITL gate for destructive tools."""

    @pytest.mark.asyncio
    async def test_knob_off_destructive_tool_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Knob off (default) -> the gate is skipped and the handler runs."""
        handler = AsyncMock(return_value="ok")
        registry = _registry(handler)
        monkeypatch.setattr(
            "src.graph.nodes.execute.get_settings", lambda: _knob_settings(False)
        )
        tc = {"function": {"name": "destructive_tool", "arguments": "{}"}}
        result = await _execute_tool_call(tc, registry)
        assert result.success is True
        handler.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_knob_on_destructive_blocks_without_handler(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Knob on + destructive, called outside a graph -> blocked; handler
        NEVER invoked (the safe default for an irreversible op)."""
        handler = AsyncMock(return_value="ok")
        registry = _registry(handler)
        monkeypatch.setattr(
            "src.graph.nodes.execute.get_settings", lambda: _knob_settings(True)
        )
        tc = {"function": {"name": "destructive_tool", "arguments": "{}"}}
        result = await _execute_tool_call(tc, registry)
        assert result.success is False
        assert result.error is not None
        assert "blocked" in result.error.lower()
        handler.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_knob_on_safe_tool_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Knob on but the tool is NOT destructive -> gate skipped, handler runs."""
        handler = AsyncMock(return_value="ok")
        registry = _registry(handler)
        monkeypatch.setattr(
            "src.graph.nodes.execute.get_settings", lambda: _knob_settings(True)
        )
        tc = {"function": {"name": "safe_tool", "arguments": "{}"}}
        result = await _execute_tool_call(tc, registry)
        assert result.success is True
        handler.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_knob_on_destructive_approved_runs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Knob on + destructive + human approves the interrupt -> handler runs."""
        handler = AsyncMock(return_value="ok")
        registry = _registry(handler)
        monkeypatch.setattr(
            "src.graph.nodes.execute.get_settings", lambda: _knob_settings(True)
        )
        monkeypatch.setattr(
            "langgraph.types.interrupt", lambda _payload: {"approved": True}
        )
        tc = {"function": {"name": "destructive_tool", "arguments": "{}"}}
        result = await _execute_tool_call(tc, registry)
        assert result.success is True
        handler.assert_awaited_once()
