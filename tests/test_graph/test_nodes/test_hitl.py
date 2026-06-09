"""Tests for src.graph.nodes.hitl — HITL gate node function."""

from __future__ import annotations

import pytest

from src.graph.enums import Phase
from src.graph.nodes.hitl import hitl_gate_node


class TestHitlGateNode:
    """Tests for the hitl_gate_node async function."""

    @pytest.mark.asyncio
    async def test_hitl_auto_approves(self, sample_state: dict) -> None:
        """Interrupt unavailable → auto-approve, phase=COMPLETE."""
        result = await hitl_gate_node(sample_state)

        assert result["phase"] == Phase.COMPLETE
        assert result["is_complete"] is True

    @pytest.mark.asyncio
    async def test_hitl_preserves_final_output(self, sample_state: dict) -> None:
        """final_output present → output preserved in result."""
        sample_state["final_output"] = "Custom result output"
        sample_state["is_complete"] = True
        result = await hitl_gate_node(sample_state)

        assert result["final_output"] == "Custom result output"

    @pytest.mark.asyncio
    async def test_hitl_uses_goal_text_when_no_output(self, sample_state: dict) -> None:
        """No final_output → uses goal text in output."""
        sample_state["final_output"] = ""
        result = await hitl_gate_node(sample_state)

        assert "Completed:" in result["final_output"]

    @pytest.mark.asyncio
    async def test_hitl_no_goal_auto_approves(self, sample_state: dict) -> None:
        """No current_goal → no crash, auto-approve."""
        sample_state["current_goal"] = None
        result = await hitl_gate_node(sample_state)

        assert result["phase"] == Phase.COMPLETE
        assert result["is_complete"] is True

    @pytest.mark.asyncio
    async def test_hitl_always_returns_complete_phase(self, sample_state: dict) -> None:
        """In unit test context (no interrupt), always returns COMPLETE."""
        result = await hitl_gate_node(sample_state)
        assert result["phase"] == Phase.COMPLETE
