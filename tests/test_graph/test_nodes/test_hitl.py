"""Tests for src.graph.nodes.hitl — HITL gate node function."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.graph.enums import Phase
from src.graph.nodes.hitl import hitl_gate_node


class TestHitlGateNode:
    """Tests for the hitl_gate_node async function."""

    @pytest.mark.asyncio
    async def test_hitl_auto_approves(self, sample_state: dict) -> None:
        """Interrupt unavailable -> auto-approve, phase=COMPLETE."""
        result = await hitl_gate_node(sample_state)

        assert result["phase"] == Phase.COMPLETE
        assert result["is_complete"] is True

    @pytest.mark.asyncio
    async def test_hitl_preserves_final_output(self, sample_state: dict) -> None:
        """final_output present -> output preserved in result."""
        sample_state["final_output"] = "Custom result output"
        sample_state["is_complete"] = True
        result = await hitl_gate_node(sample_state)

        assert result["final_output"] == "Custom result output"

    @pytest.mark.asyncio
    async def test_hitl_uses_goal_text_when_no_output(self, sample_state: dict) -> None:
        """No final_output -> uses goal text in output."""
        sample_state["final_output"] = ""
        result = await hitl_gate_node(sample_state)

        assert "Completed:" in result["final_output"]

    @pytest.mark.asyncio
    async def test_hitl_no_goal_auto_approves(self, sample_state: dict) -> None:
        """No current_goal -> no crash, auto-approve."""
        sample_state["current_goal"] = None
        result = await hitl_gate_node(sample_state)

        assert result["phase"] == Phase.COMPLETE
        assert result["is_complete"] is True

    @pytest.mark.asyncio
    async def test_hitl_always_returns_complete_phase(self, sample_state: dict) -> None:
        """In unit test context (no interrupt), always returns COMPLETE."""
        result = await hitl_gate_node(sample_state)
        assert result["phase"] == Phase.COMPLETE


class TestHitlReviewMessage:
    """The full review is recorded as a HumanMessage, not just flattened errors (Q100)."""

    @pytest.mark.asyncio
    async def test_auto_approve_records_review_message(self, sample_state: dict) -> None:
        """Auto-approve path appends a HumanMessage carrying the review context."""
        sample_state["final_output"] = "Done output"
        result = await hitl_gate_node(sample_state)

        messages = result.get("messages", [])
        assert len(messages) == 1
        content = messages[0].content
        assert "AUTO-APPROVED" in content
        assert "Done output" in content

    @pytest.mark.asyncio
    async def test_reject_records_review_message_with_feedback(self, sample_state: dict) -> None:
        """Rejected review is a HumanMessage (not only a flattened errors string)."""
        sample_state["final_output"] = "Partial"
        sample_state["is_complete"] = False

        with patch(
            "langgraph.types.interrupt",
            return_value={"approved": False, "feedback": "needs more detail"},
            create=True,
        ):
            result = await hitl_gate_node(sample_state)

        content = result["messages"][0].content
        assert "REJECTED" in content
        assert "needs more detail" in content
        # The flattened errors string still carries the short signal too.
        assert any("rejected" in str(e).lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_approve_records_review_message(self, sample_state: dict) -> None:
        """Approved review (via interrupt) is recorded as a HumanMessage."""
        sample_state["final_output"] = "Approved output"

        with patch(
            "langgraph.types.interrupt", return_value={"approved": True}, create=True
        ):
            result = await hitl_gate_node(sample_state)

        content = result["messages"][0].content
        assert "APPROVED" in content
        assert "Approved output" in content


class TestHitlGateNodeWithInterrupt:
    """Tests for hitl_gate_node with mocked LangGraph interrupt."""

    @pytest.mark.asyncio
    async def test_interrupt_human_approves(self, sample_state: dict) -> None:
        """interrupt() returns approved=True -> phase COMPLETE, is_complete True."""
        sample_state["final_output"] = "Task done"
        sample_state["is_complete"] = True

        with patch("langgraph.types.interrupt", return_value={"approved": True}, create=True):
            result = await hitl_gate_node(sample_state)

        assert result["phase"] == Phase.COMPLETE
        assert result["is_complete"] is True
        assert result["final_output"] == "Task done"

    @pytest.mark.asyncio
    async def test_interrupt_human_rejects_with_feedback(self, sample_state: dict) -> None:
        """interrupt() returns approved=False with feedback -> phase EXECUTE."""
        sample_state["final_output"] = "Partial result"
        sample_state["is_complete"] = False

        with patch("langgraph.types.interrupt", return_value={
            "approved": False,
            "feedback": "try again with more detail",
        }, create=True):
            result = await hitl_gate_node(sample_state)

        assert result["phase"] == Phase.EXECUTE
        assert result["is_complete"] is False
        assert len(result["errors"]) >= 1
        assert "try again with more detail" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_interrupt_human_rejects_without_feedback(self, sample_state: dict) -> None:
        """interrupt() returns approved=False, no feedback -> phase EXECUTE, generic error."""
        sample_state["final_output"] = "Something"
        sample_state["is_complete"] = False

        with patch("langgraph.types.interrupt", return_value={
            "approved": False,
        }, create=True):
            result = await hitl_gate_node(sample_state)

        assert result["phase"] == Phase.EXECUTE
        assert result["is_complete"] is False
        assert any("rejected" in str(e).lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_interrupt_human_approves_boolean_true(self, sample_state: dict) -> None:
        """interrupt() returns raw True (not a dict) -> treated as approved."""
        sample_state["final_output"] = "Done result"
        sample_state["is_complete"] = True

        with patch("langgraph.types.interrupt", return_value=True, create=True):
            result = await hitl_gate_node(sample_state)

        assert result["phase"] == Phase.COMPLETE
        assert result["is_complete"] is True

    @pytest.mark.asyncio
    async def test_interrupt_runtime_error_auto_approves(self, sample_state: dict) -> None:
        """interrupt() raises RuntimeError -> auto-approve fallback."""
        sample_state["final_output"] = "Cached result"
        sample_state["is_complete"] = True

        with patch("langgraph.types.interrupt", side_effect=RuntimeError("not in graph context"), create=True):
            result = await hitl_gate_node(sample_state)

        assert result["phase"] == Phase.COMPLETE
        assert result["is_complete"] is True
        assert result["final_output"] == "Cached result"

    @pytest.mark.asyncio
    async def test_interrupt_import_error_auto_approves(self, sample_state: dict) -> None:
        """interrupt raises ImportError -> auto-approve fallback."""
        sample_state["final_output"] = "Fallback output"
        sample_state["is_complete"] = True

        with patch("langgraph.types.interrupt", side_effect=ImportError("no module"), create=True):
            result = await hitl_gate_node(sample_state)

        assert result["phase"] == Phase.COMPLETE
        assert result["is_complete"] is True

    @pytest.mark.asyncio
    async def test_interrupt_no_goal_with_final_output(self, sample_state: dict) -> None:
        """No goal but final_output present -> output preserved through interrupt."""
        sample_state["current_goal"] = None
        sample_state["final_output"] = "Preserved output"
        sample_state["is_complete"] = True

        with patch("langgraph.types.interrupt", return_value={"approved": True}, create=True):
            result = await hitl_gate_node(sample_state)

        assert result["final_output"] == "Preserved output"
        assert result["phase"] == Phase.COMPLETE
