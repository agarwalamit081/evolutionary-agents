"""Tests for src.graph.nodes.evolve — evolve node function."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.graph.enums import Phase
from src.graph.factory import initial_state
from src.graph.models import ReflectionResult
from src.graph.nodes.evolve import evolve_node


class TestEvolveNode:
    """Tests for the evolve_node async function."""

    @pytest.mark.asyncio
    async def test_evolve_no_gateway_skips(self) -> None:
        """No gateway → phase=STORE_MEMORY, outcome=skipped_no_gateway."""
        state = initial_state("test goal", "thread-nogw")
        state["generation"] = 0
        result = await evolve_node(state)

        assert result["phase"] == Phase.STORE_MEMORY
        record = result["evolution_history"][0]
        assert record["outcome"] == "skipped_no_gateway"

    @pytest.mark.asyncio
    async def test_evolve_increments_generation(self) -> None:
        """Each call increments generation by 1."""
        state = initial_state("test goal", "thread-gen")
        state["generation"] = 5
        result = await evolve_node(state)

        assert result["generation"] == 6

    @pytest.mark.asyncio
    async def test_evolve_records_reflection_summary(self) -> None:
        """Reflection summary is recorded in evolution history."""
        state = initial_state("test goal", "thread-refl")
        state["generation"] = 0
        state["reflection"] = ReflectionResult(
            summary="Reflection summary text",
            lessons_learned=["lesson1"],
            confidence="high",
            should_evolve=True,
            should_replan=False,
            memory_observations=[],
            cost_efficiency=1.0,
        )
        result = await evolve_node(state)

        record = result["evolution_history"][0]
        assert record["summary"] == "Reflection summary text"
        assert record["lessons"] == ["lesson1"]

    @pytest.mark.asyncio
    async def test_evolve_with_gateway_falls_back_on_failure(self, mock_gateway: MagicMock) -> None:
        """Gateway present but engine fails → fallback path."""
        state = initial_state("test goal", "thread-gwfail")
        state["generation"] = 0
        result = await evolve_node(state, gateway=mock_gateway)

        # Engine will likely fail due to missing DB, falls back to skip
        assert result["phase"] == Phase.STORE_MEMORY
        assert result["generation"] == 1

    @pytest.mark.asyncio
    async def test_evolve_no_reflection_uses_default(self) -> None:
        """No reflection → summary defaults to 'no reflection'."""
        state = initial_state("test goal", "thread-norefl")
        state["generation"] = 0
        state["reflection"] = None
        result = await evolve_node(state)

        record = result["evolution_history"][0]
        assert record["summary"] == "no reflection"
        assert record["lessons"] == []

    @pytest.mark.asyncio
    async def test_evolve_always_returns_store_memory_phase(self) -> None:
        """All paths return phase=STORE_MEMORY."""
        state = initial_state("test goal", "thread-phase")
        state["generation"] = 0
        result = await evolve_node(state)

        assert result["phase"] == Phase.STORE_MEMORY
