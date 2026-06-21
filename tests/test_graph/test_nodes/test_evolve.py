"""Tests for src.graph.nodes.evolve — evolve node function."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.graph.enums import Confidence, Phase
from src.graph.factory import initial_state
from src.graph.models import ReflectionResult
from src.graph.nodes.evolve import evolve_node


class TestEvolveNode:
    """Tests for the evolve_node async function."""

    @pytest.mark.asyncio
    async def test_evolve_no_gateway_skips(self) -> None:
        """No gateway -> phase=STORE_MEMORY, outcome=skipped_no_gateway."""
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
    async def test_evolve_no_reflection_uses_default(self) -> None:
        """No reflection -> summary defaults to 'no reflection'."""
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

    @pytest.mark.asyncio
    async def test_evolve_no_gateway_records_skip(self) -> None:
        """With no gateway, records 'skipped_no_gateway' evolution outcome."""
        state = initial_state("test goal", "thread-skip")
        state["generation"] = 2
        state["reflection"] = ReflectionResult(
            summary="good run",
            lessons_learned=["tried fast path"],
            confidence=Confidence.HIGH,
            should_evolve=True,
            should_replan=False,
            memory_observations=[],
            cost_efficiency=1.0,
        )
        result = await evolve_node(state)

        record = result["evolution_history"][0]
        assert record["outcome"] == "skipped_no_gateway"
        assert record["generation"] == 2
        assert record["summary"] == "good run"
        assert record["lessons"] == ["tried fast path"]

    @pytest.mark.asyncio
    async def test_evolve_with_mock_gateway(self) -> None:
        """Mock gateway + patched SelfEvolutionEngine verifies engine is called correctly."""
        mock_gateway = MagicMock()

        cycle_result = {
            "status": "deployed",
            "deployed": True,
            "mutations_proposed": 1,
            "mutations_deployed": 1,
            "proposal": {"rationale": "improve speed"},
            "deployment": {"commit_hash": "abc123def456"},
        }

        mock_engine_instance = MagicMock()
        mock_engine_instance.run_cycle = AsyncMock(return_value=cycle_result)

        with patch("src.evolution.engine.SelfEvolutionEngine", return_value=mock_engine_instance), \
             patch("src.safety.pipeline.SafetyPipeline"), \
             patch("src.sandbox.executor.SandboxExecutor", side_effect=Exception("no sandbox")), \
             patch("src.evolution.git_tracker.GitTracker", side_effect=Exception("no git")), \
             patch("src.config.get_settings") as mock_settings:

            mock_settings.return_value = MagicMock()

            state = initial_state("test goal", "thread-mockgw")
            state["generation"] = 3
            state["execution_history"] = [{"tool": "search", "duration_ms": 100}]
            state["reflection"] = ReflectionResult(
                summary="decent performance",
                lessons_learned=["optimize search"],
                confidence=Confidence.MEDIUM,
                should_evolve=True,
                should_replan=False,
                memory_observations=[],
                cost_efficiency=0.9,
            )

            result = await evolve_node(state, gateway=mock_gateway)

            assert result["phase"] == Phase.STORE_MEMORY
            assert result["generation"] == 4
            # Verify engine.run_cycle was called with the correct params
            mock_engine_instance.run_cycle.assert_awaited_once()
            call_kwargs = mock_engine_instance.run_cycle.call_args
            assert call_kwargs.kwargs["execution_history"] == state["execution_history"]
            assert call_kwargs.kwargs["reflection"] == state["reflection"]
            assert call_kwargs.kwargs["sandbox"] is None
            assert call_kwargs.kwargs["git_tracker"] is None

    @pytest.mark.asyncio
    async def test_evolve_records_commit_hash(self) -> None:
        """Mock engine returns a cycle result with commit_hash, verify it's in evolution_history."""
        mock_gateway = MagicMock()

        cycle_result = {
            "status": "deployed",
            "deployed": True,
            "mutations_proposed": 1,
            "mutations_deployed": 1,
            "proposal": {"rationale": "speed up classify"},
            "deployment": {"commit_hash": "deadbeef1234"},
        }

        mock_engine_instance = MagicMock()
        mock_engine_instance.run_cycle = AsyncMock(return_value=cycle_result)

        with patch("src.evolution.engine.SelfEvolutionEngine", return_value=mock_engine_instance), \
             patch("src.safety.pipeline.SafetyPipeline"), \
             patch("src.sandbox.executor.SandboxExecutor", side_effect=Exception("no sandbox")), \
             patch("src.evolution.git_tracker.GitTracker", side_effect=Exception("no git")), \
             patch("src.config.get_settings") as mock_settings:

            mock_settings.return_value = MagicMock()

            state = initial_state("test goal", "thread-commit")
            state["generation"] = 1
            state["execution_history"] = []

            result = await evolve_node(state, gateway=mock_gateway)

            record = result["evolution_history"][0]
            assert record["commit_hash"] == "deadbeef1234"
            assert record["rationale"] == "speed up classify"

    @pytest.mark.asyncio
    async def test_evolve_records_outcome(self) -> None:
        """Outcome from cycle result is recorded in evolution_history."""
        mock_gateway = MagicMock()

        cycle_result = {
            "status": "validation_failed",
            "deployed": False,
            "mutations_proposed": 1,
            "mutations_deployed": 0,
            "proposal": {"rationale": "try tool optimization"},
            "deployment": {},
        }

        mock_engine_instance = MagicMock()
        mock_engine_instance.run_cycle = AsyncMock(return_value=cycle_result)

        with patch("src.evolution.engine.SelfEvolutionEngine", return_value=mock_engine_instance), \
             patch("src.safety.pipeline.SafetyPipeline"), \
             patch("src.sandbox.executor.SandboxExecutor", side_effect=Exception("no sandbox")), \
             patch("src.evolution.git_tracker.GitTracker", side_effect=Exception("no git")), \
             patch("src.config.get_settings") as mock_settings:

            mock_settings.return_value = MagicMock()

            state = initial_state("test goal", "thread-outcome")
            state["generation"] = 4
            state["execution_history"] = []

            result = await evolve_node(state, gateway=mock_gateway)

            record = result["evolution_history"][0]
            assert record["outcome"] == "validation_failed"
            assert record["mutations_proposed"] == 1
            assert record["mutations_deployed"] == 0

    @pytest.mark.asyncio
    async def test_evolve_engine_failure_falls_back(self) -> None:
        """When engine raises exception, falls back to skip record."""
        mock_gateway = MagicMock()

        mock_engine_instance = MagicMock()
        mock_engine_instance.run_cycle = AsyncMock(side_effect=RuntimeError("database connection failed"))

        with patch("src.evolution.engine.SelfEvolutionEngine", return_value=mock_engine_instance), \
             patch("src.safety.pipeline.SafetyPipeline"), \
             patch("src.sandbox.executor.SandboxExecutor", side_effect=Exception("no sandbox")), \
             patch("src.evolution.git_tracker.GitTracker", side_effect=Exception("no git")), \
             patch("src.config.get_settings") as mock_settings:

            mock_settings.return_value = MagicMock()

            state = initial_state("test goal", "thread-enginefail")
            state["generation"] = 7
            state["execution_history"] = []

            result = await evolve_node(state, gateway=mock_gateway)

            # Falls back to the no-gateway path when engine throws
            assert result["phase"] == Phase.STORE_MEMORY
            record = result["evolution_history"][0]
            assert record["outcome"] == "skipped_no_gateway"
            assert result["generation"] == 8

    @pytest.mark.asyncio
    async def test_evolve_records_mutations_counts_from_cycle(self) -> None:
        """Mutations proposed and deployed counts are recorded from cycle result."""
        mock_gateway = MagicMock()

        cycle_result = {
            "status": "deployed",
            "deployed": True,
            "mutations_proposed": 3,
            "mutations_deployed": 2,
            "proposal": {"rationale": "batch optimization"},
            "deployment": {"commit_hash": "beefcafe5678"},
        }

        mock_engine_instance = MagicMock()
        mock_engine_instance.run_cycle = AsyncMock(return_value=cycle_result)

        with patch("src.evolution.engine.SelfEvolutionEngine", return_value=mock_engine_instance), \
             patch("src.safety.pipeline.SafetyPipeline"), \
             patch("src.sandbox.executor.SandboxExecutor", side_effect=Exception("no sandbox")), \
             patch("src.evolution.git_tracker.GitTracker", side_effect=Exception("no git")), \
             patch("src.config.get_settings") as mock_settings:

            mock_settings.return_value = MagicMock()

            state = initial_state("test goal", "thread-counts")
            state["generation"] = 0
            state["execution_history"] = []

            result = await evolve_node(state, gateway=mock_gateway)

            record = result["evolution_history"][0]
            assert record["mutations_proposed"] == 3
            assert record["mutations_deployed"] == 2
            assert record["commit_hash"] == "beefcafe5678"
