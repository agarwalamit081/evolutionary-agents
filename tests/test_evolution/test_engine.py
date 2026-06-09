"""Tests for src.evolution.engine — SelfEvolutionEngine 4-phase pipeline."""

from __future__ import annotations

import pytest

from src.evolution.engine import SelfEvolutionEngine
from src.graph.enums import MutationType


class TestEvolutionEngine:
    """Tests for the SelfEvolutionEngine class."""

    @pytest.mark.asyncio
    async def test_analyze_with_failure_patterns(self) -> None:
        """Failure patterns produce PROMPT opportunity with high priority."""
        engine = SelfEvolutionEngine()
        result = await engine.analyze(
            execution_history=[],
            failure_patterns=["timeout on tool call", "invalid JSON from LLM"],
        )

        opportunities = result["opportunities"]
        prompt_opps = [o for o in opportunities if o["type"] == MutationType.PROMPT]
        assert len(prompt_opps) == 1
        assert prompt_opps[0]["priority"] == "high"

    @pytest.mark.asyncio
    async def test_analyze_with_slow_execution(self) -> None:
        """Slow execution (>5000ms avg) produces WORKFLOW opportunity."""
        engine = SelfEvolutionEngine()
        result = await engine.analyze(
            execution_history=[
                {"duration_ms": 8000},
                {"duration_ms": 6000},
            ],
            failure_patterns=[],
        )

        opportunities = result["opportunities"]
        workflow_opps = [o for o in opportunities if o["type"] == MutationType.WORKFLOW]
        assert len(workflow_opps) == 1
        assert workflow_opps[0]["priority"] == "medium"

    @pytest.mark.asyncio
    async def test_analyze_always_includes_tool_optimization(self) -> None:
        """TOOL optimization opportunity is always present."""
        engine = SelfEvolutionEngine()
        result = await engine.analyze(execution_history=[], failure_patterns=[])

        types = [o["type"] for o in result["opportunities"]]
        assert MutationType.TOOL in types

    @pytest.mark.asyncio
    async def test_analyze_returns_performance_metrics(self) -> None:
        """Analysis result includes performance_metrics."""
        engine = SelfEvolutionEngine()
        result = await engine.analyze(
            execution_history=[{"duration_ms": 100}],
            failure_patterns=["error1"],
        )

        metrics = result["performance_metrics"]
        assert metrics["execution_count"] == 1
        assert metrics["failure_count"] == 1

    @pytest.mark.asyncio
    async def test_generate_returns_proposal(self) -> None:
        """Generate produces a proposal with mutation_type and content."""
        engine = SelfEvolutionEngine()
        opportunity = {"type": MutationType.PROMPT, "description": "Fix prompts", "priority": "high"}
        result = await engine.generate(opportunity, current_content="old prompt")

        assert result["mutation_type"] == MutationType.PROMPT
        assert result["original_content"] == "old prompt"
        assert result["mutated_content"] is not None

    @pytest.mark.asyncio
    async def test_validate_clean_code_passes(self) -> None:
        """Clean mutation proposal passes safety validation."""
        engine = SelfEvolutionEngine()
        proposal = {
            "mutation_type": MutationType.PROMPT,
            "description": "Improve prompt",
            "mutated_content": "# Simple improvement comment\npass",
        }
        result = await engine.validate(proposal)
        assert result["passed"] is True

    @pytest.mark.asyncio
    async def test_validate_dangerous_code_fails(self) -> None:
        """Dangerous code in proposal fails safety validation."""
        engine = SelfEvolutionEngine()
        proposal = {
            "mutation_type": MutationType.CODE,
            "description": "Dangerous mutation",
            "mutated_content": "import os\nos.system('rm -rf /')",
        }
        result = await engine.validate(proposal)
        assert result["passed"] is False

    @pytest.mark.asyncio
    async def test_deploy_rejected_if_validation_fails(self) -> None:
        """Deploy with failed validation returns deployed=False."""
        engine = SelfEvolutionEngine()
        proposal = {"mutation_type": MutationType.CODE, "description": "test"}
        validation = {"passed": False, "reason": "Safety check failed"}
        result = await engine.deploy(proposal, validation)

        assert result["deployed"] is False
