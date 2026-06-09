"""Tests for src.evolution.engine — SelfEvolutionEngine 4-phase pipeline."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.evolution.engine import SelfEvolutionEngine
from src.graph.enums import MutationType
from src.sandbox.executor import SandboxResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_gateway(response_content: str | None = None) -> MagicMock:
    """Create a mock LLMGateway with a configurable acompletion response."""
    mock_gateway = MagicMock()
    mock_response = MagicMock()
    mock_response.content = response_content or json.dumps({
        "mutation_type": "code",
        "target_path": "test.py",
        "mutated_content": "print(1)",
        "description": "test mutation",
        "rationale": "testing",
    })
    mock_response.model = "test-model"
    mock_response.total_tokens = 100
    mock_gateway.acompletion = AsyncMock(return_value=mock_response)
    return mock_gateway


def _make_sandbox_result(
    success: bool = True,
    exit_code: int = 0,
    stdout: str = "ok",
    stderr: str = "",
    duration_seconds: float = 0.1,
    timed_out: bool = False,
) -> SandboxResult:
    """Create a SandboxResult with sensible defaults."""
    return SandboxResult(
        success=success,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=duration_seconds,
        memory_mb=None,
        timed_out=timed_out,
    )


def _make_mock_sandbox(
    success: bool = True,
    exit_code: int = 0,
    duration_seconds: float = 0.1,
    timed_out: bool = False,
) -> MagicMock:
    """Create a mock SandboxExecutor."""
    mock_sandbox = MagicMock()
    result = _make_sandbox_result(
        success=success,
        exit_code=exit_code,
        duration_seconds=duration_seconds,
        timed_out=timed_out,
    )
    mock_sandbox.execute_code = AsyncMock(return_value=result)
    return mock_sandbox


def _make_mock_git_tracker() -> MagicMock:
    """Create a mock GitTracker."""
    mock_tracker = MagicMock()
    mock_tracker.apply_mutation = AsyncMock(return_value=None)
    mock_tracker.snapshot = AsyncMock(return_value="abcdef1234567890")
    return mock_tracker


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestEvolutionEngine:
    """Tests for the SelfEvolutionEngine class."""

    # ── Constructor tests ────────────────────────────────────────────

    def test_constructor_with_gateway(self) -> None:
        """Engine accepts gateway parameter and stores it."""
        mock_gateway = _make_mock_gateway()
        engine = SelfEvolutionEngine(gateway=mock_gateway)
        assert engine._gateway is mock_gateway

    def test_constructor_without_gateway(self) -> None:
        """Engine works without gateway (uses heuristic fallback)."""
        engine = SelfEvolutionEngine()
        assert engine._gateway is None

    def test_constructor_with_safety_pipeline(self) -> None:
        """Engine accepts safety_pipeline parameter."""
        mock_safety = MagicMock()
        engine = SelfEvolutionEngine(safety_pipeline=mock_safety)
        assert engine._safety is mock_safety

    def test_constructor_default_safety_pipeline(self) -> None:
        """Engine creates default SafetyPipeline when none provided."""
        from src.safety.pipeline import SafetyPipeline
        engine = SelfEvolutionEngine()
        assert isinstance(engine._safety, SafetyPipeline)

    # ── Analyze tests ────────────────────────────────────────────────

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

    # ── Generate tests ───────────────────────────────────────────────

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
    async def test_generate_with_no_gateway_uses_heuristic(self) -> None:
        """Generate without gateway uses heuristic fallback."""
        engine = SelfEvolutionEngine()
        opportunity = {"type": MutationType.CODE, "description": "Optimize", "priority": "low"}
        result = await engine.generate(opportunity, current_content="x = 1")

        assert result["mutation_type"] == MutationType.CODE
        assert result["rationale"] == "Heuristic generation (no LLM available)"
        assert result["model_used"] is None
        assert result["tokens_used"] == 0

    @pytest.mark.asyncio
    async def test_generate_with_gateway_uses_llm(self) -> None:
        """Generate with gateway calls LLM and returns structured proposal."""
        mock_gateway = _make_mock_gateway()
        engine = SelfEvolutionEngine(gateway=mock_gateway)
        opportunity = {
            "type": MutationType.CODE,
            "description": "Optimize",
            "priority": "high",
            "patterns": ["slow execution"],
        }
        result = await engine.generate(opportunity, current_content="old code")

        mock_gateway.acompletion.assert_awaited_once()
        assert result["mutation_type"] == MutationType.CODE
        assert result["mutated_content"] == "print(1)"
        assert result["target_path"] == "test.py"
        assert result["model_used"] == "test-model"
        assert result["tokens_used"] == 100

    @pytest.mark.asyncio
    async def test_generate_with_gateway_failure_falls_back_to_heuristic(self) -> None:
        """Generate falls back to heuristic when LLM call fails."""
        mock_gateway = MagicMock()
        mock_gateway.acompletion = AsyncMock(side_effect=Exception("LLM provider error"))
        engine = SelfEvolutionEngine(gateway=mock_gateway)

        opportunity = {
            "type": MutationType.PROMPT,
            "description": "Fix prompts",
            "priority": "medium",
        }
        result = await engine.generate(opportunity, current_content="old")

        # Should still produce a result via heuristic fallback
        assert result["mutation_type"] == MutationType.PROMPT
        assert result["rationale"] == "Heuristic generation (no LLM available)"
        assert result["mutated_content"] is not None

    # ── Validate tests ───────────────────────────────────────────────

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

    # ── Sandbox test tests ───────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_sandbox_test_with_no_sandbox(self) -> None:
        """Sandbox test without sandbox returns passed=True with note."""
        engine = SelfEvolutionEngine()
        proposal = {"mutated_content": "x = 1"}
        result = await engine.sandbox_test(proposal, sandbox=None)

        assert result["passed"] is True
        assert result["note"] == "sandbox not available"

    @pytest.mark.asyncio
    async def test_sandbox_test_with_mock_sandbox_success(self) -> None:
        """Sandbox test with mock sandbox returning success passes."""
        mock_sandbox = _make_mock_sandbox(success=True, duration_seconds=0.1)
        engine = SelfEvolutionEngine()
        proposal = {"mutated_content": "x = 1"}
        result = await engine.sandbox_test(proposal, sandbox=mock_sandbox)

        assert result["passed"] is True
        assert result["sandbox_result"]["success"] is True
        assert result["sandbox_result"]["exit_code"] == 0
        mock_sandbox.execute_code.assert_awaited_once_with("x = 1")

    @pytest.mark.asyncio
    async def test_sandbox_test_with_failing_sandbox(self) -> None:
        """Sandbox test with failing sandbox returns passed=False."""
        mock_sandbox = _make_mock_sandbox(success=False, exit_code=1, duration_seconds=0.05)
        engine = SelfEvolutionEngine()
        proposal = {"mutated_content": "raise RuntimeError('boom')"}
        result = await engine.sandbox_test(proposal, sandbox=mock_sandbox)

        assert result["passed"] is False
        assert result["sandbox_result"]["success"] is False
        assert result["sandbox_result"]["exit_code"] == 1

    @pytest.mark.asyncio
    async def test_sandbox_test_with_timed_out_sandbox(self) -> None:
        """Sandbox test with timed out execution returns passed=False."""
        mock_sandbox = _make_mock_sandbox(success=False, timed_out=True, duration_seconds=30.0)
        engine = SelfEvolutionEngine()
        proposal = {"mutated_content": "while True: pass"}
        result = await engine.sandbox_test(proposal, sandbox=mock_sandbox)

        assert result["passed"] is False

    @pytest.mark.asyncio
    async def test_sandbox_test_with_empty_content(self) -> None:
        """Sandbox test with empty mutated content returns passed=False."""
        mock_sandbox = _make_mock_sandbox()
        engine = SelfEvolutionEngine()
        proposal = {"mutated_content": ""}
        result = await engine.sandbox_test(proposal, sandbox=mock_sandbox)

        assert result["passed"] is False
        assert "reason" in result

    @pytest.mark.asyncio
    async def test_sandbox_test_with_sandbox_exception(self) -> None:
        """Sandbox test handles sandbox exceptions gracefully."""
        mock_sandbox = MagicMock()
        mock_sandbox.execute_code = AsyncMock(side_effect=RuntimeError("sandbox crashed"))
        engine = SelfEvolutionEngine()
        proposal = {"mutated_content": "x = 1"}
        result = await engine.sandbox_test(proposal, sandbox=mock_sandbox)

        assert result["passed"] is False
        assert "sandbox crashed" in result["reason"]

    # ── A/B test tests ───────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_ab_test_with_no_sandbox(self) -> None:
        """A/B test without sandbox returns is_significant=True."""
        engine = SelfEvolutionEngine()
        proposal = {"original_content": "old", "mutated_content": "new"}
        result = await engine.ab_test(proposal, sandbox=None)

        assert result["is_significant"] is True
        assert result["note"] == "A/B test skipped (no sandbox)"

    @pytest.mark.asyncio
    async def test_ab_test_with_mock_sandbox_faster_treatment(self) -> None:
        """A/B test: treatment is faster than control -> is_significant=True."""
        mock_sandbox = MagicMock()
        control_result = _make_sandbox_result(success=True, duration_seconds=0.5)
        treatment_result = _make_sandbox_result(success=True, duration_seconds=0.2)
        mock_sandbox.execute_code = AsyncMock(
            side_effect=[control_result, treatment_result]
        )
        engine = SelfEvolutionEngine()
        proposal = {"original_content": "old code", "mutated_content": "new code"}
        result = await engine.ab_test(proposal, sandbox=mock_sandbox)

        assert result["is_significant"] is True
        assert result["treatment_result"]["success"] is True
        assert result["sample_size"] == 1
        assert mock_sandbox.execute_code.await_count == 2

    @pytest.mark.asyncio
    async def test_ab_test_with_slower_treatment(self) -> None:
        """A/B test: treatment is 2x slower -> is_significant=False."""
        mock_sandbox = MagicMock()
        control_result = _make_sandbox_result(success=True, duration_seconds=0.1)
        treatment_result = _make_sandbox_result(success=True, duration_seconds=0.5)
        mock_sandbox.execute_code = AsyncMock(
            side_effect=[control_result, treatment_result]
        )
        engine = SelfEvolutionEngine()
        proposal = {"original_content": "old code", "mutated_content": "new code"}
        result = await engine.ab_test(proposal, sandbox=mock_sandbox)

        # 0.5 > 0.1 * 1.1 = 0.11, so treatment is too slow
        assert result["is_significant"] is False

    @pytest.mark.asyncio
    async def test_ab_test_with_failing_treatment(self) -> None:
        """A/B test: treatment fails -> is_significant=False."""
        mock_sandbox = MagicMock()
        control_result = _make_sandbox_result(success=True, duration_seconds=0.1)
        treatment_result = _make_sandbox_result(success=False, exit_code=1, duration_seconds=0.05)
        mock_sandbox.execute_code = AsyncMock(
            side_effect=[control_result, treatment_result]
        )
        engine = SelfEvolutionEngine()
        proposal = {"original_content": "old code", "mutated_content": "new code"}
        result = await engine.ab_test(proposal, sandbox=mock_sandbox)

        assert result["is_significant"] is False

    @pytest.mark.asyncio
    async def test_ab_test_with_failing_control(self) -> None:
        """A/B test: control fails but treatment succeeds -> is_significant=True."""
        mock_sandbox = MagicMock()
        control_result = _make_sandbox_result(success=False, exit_code=1, duration_seconds=0.1)
        treatment_result = _make_sandbox_result(success=True, duration_seconds=0.5)
        mock_sandbox.execute_code = AsyncMock(
            side_effect=[control_result, treatment_result]
        )
        engine = SelfEvolutionEngine()
        proposal = {"original_content": "old code", "mutated_content": "new code"}
        result = await engine.ab_test(proposal, sandbox=mock_sandbox)

        # When control fails, treatment succeeds => significant
        assert result["is_significant"] is True

    @pytest.mark.asyncio
    async def test_ab_test_with_no_original_content(self) -> None:
        """A/B test with no original content runs treatment only."""
        mock_sandbox = _make_mock_sandbox(success=True, duration_seconds=0.1)
        engine = SelfEvolutionEngine()
        proposal = {"original_content": "", "mutated_content": "new code"}
        result = await engine.ab_test(proposal, sandbox=mock_sandbox)

        assert result["is_significant"] is True
        assert result["control_result"] is None
        mock_sandbox.execute_code.assert_awaited_once_with("new code")

    @pytest.mark.asyncio
    async def test_ab_test_with_sandbox_exception(self) -> None:
        """A/B test handles sandbox exceptions gracefully."""
        mock_sandbox = MagicMock()
        mock_sandbox.execute_code = AsyncMock(side_effect=RuntimeError("sandbox crashed"))
        engine = SelfEvolutionEngine()
        proposal = {"original_content": "old code", "mutated_content": "new code"}
        result = await engine.ab_test(proposal, sandbox=mock_sandbox)

        assert result["is_significant"] is False
        assert "reason" in result

    # ── Deploy tests ─────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_deploy_rejected_if_validation_fails(self) -> None:
        """Deploy with failed validation returns deployed=False."""
        engine = SelfEvolutionEngine()
        proposal = {"mutation_type": MutationType.CODE, "description": "test"}
        validation = {"passed": False, "reason": "Safety check failed"}
        result = await engine.deploy(proposal, validation)

        assert result["deployed"] is False

    @pytest.mark.asyncio
    async def test_deploy_with_git_tracker(self) -> None:
        """Deploy with git tracker applies mutation and creates snapshot."""
        mock_tracker = _make_mock_git_tracker()
        engine = SelfEvolutionEngine()
        proposal = {
            "mutation_type": MutationType.CODE,
            "description": "test mutation",
            "target_path": "test.py",
            "mutated_content": "print(1)",
            "rationale": "testing",
        }
        validation = {"passed": True}
        result = await engine.deploy(proposal, validation, git_tracker=mock_tracker)

        assert result["deployed"] is True
        assert result["commit_hash"] == "abcdef1234567890"
        assert result["generation"] == 1
        mock_tracker.apply_mutation.assert_awaited_once_with("test.py", "print(1)")
        mock_tracker.snapshot.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_deploy_without_git_tracker(self) -> None:
        """Deploy without git tracker still succeeds with commit_hash=None."""
        engine = SelfEvolutionEngine()
        proposal = {
            "mutation_type": MutationType.CODE,
            "description": "test",
            "mutated_content": "x = 1",
        }
        validation = {"passed": True}
        result = await engine.deploy(proposal, validation)

        assert result["deployed"] is True
        assert result["commit_hash"] is None

    @pytest.mark.asyncio
    async def test_deploy_increments_generation(self) -> None:
        """Each deployment increments the generation counter."""
        engine = SelfEvolutionEngine()
        proposal = {"mutation_type": MutationType.CODE, "description": "t1", "mutated_content": "x"}
        validation = {"passed": True}

        result1 = await engine.deploy(proposal, validation)
        assert result1["generation"] == 1

        result2 = await engine.deploy(proposal, validation)
        assert result2["generation"] == 2

    # ── Reject tests ─────────────────────────────────────────────────

    def test_reject_creates_rejection_record(self) -> None:
        """Reject creates a rejection record with reason."""
        proposal = {
            "description": "Optimize prompts",
            "mutation_type": MutationType.PROMPT,
        }
        reason = {"reason": "Sandbox test failed", "note": "timeout"}
        result = SelfEvolutionEngine.reject(proposal, reason)

        assert result["deployed"] is False
        assert "Sandbox test failed" in result["reason"]
        assert result["proposal"] == "Optimize prompts"

    def test_reject_extracts_reason_from_note(self) -> None:
        """Reject extracts reason from 'note' when 'reason' is absent."""
        proposal = {"description": "test"}
        reason = {"note": "sandbox not available"}
        result = SelfEvolutionEngine.reject(proposal, reason)

        assert result["deployed"] is False
        assert "sandbox not available" in result["reason"]

    # ── Run cycle tests ──────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_run_cycle_full_pipeline_deploy(self) -> None:
        """Full run_cycle with mocked deps deploys successfully."""
        # Use a safety pipeline that passes clean code
        mock_safety = MagicMock()
        mock_safety.validate = AsyncMock(return_value={
            "passed": True,
            "layers": {"syntax": {"passed": True}},
        })

        mock_gateway = _make_mock_gateway()
        mock_sandbox = _make_mock_sandbox(success=True, duration_seconds=0.1)
        mock_tracker = _make_mock_git_tracker()

        engine = SelfEvolutionEngine(
            safety_pipeline=mock_safety,
            gateway=mock_gateway,
        )
        result = await engine.run_cycle(
            execution_history=[],
            failure_patterns=["timeout"],
            sandbox=mock_sandbox,
            git_tracker=mock_tracker,
        )

        assert result["status"] == "deployed"
        assert result["deployed"] is True
        assert result["mutations_proposed"] == 1
        assert result["mutations_deployed"] == 1
        assert result["proposal"] is not None
        assert result["validation"] is not None
        assert result["sandbox_result"] is not None
        assert result["ab_result"] is not None
        assert result["deployment"] is not None

    @pytest.mark.asyncio
    async def test_run_cycle_with_no_opportunities(self) -> None:
        """Run cycle with no opportunities returns status=no_opportunities.

        We override analyze to return no opportunities, then call run_cycle
        which checks the opportunities list.
        """
        engine = SelfEvolutionEngine()

        # Patch analyze to return empty opportunities
        with patch.object(
            engine, "analyze",
            new_callable=AsyncMock,
            return_value={"opportunities": [], "performance_metrics": {}},
        ):
            result = await engine.run_cycle(
                execution_history=[],
                failure_patterns=[],
            )

        assert result["status"] == "no_opportunities"
        assert result["deployed"] is False
        assert result["mutations_proposed"] == 0
        assert result["mutations_deployed"] == 0

    @pytest.mark.asyncio
    async def test_run_cycle_validation_failure(self) -> None:
        """Run cycle with safety validation failure returns status=validation_failed."""
        mock_safety = MagicMock()
        mock_safety.validate = AsyncMock(return_value={
            "passed": False,
            "layers": {"security": {"passed": False, "details": "forbidden pattern"}},
        })

        engine = SelfEvolutionEngine(safety_pipeline=mock_safety)
        result = await engine.run_cycle(
            execution_history=[],
            failure_patterns=["test failure"],
        )

        assert result["status"] == "validation_failed"
        assert result["deployed"] is False
        assert result["mutations_proposed"] == 1
        assert result["mutations_deployed"] == 0
        assert result["proposal"] is not None
        assert result["validation"] is not None
        assert result["rejection"] is not None

    @pytest.mark.asyncio
    async def test_run_cycle_sandbox_failure(self) -> None:
        """Run cycle with sandbox failure returns status=sandbox_failed."""
        mock_safety = MagicMock()
        mock_safety.validate = AsyncMock(return_value={
            "passed": True,
            "layers": {},
        })
        mock_sandbox = _make_mock_sandbox(success=False, exit_code=1)

        engine = SelfEvolutionEngine(safety_pipeline=mock_safety)
        result = await engine.run_cycle(
            execution_history=[],
            failure_patterns=["test failure"],
            sandbox=mock_sandbox,
        )

        assert result["status"] == "sandbox_failed"
        assert result["deployed"] is False
        assert result["mutations_proposed"] == 1
        assert result["mutations_deployed"] == 0

    @pytest.mark.asyncio
    async def test_run_cycle_ab_test_rejection(self) -> None:
        """Run cycle where A/B test shows regression returns status=rejected.

        Since run_cycle calls generate() without current_content, the proposal's
        original_content is None. ab_test then runs treatment only (no original
        to compare against). If treatment fails, is_significant=False.
        """
        mock_safety = MagicMock()
        mock_safety.validate = AsyncMock(return_value={
            "passed": True,
            "layers": {},
        })

        mock_sandbox = MagicMock()
        sandbox_pass_result = _make_sandbox_result(success=True, duration_seconds=0.1)
        # ab_test treatment-only branch: treatment fails -> is_significant=False
        treatment_fail = _make_sandbox_result(success=False, exit_code=1, duration_seconds=0.5)
        mock_sandbox.execute_code = AsyncMock(
            side_effect=[
                sandbox_pass_result,  # sandbox_test (must succeed)
                treatment_fail,       # ab_test treatment-only (fails)
            ]
        )

        engine = SelfEvolutionEngine(safety_pipeline=mock_safety)
        result = await engine.run_cycle(
            execution_history=[],
            failure_patterns=["test failure"],
            sandbox=mock_sandbox,
        )

        assert result["status"] == "rejected"
        assert result["deployed"] is False
        assert result["mutations_proposed"] == 1
        assert result["mutations_deployed"] == 0

    @pytest.mark.asyncio
    async def test_run_cycle_with_reflection(self) -> None:
        """Run cycle extracts failure_patterns from reflection when not provided."""
        mock_safety = MagicMock()
        mock_safety.validate = AsyncMock(return_value={
            "passed": True,
            "layers": {},
        })

        # Create a mock reflection with lessons_learned and errors
        mock_reflection = MagicMock()
        mock_reflection.lessons_learned = ["slow response time", "retries exhausted"]
        mock_reflection.errors = ["error: timeout"]

        mock_sandbox = _make_mock_sandbox(success=True, duration_seconds=0.1)
        engine = SelfEvolutionEngine(safety_pipeline=mock_safety)

        # We patch analyze to verify that the failure_patterns extracted from
        # reflection are passed through correctly
        original_analyze = engine.analyze
        captured_patterns: list[str] = []

        async def _capturing_analyze(
            execution_history: list[dict],
            failure_patterns: list[str],
        ) -> dict:
            captured_patterns.extend(failure_patterns)
            return await original_analyze(execution_history, failure_patterns)

        with patch.object(engine, "analyze", side_effect=_capturing_analyze):
            await engine.run_cycle(
                execution_history=[],
                failure_patterns=None,
                reflection=mock_reflection,
                sandbox=mock_sandbox,
            )

        # failure_patterns should be extracted from reflection
        assert "slow response time" in captured_patterns
        assert "retries exhausted" in captured_patterns
        assert "error: timeout" in captured_patterns

    @pytest.mark.asyncio
    async def test_run_cycle_with_reflection_no_patterns(self) -> None:
        """Run cycle with reflection that has no lessons or errors uses empty list."""
        mock_reflection = MagicMock(spec=[])
        # No lessons_learned or errors attributes
        engine = SelfEvolutionEngine()

        with patch.object(
            engine, "analyze",
            new_callable=AsyncMock,
            return_value={"opportunities": [], "performance_metrics": {}},
        ):
            result = await engine.run_cycle(
                execution_history=[],
                failure_patterns=None,
                reflection=mock_reflection,
            )

        assert result["status"] == "no_opportunities"

    @pytest.mark.asyncio
    async def test_run_cycle_no_reflection_no_patterns(self) -> None:
        """Run cycle with no reflection and no failure_patterns uses empty list."""
        engine = SelfEvolutionEngine()

        with patch.object(
            engine, "analyze",
            new_callable=AsyncMock,
            return_value={"opportunities": [], "performance_metrics": {}},
        ):
            result = await engine.run_cycle(
                execution_history=[],
                failure_patterns=None,
                reflection=None,
            )

        assert result["status"] == "no_opportunities"
