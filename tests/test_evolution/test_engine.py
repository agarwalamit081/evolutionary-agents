"""Tests for src.evolution.engine — SelfEvolutionEngine 4-phase pipeline."""

from __future__ import annotations

import json
import uuid
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
    mock_tracker.get_current_hash = AsyncMock(return_value="predeploy000000")
    return mock_tracker


def _make_mock_persister() -> MagicMock:
    """Create a mock EvolutionPersister with all async methods stubbed.

    Returns realistic values (chain/mutation UUIDs, True for status updates)
    so the engine's persistence calls flow through their normal branches.
    """
    p = MagicMock()
    p.create_chain = AsyncMock(return_value=uuid.uuid4())
    p.record_mutation = AsyncMock(return_value=uuid.uuid4())
    p.update_mutation_status = AsyncMock(return_value=True)
    p.record_event = AsyncMock(return_value=None)
    p.complete_chain = AsyncMock(return_value=True)
    return p


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

    @pytest.mark.asyncio
    async def test_analyze_emits_tool_opportunity_for_low_success(self) -> None:
        """A tool with >=3 calls and <50% success yields a targeted TOOL opp."""
        engine = SelfEvolutionEngine()
        history = [
            {"tool_results": [{"tool_name": "flaky_tool", "success": False, "output": ""}]}
        ] * 3
        result = await engine.analyze(execution_history=history, failure_patterns=[])

        tool_opps = [
            o for o in result["opportunities"] if o.get("target_tool") == "flaky_tool"
        ]
        assert len(tool_opps) == 1
        assert tool_opps[0]["priority"] == "high"
        assert tool_opps[0]["tool_metrics"]["calls"] == 3
        assert tool_opps[0]["tool_metrics"]["success_rate"] < 0.5

    @pytest.mark.asyncio
    async def test_analyze_emits_tool_opportunity_for_high_empty(self) -> None:
        """A tool succeeding but producing empty output yields a targeted opp."""
        engine = SelfEvolutionEngine()
        history = [
            {"tool_results": [{"tool_name": "empty_tool", "success": True, "output": ""}]},
            {"tool_results": [{"tool_name": "empty_tool", "success": True, "output": "   "}]},
            {"tool_results": [{"tool_name": "empty_tool", "success": True, "output": ""}]},
        ]
        result = await engine.analyze(execution_history=history, failure_patterns=[])

        tool_opps = [
            o for o in result["opportunities"] if o.get("target_tool") == "empty_tool"
        ]
        assert len(tool_opps) == 1
        assert tool_opps[0]["tool_metrics"]["empty_output_rate"] > 0.5

    @pytest.mark.asyncio
    async def test_analyze_healthy_tool_falls_back_to_generic(self) -> None:
        """A healthy tool gets no targeted opp; the generic TOOL opp remains."""
        engine = SelfEvolutionEngine()
        history = [
            {"tool_results": [{"tool_name": "good_tool", "success": True, "output": "data"}]}
        ] * 5
        result = await engine.analyze(execution_history=history, failure_patterns=[])

        assert not any(
            o.get("target_tool") == "good_tool" for o in result["opportunities"]
        )
        assert MutationType.TOOL in [o["type"] for o in result["opportunities"]]

    @pytest.mark.asyncio
    async def test_analyze_too_few_calls_no_targeted_opportunity(self) -> None:
        """Under the 3-call significance floor, no targeted opp is emitted."""
        engine = SelfEvolutionEngine()
        history = [
            {"tool_results": [{"tool_name": "rare_tool", "success": False, "output": ""}]},
            {"tool_results": [{"tool_name": "rare_tool", "success": False, "output": ""}]},
        ]
        result = await engine.analyze(execution_history=history, failure_patterns=[])

        assert not any(
            o.get("target_tool") == "rare_tool" for o in result["opportunities"]
        )

    @pytest.mark.asyncio
    async def test_analyze_aggregates_tool_result_objects(self) -> None:
        """ToolResult pydantic objects (attribute access) are aggregated too."""
        from src.graph.models import ToolResult

        engine = SelfEvolutionEngine()
        history = [
            {
                "tool_results": [
                    ToolResult(tool_name="obj_tool", success=False, output=""),
                    ToolResult(tool_name="obj_tool", success=False, output=""),
                    ToolResult(tool_name="obj_tool", success=False, output=""),
                ]
            }
        ]
        result = await engine.analyze(execution_history=history, failure_patterns=[])

        tool_opps = [
            o for o in result["opportunities"] if o.get("target_tool") == "obj_tool"
        ]
        assert len(tool_opps) == 1
        assert tool_opps[0]["tool_metrics"]["calls"] == 3

    # ── Memory analyzer (2c) ─────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_low_recall_history_proposes_memory_mutation(self) -> None:
        """Most retrieval steps return nothing → MEMORY opp (recall-focused)."""
        engine = SelfEvolutionEngine()
        # 4 of 5 retrieval steps missed (retrieved 0) → miss_rate 0.8 > 0.5.
        history = [
            {"memory_retrieval": {"retrieved": 0, "used": 0}},
            {"memory_retrieval": {"retrieved": 2, "used": 1}},
            {"memory_retrieval": {"retrieved": 0, "used": 0}},
            {"memory_retrieval": {"retrieved": 0, "used": 0}},
            {"memory_retrieval": {"retrieved": 0, "used": 0}},
        ]
        result = await engine.analyze(execution_history=history, failure_patterns=[])

        memory_opps = [
            o for o in result["opportunities"] if o["type"] == MutationType.MEMORY
        ]
        assert len(memory_opps) == 1
        assert "recall" in memory_opps[0]["description"].lower()
        assert memory_opps[0]["memory_signal"]["miss_rate"] > 0.5

    @pytest.mark.asyncio
    async def test_noisy_retrieval_proposes_precision_mutation(self) -> None:
        """Much retrieved, little used → MEMORY opp (precision-focused)."""
        engine = SelfEvolutionEngine()
        # Every step retrieves 10, uses 1 → useful_rate 0.1 < 0.3 (no misses).
        history = [
            {"memory_retrieval": {"retrieved": 10, "used": 1}},
        ] * 4
        result = await engine.analyze(execution_history=history, failure_patterns=[])

        memory_opps = [
            o for o in result["opportunities"] if o["type"] == MutationType.MEMORY
        ]
        assert len(memory_opps) == 1
        assert "precision" in memory_opps[0]["description"].lower()
        assert memory_opps[0]["memory_signal"]["useful_rate"] < 0.3

    @pytest.mark.asyncio
    async def test_no_memory_data_proposes_nothing(self) -> None:
        """No memory_retrieval records → no MEMORY opp (backward compat)."""
        engine = SelfEvolutionEngine()
        history = [{"duration_ms": 100}, {"tool_results": []}]
        result = await engine.analyze(execution_history=history, failure_patterns=[])

        assert not any(
            o["type"] == MutationType.MEMORY for o in result["opportunities"]
        )

    @pytest.mark.asyncio
    async def test_too_few_memory_samples_proposes_nothing(self) -> None:
        """Under the 3-sample floor, no MEMORY opp is emitted."""
        engine = SelfEvolutionEngine()
        history = [{"memory_retrieval": {"retrieved": 0, "used": 0}}] * 2
        result = await engine.analyze(execution_history=history, failure_patterns=[])

        assert not any(
            o["type"] == MutationType.MEMORY for o in result["opportunities"]
        )

    @pytest.mark.asyncio
    async def test_memory_opportunity_deploys_to_config(self) -> None:
        """A MEMORY opp flows through generate() (heuristic) to a strategy config."""
        import json

        engine = SelfEvolutionEngine()  # no gateway → heuristic template
        # Low-recall description → generate_memory_config picks recall_focused.
        opportunity = {
            "type": MutationType.MEMORY,
            "description": (
                "Memory recall is low — 80% of retrieval steps missed (4/5). "
                "Broaden retrieval (recall-focused)."
            ),
            "priority": "medium",
        }
        result = await engine.generate(opportunity)

        assert result["mutation_type"] == MutationType.MEMORY
        config = json.loads(result["mutated_content"])
        assert config["strategy"] == "recall_focused"
        assert "min_fitness" in config and "max_results" in config
        assert result["target_path"] == "evolution/memory_config.json"

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
        """Generate without gateway uses heuristic fallback with real content."""
        engine = SelfEvolutionEngine()
        opportunity = {"type": MutationType.CODE, "description": "Optimize", "priority": "low"}
        result = await engine.generate(opportunity, current_content="x = 1")

        assert result["mutation_type"] == MutationType.CODE
        assert result["model_used"] is None
        assert result["tokens_used"] == 0
        # Heuristic now produces structured JSON content, not comments
        assert result["mutated_content"] is not None
        assert not result["mutated_content"].startswith("#")  # No comment-only content
        assert result["target_path"] is not None  # Always set now

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
        assert result["mutated_content"] is not None
        assert not result["mutated_content"].startswith("#")

    @pytest.mark.asyncio
    async def test_generate_code_routes_to_codegen_model(self) -> None:
        """CODE mutations use the codegen model + JSON mode, not complexity."""
        mock_gateway = _make_mock_gateway()
        engine = SelfEvolutionEngine(gateway=mock_gateway)
        opportunity = {
            "type": MutationType.CODE,
            "description": "Optimize handler",
            "priority": "high",
        }
        await engine.generate(opportunity, current_content="async def f():\n    return 1\n")

        assert mock_gateway.acompletion.await_count == 1
        _, kwargs = mock_gateway.acompletion.call_args
        assert kwargs.get("model") == "deepseek-v4-pro"
        assert kwargs.get("response_format") == {"type": "json_object"}
        assert "complexity" not in kwargs

    @pytest.mark.asyncio
    async def test_generate_tool_routes_to_codegen_model(self) -> None:
        """TOOL mutations emit a tools/*.py module, so they use the codegen model.

        Regression: TOOL mutations previously fell through to the generic
        complexity-routed (code-weak) model, which truncated the handler every
        retry — evolution never deployed a tool fix. Code-emitting mutations
        (CODE + TOOL) both route to the code-strong codegen model + JSON mode.
        """
        mock_gateway = _make_mock_gateway()
        engine = SelfEvolutionEngine(gateway=mock_gateway)
        opportunity = {
            "type": MutationType.TOOL,
            "description": "Optimize duplicate_finder tool",
            "priority": "high",
        }
        await engine.generate(opportunity, current_content="async def f():\n    return 1\n")

        assert mock_gateway.acompletion.await_count == 1
        _, kwargs = mock_gateway.acompletion.call_args
        assert kwargs.get("model") == "deepseek-v4-pro"
        assert kwargs.get("response_format") == {"type": "json_object"}
        assert "complexity" not in kwargs

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

    @pytest.mark.asyncio
    async def test_validate_preservation_violation_surfaces_category(self) -> None:
        """A mutation that flips eval_enabled is rejected with the typed Q93
        category in the reason (not the generic 'failed safety layers')."""
        engine = SelfEvolutionEngine()
        proposal = {
            "mutation_type": MutationType.CONFIG,
            "description": "disable eval",
            "mutated_content": "eval_enabled = False\n",
        }
        result = await engine.validate(proposal)
        assert result["passed"] is False
        assert "safety-preservation" in result["reason"]
        assert "gate_flag_flip" in result["reason"]
        # The full typed violations list is in details for the dashboard.
        violations = result["details"]["layers"]["preservation"]["violations"]
        assert any(v["category"] == "gate_flag_flip" for v in violations)

    @pytest.mark.asyncio
    async def test_validate_code_threads_sandbox_root(self) -> None:
        """validate() adds sandbox_root to the safety context for CODE mutations."""
        mock_safety = MagicMock()
        mock_safety.validate = AsyncMock(
            return_value={"passed": True, "layers": {}, "issues": []}
        )
        engine = SelfEvolutionEngine(safety_pipeline=mock_safety)
        proposal = {
            "mutation_type": MutationType.CODE,
            "description": "memoize",
            "mutated_content": "async def f():\n    return 1\n",
        }
        await engine.validate(proposal)
        _, kwargs = mock_safety.validate.call_args
        assert "sandbox_root" in kwargs["context"]

    @pytest.mark.asyncio
    async def test_validate_tool_threads_sandbox_root(self) -> None:
        """TOOL mutations emit code too, so validate() threads sandbox_root.

        Regression for the same Bug A class: only CODE got the sandbox_root
        context, so a TOOL mutation's writes were Layer-5-checked against no
        root. Code-emitting mutations (CODE + TOOL) both scope writes.
        """
        mock_safety = MagicMock()
        mock_safety.validate = AsyncMock(
            return_value={"passed": True, "layers": {}, "issues": []}
        )
        engine = SelfEvolutionEngine(safety_pipeline=mock_safety)
        proposal = {
            "mutation_type": MutationType.TOOL,
            "description": "memoize tool handler",
            "mutated_content": "async def f():\n    return 1\n",
        }
        await engine.validate(proposal)
        _, kwargs = mock_safety.validate.call_args
        assert "sandbox_root" in kwargs["context"]

    @pytest.mark.asyncio
    async def test_validate_non_code_omits_sandbox_root(self) -> None:
        """Non-CODE mutations don't add sandbox_root (no file-write scoping)."""
        mock_safety = MagicMock()
        mock_safety.validate = AsyncMock(
            return_value={"passed": True, "layers": {}, "issues": []}
        )
        engine = SelfEvolutionEngine(safety_pipeline=mock_safety)
        proposal = {
            "mutation_type": MutationType.PROMPT,
            "description": "tweak",
            "mutated_content": "{}",
        }
        await engine.validate(proposal)
        _, kwargs = mock_safety.validate.call_args
        assert "sandbox_root" not in kwargs["context"]

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

    @pytest.mark.asyncio
    async def test_deploy_captures_pre_deploy_hash(self) -> None:
        """Deploy captures the pre-deploy hash so M6 can roll back to it."""
        mock_tracker = _make_mock_git_tracker()
        engine = SelfEvolutionEngine()
        proposal = {
            "mutation_type": MutationType.CODE,
            "description": "test",
            "target_path": "t.py",
            "mutated_content": "x = 1",
        }
        result = await engine.deploy(
            proposal, {"passed": True}, git_tracker=mock_tracker
        )

        assert result["deployed"] is True
        assert result["pre_deploy_hash"] == "predeploy000000"
        # Captured BEFORE the mutation is applied.
        mock_tracker.get_current_hash.assert_awaited_once()
        assert (
            mock_tracker.get_current_hash.await_count
            <= mock_tracker.apply_mutation.await_count
        )

    @pytest.mark.asyncio
    async def test_deploy_without_tracker_has_no_pre_deploy_hash(self) -> None:
        """No git tracker → pre_deploy_hash is None (nothing to roll back to)."""
        engine = SelfEvolutionEngine()
        proposal = {"mutation_type": MutationType.CODE, "description": "t", "mutated_content": "x"}
        result = await engine.deploy(proposal, {"passed": True})

        assert result["pre_deploy_hash"] is None

    @pytest.mark.asyncio
    async def test_deploy_captures_diff_content(self) -> None:
        """Phase 3a: a tracker with get_diff records the mutation's diff.

        The diff is computed from the pre-deploy hash (captured BEFORE apply) so
        the mutation record carries a reviewable, rollback-ready diff — populating
        the previously-unused diff_content column.
        """
        mock_tracker = _make_mock_git_tracker()
        mock_tracker.get_diff = AsyncMock(return_value="@@ -1 +1 @@\n-x\n+y")
        engine = SelfEvolutionEngine()
        proposal = {
            "mutation_type": MutationType.CODE,
            "description": "test",
            "target_path": "t.py",
            "mutated_content": "y",
        }
        result = await engine.deploy(
            proposal, {"passed": True}, git_tracker=mock_tracker
        )

        assert result["deployed"] is True
        assert result["diff_content"] == "@@ -1 +1 @@\n-x\n+y"
        # The diff is scoped to the pre-deploy hash, so it isolates THIS mutation.
        mock_tracker.get_diff.assert_awaited_once_with(since_hash="predeploy000000")

    @pytest.mark.asyncio
    async def test_deploy_diff_computed_after_apply(self) -> None:
        """Phase 3a: the diff is captured AFTER apply (so it reflects the change)."""
        mock_tracker = _make_mock_git_tracker()
        mock_tracker.get_diff = AsyncMock(return_value="diff")
        engine = SelfEvolutionEngine()
        proposal = {
            "mutation_type": MutationType.CODE,
            "description": "test",
            "target_path": "t.py",
            "mutated_content": "y",
        }
        await engine.deploy(proposal, {"passed": True}, git_tracker=mock_tracker)

        # Order must be: current_hash → apply_mutation → snapshot → get_diff.
        # method_calls entries are `call` objects whose first element is the name.
        names = [c[0] for c in mock_tracker.method_calls]
        idx_apply = names.index("apply_mutation")
        idx_diff = names.index("get_diff")
        assert idx_diff > idx_apply

    @pytest.mark.asyncio
    async def test_deploy_best_effort_diff_failure(self) -> None:
        """Phase 3a: a get_diff failure is non-fatal — deploy still succeeds, diff None."""
        mock_tracker = _make_mock_git_tracker()
        mock_tracker.get_diff = AsyncMock(side_effect=RuntimeError("git broken"))
        engine = SelfEvolutionEngine()
        proposal = {
            "mutation_type": MutationType.CODE,
            "description": "test",
            "target_path": "t.py",
            "mutated_content": "y",
        }
        result = await engine.deploy(
            proposal, {"passed": True}, git_tracker=mock_tracker
        )

        assert result["deployed"] is True
        assert result["diff_content"] is None
        assert result["commit_hash"] == "abcdef1234567890"

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
        """Run cycle with sandbox failure returns status=sandbox_failed.

        Patches generate to return a CODE mutation so sandbox is actually
        executed (PROMPT mutations skip sandbox).
        """
        mock_safety = MagicMock()
        mock_safety.validate = AsyncMock(return_value={
            "passed": True,
            "layers": {},
        })
        mock_sandbox = _make_mock_sandbox(success=False, exit_code=1)

        engine = SelfEvolutionEngine(safety_pipeline=mock_safety)

        with patch.object(
            engine, "generate",
            new_callable=AsyncMock,
            return_value={
                "mutation_type": MutationType.CODE,
                "description": "test code mutation",
                "original_content": "old",
                "mutated_content": "raise RuntimeError('boom')",
                "target_path": "test.py",
                "priority": "high",
                "rationale": "testing sandbox failure",
                "model_used": None,
                "tokens_used": 0,
            },
        ):
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

        Patches generate to return a CODE mutation so A/B test is actually
        executed (PROMPT mutations skip A/B test).
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

        with patch.object(
            engine, "generate",
            new_callable=AsyncMock,
            return_value={
                "mutation_type": MutationType.CODE,
                "description": "test code mutation",
                "original_content": None,
                "mutated_content": "new code",
                "target_path": "test.py",
                "priority": "high",
                "rationale": "testing AB rejection",
                "model_used": None,
                "tokens_used": 0,
            },
        ):
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

    # ── Heuristic template tests ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_heuristic_generate_produces_json_content(self) -> None:
        """Heuristic generation produces valid JSON content, not comments."""
        engine = SelfEvolutionEngine()
        opportunity = {
            "type": MutationType.PROMPT,
            "description": "Fix JSON parsing",
            "priority": "high",
            "patterns": ["invalid JSON from LLM"],
        }
        result = await engine.generate(opportunity)

        import json
        parsed = json.loads(result["mutated_content"])
        assert "suffixes" in parsed
        assert result["target_path"] is not None

    @pytest.mark.asyncio
    async def test_heuristic_generate_sets_target_path_for_all_types(self) -> None:
        """All mutation types get a non-None target_path from templates."""
        engine = SelfEvolutionEngine()
        for mtype in MutationType:
            opportunity = {"type": mtype, "description": "test", "priority": "low"}
            result = await engine.generate(opportunity)
            assert result["target_path"] is not None, f"target_path is None for {mtype}"

    # ── Sandbox skip for non-code mutations ───────────────────────────

    @pytest.mark.asyncio
    async def test_sandbox_skips_prompt_mutation(self) -> None:
        """Sandbox test is skipped for PROMPT mutations."""
        mock_sandbox = _make_mock_sandbox()
        engine = SelfEvolutionEngine()
        proposal = {
            "mutated_content": '{"suffixes": ["test"]}',
            "mutation_type": MutationType.PROMPT,
        }
        result = await engine.sandbox_test(proposal, sandbox=mock_sandbox)

        assert result["passed"] is True
        assert "sandbox skipped" in result["note"]
        mock_sandbox.execute_code.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_sandbox_skips_config_mutation(self) -> None:
        """Sandbox test is skipped for CONFIG mutations."""
        mock_sandbox = _make_mock_sandbox()
        engine = SelfEvolutionEngine()
        proposal = {
            "mutated_content": '{"temperature": 0.4}',
            "mutation_type": MutationType.CONFIG,
        }
        result = await engine.sandbox_test(proposal, sandbox=mock_sandbox)

        assert result["passed"] is True
        mock_sandbox.execute_code.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_sandbox_runs_code_mutation(self) -> None:
        """Sandbox test IS executed for CODE mutations."""
        mock_sandbox = _make_mock_sandbox(success=True)
        engine = SelfEvolutionEngine()
        proposal = {
            "mutated_content": "x = 1",
            "mutation_type": MutationType.CODE,
        }
        result = await engine.sandbox_test(proposal, sandbox=mock_sandbox)

        assert result["passed"] is True
        mock_sandbox.execute_code.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ab_test_skips_prompt_mutation(self) -> None:
        """A/B test is skipped for PROMPT mutations."""
        mock_sandbox = _make_mock_sandbox()
        engine = SelfEvolutionEngine()
        proposal = {
            "original_content": "old",
            "mutated_content": "new",
            "mutation_type": MutationType.PROMPT,
        }
        result = await engine.ab_test(proposal, sandbox=mock_sandbox)

        assert result["is_significant"] is True
        assert "A/B test skipped" in result["note"]
        mock_sandbox.execute_code.assert_not_awaited()

    # ── Deploy fallback target_path ───────────────────────────────────

    @pytest.mark.asyncio
    async def test_deploy_with_no_target_path_uses_fallback(self) -> None:
        """Deploy with None target_path still commits to shadow repo."""
        mock_tracker = _make_mock_git_tracker()
        engine = SelfEvolutionEngine()
        proposal = {
            "mutation_type": MutationType.PROMPT,
            "description": "test",
            "target_path": None,
            "mutated_content": '{"suffixes": ["test"]}',
        }
        validation = {"passed": True}
        result = await engine.deploy(proposal, validation, git_tracker=mock_tracker)

        assert result["deployed"] is True
        mock_tracker.apply_mutation.assert_awaited_once()
        called_path = mock_tracker.apply_mutation.call_args[0][0]
        assert "evolution/" in called_path

    # ── Retry-with-feedback + persistence tests ──────────────────────

    @pytest.mark.asyncio
    async def test_run_cycle_retries_and_feeds_back_validation_error(self) -> None:
        """A validation failure triggers a retry; the error is fed back to the LLM."""
        mock_safety = MagicMock()
        mock_safety.validate = AsyncMock(side_effect=[
            {
                "passed": False,
                "layers": {"sec": {"passed": False}},
                "reason": "syntax error: unclosed paren at line 41",
            },
            {"passed": True, "layers": {}},
        ])
        mock_gateway = _make_mock_gateway()
        mock_sandbox = _make_mock_sandbox(success=True, duration_seconds=0.1)

        engine = SelfEvolutionEngine(
            safety_pipeline=mock_safety, gateway=mock_gateway, max_retries=2,
        )
        result = await engine.run_cycle(
            execution_history=[], failure_patterns=["timeout"],
            sandbox=mock_sandbox,
        )

        # Initial attempt + one retry that passed.
        assert mock_gateway.acompletion.await_count == 2
        # The retry's user message contains the fed-back validation error block;
        # the initial attempt's message must NOT contain it.
        first_messages = mock_gateway.acompletion.await_args_list[0].kwargs["messages"]
        retry_messages = mock_gateway.acompletion.await_args_list[1].kwargs["messages"]
        assert "PREVIOUS ATTEMPT FAILED VALIDATION" not in first_messages[1]["content"]
        assert "PREVIOUS ATTEMPT FAILED VALIDATION" in retry_messages[1]["content"]
        assert "Failed safety layers" in retry_messages[1]["content"]
        # Retry succeeded → deployed.
        assert result["status"] == "deployed"
        assert result["deployed"] is True

    @pytest.mark.asyncio
    async def test_run_cycle_retries_exhaust_on_persistent_failure(self) -> None:
        """Always-failing validation → max_retries+1 attempts, status validation_failed."""
        mock_safety = MagicMock()
        mock_safety.validate = AsyncMock(return_value={
            "passed": False, "layers": {"sec": {"passed": False}}, "reason": "always bad",
        })
        mock_gateway = _make_mock_gateway()

        engine = SelfEvolutionEngine(
            safety_pipeline=mock_safety, gateway=mock_gateway, max_retries=2,
        )
        result = await engine.run_cycle(
            execution_history=[], failure_patterns=["timeout"],
        )

        # 1 initial + 2 retries.
        assert mock_gateway.acompletion.await_count == 3
        assert result["status"] == "validation_failed"
        assert result["deployed"] is False

    @pytest.mark.asyncio
    async def test_run_cycle_max_retries_zero_single_attempt(self) -> None:
        """max_retries=0 (default) → exactly one generate call, no retry."""
        mock_safety = MagicMock()
        mock_safety.validate = AsyncMock(return_value={
            "passed": False, "layers": {}, "reason": "bad",
        })
        mock_gateway = _make_mock_gateway()
        engine = SelfEvolutionEngine(safety_pipeline=mock_safety, gateway=mock_gateway)
        result = await engine.run_cycle(
            execution_history=[], failure_patterns=["timeout"],
        )

        assert mock_gateway.acompletion.await_count == 1
        assert result["status"] == "validation_failed"

    @pytest.mark.asyncio
    async def test_run_cycle_persists_on_validation_failed(self) -> None:
        """A validation failure records a rejected mutation + closed chain."""
        mock_safety = MagicMock()
        mock_safety.validate = AsyncMock(return_value={
            "passed": False, "layers": {"sec": {"passed": False}}, "reason": "bad",
        })
        mock_persister = _make_mock_persister()
        engine = SelfEvolutionEngine(
            safety_pipeline=mock_safety, persister=mock_persister, max_retries=0,
        )
        result = await engine.run_cycle(
            execution_history=[], failure_patterns=["timeout"],
        )

        assert result["status"] == "validation_failed"
        assert result["chain_id"] is mock_persister.create_chain.return_value
        mock_persister.create_chain.assert_awaited_once()
        # Mutation recorded as rejected.
        mock_persister.record_mutation.assert_awaited_once()
        assert mock_persister.record_mutation.await_args.kwargs["status"] == "rejected"
        # Chain closed as validation_failed.
        mock_persister.complete_chain.assert_awaited_once()
        assert mock_persister.complete_chain.await_args.args[1] == "validation_failed"
        # Telemetry events recorded (generation_attempt + validation_result).
        assert mock_persister.record_event.await_count >= 2

    @pytest.mark.asyncio
    async def test_run_cycle_persists_on_deployed(self) -> None:
        """A successful deploy transitions the mutation to deployed + closes chain."""
        mock_safety = MagicMock()
        mock_safety.validate = AsyncMock(return_value={"passed": True, "layers": {}})
        mock_sandbox = _make_mock_sandbox(success=True, duration_seconds=0.1)
        mock_persister = _make_mock_persister()
        mock_gateway = _make_mock_gateway()
        engine = SelfEvolutionEngine(
            safety_pipeline=mock_safety,
            gateway=mock_gateway,
            persister=mock_persister,
        )
        result = await engine.run_cycle(
            execution_history=[], failure_patterns=["timeout"], sandbox=mock_sandbox,
        )

        assert result["status"] == "deployed"
        mock_persister.record_mutation.assert_awaited_once()
        mock_persister.update_mutation_status.assert_awaited_once()
        assert mock_persister.update_mutation_status.await_args.args[1] == "deployed"
        mock_persister.complete_chain.assert_awaited_once()
        assert mock_persister.complete_chain.await_args.args[1] == "deployed"
        # A 'deployed' telemetry event was recorded.
        event_types = [c.args[1] for c in mock_persister.record_event.await_args_list]
        assert "deployed" in event_types

    @pytest.mark.asyncio
    async def test_run_cycle_no_persister_backward_compat(self) -> None:
        """Without a persister, run_cycle still works (chain_id is None)."""
        mock_safety = MagicMock()
        mock_safety.validate = AsyncMock(return_value={
            "passed": False, "layers": {"sec": {"passed": False}}, "reason": "bad",
        })
        engine = SelfEvolutionEngine(safety_pipeline=mock_safety)
        result = await engine.run_cycle(
            execution_history=[], failure_patterns=["timeout"],
        )

        assert result["status"] == "validation_failed"
        assert result["chain_id"] is None

    # ── Post-deploy verify tests (M6) ─────────────────────────────────

    @pytest.mark.asyncio
    async def test_post_deploy_verify_no_sandbox_skips(self) -> None:
        """No sandbox → verify skipped (passed) — nothing to smoke-test."""
        engine = SelfEvolutionEngine()
        result = await engine.post_deploy_verify(
            {"mutated_content": "x = 1"}, sandbox=None
        )
        assert result["passed"] is True

    @pytest.mark.asyncio
    async def test_post_deploy_verify_non_code_skips(self) -> None:
        """Non-executable mutations skip the sandbox (PROMPT → passed)."""
        mock_sandbox = _make_mock_sandbox()
        engine = SelfEvolutionEngine()
        proposal = {"mutated_content": "{}", "mutation_type": MutationType.PROMPT}
        result = await engine.post_deploy_verify(proposal, sandbox=mock_sandbox)
        assert result["passed"] is True
        mock_sandbox.execute_code.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_post_deploy_verify_success(self) -> None:
        """A CODE mutation that re-executes cleanly passes the smoke verify."""
        mock_sandbox = _make_mock_sandbox(success=True)
        engine = SelfEvolutionEngine()
        proposal = {"mutated_content": "x = 1", "mutation_type": MutationType.CODE}
        result = await engine.post_deploy_verify(proposal, sandbox=mock_sandbox)
        assert result["passed"] is True

    @pytest.mark.asyncio
    async def test_post_deploy_verify_failure(self) -> None:
        """A CODE mutation that fails on re-execution fails the smoke verify."""
        mock_sandbox = _make_mock_sandbox(success=False, exit_code=1)
        engine = SelfEvolutionEngine()
        proposal = {"mutated_content": "bad", "mutation_type": MutationType.CODE}
        result = await engine.post_deploy_verify(proposal, sandbox=mock_sandbox)
        assert result["passed"] is False
        assert "smoke failed" in result["reason"]

    @pytest.mark.asyncio
    async def test_post_deploy_verify_exception(self) -> None:
        """A sandbox exception during verify yields passed=False with the error."""
        mock_sandbox = MagicMock()
        mock_sandbox.execute_code = AsyncMock(side_effect=RuntimeError("sandbox died"))
        engine = SelfEvolutionEngine()
        proposal = {"mutated_content": "x", "mutation_type": MutationType.CODE}
        result = await engine.post_deploy_verify(proposal, sandbox=mock_sandbox)
        assert result["passed"] is False
        assert "sandbox died" in result["reason"]

    # ── Rollback deployment tests (M6) ────────────────────────────────

    @pytest.mark.asyncio
    async def test_rollback_deployment_no_pre_deploy_hash(self) -> None:
        """No pre_deploy_hash → no-op returning rolled_back=False."""
        engine = SelfEvolutionEngine()
        mock_tracker = MagicMock()
        mock_tracker.rollback = AsyncMock(return_value=True)
        result = await engine.rollback_deployment(
            {"pre_deploy_hash": None}, mock_tracker
        )
        assert result["rolled_back"] is False
        mock_tracker.rollback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rollback_deployment_captures_diff_then_rolls_back(self) -> None:
        """The diff is captured BEFORE rollback; rollback uses the pre-deploy hash."""
        engine = SelfEvolutionEngine()
        order: list[str] = []
        mock_tracker = MagicMock()

        def _diff(*args: object, **kwargs: object) -> str:
            del args, kwargs  # AsyncMock passes the call args; order is all we track
            order.append("diff")
            return "deployed diff"

        def _rollback(*args: object, **kwargs: object) -> bool:
            del args, kwargs
            order.append("rollback")
            return True

        mock_tracker.get_diff = AsyncMock(side_effect=_diff)
        mock_tracker.rollback = AsyncMock(side_effect=_rollback)

        result = await engine.rollback_deployment(
            {"pre_deploy_hash": "abc123"}, mock_tracker
        )

        assert result["rolled_back"] is True
        assert result["reverted_diff"] == "deployed diff"
        assert result["pre_deploy_hash"] == "abc123"
        mock_tracker.rollback.assert_awaited_once_with("abc123")
        mock_tracker.get_diff.assert_awaited_once()
        assert mock_tracker.get_diff.await_args.kwargs["since_hash"] == "abc123"
        assert order == ["diff", "rollback"]

    @pytest.mark.asyncio
    async def test_rollback_deployment_on_exception_returns_false(self) -> None:
        """A rollback exception yields rolled_back=False carrying the error."""
        engine = SelfEvolutionEngine()
        mock_tracker = MagicMock()
        mock_tracker.get_diff = AsyncMock(return_value="")
        mock_tracker.rollback = AsyncMock(side_effect=RuntimeError("git blew up"))
        result = await engine.rollback_deployment(
            {"pre_deploy_hash": "abc"}, mock_tracker
        )
        assert result["rolled_back"] is False
        assert "git blew up" in result["reason"]

    @pytest.mark.asyncio
    async def test_rollback_deployment_restores_state_with_real_tracker(self) -> None:
        """End-to-end: rollback_deployment + a real GitTracker restores files."""
        import shutil
        import tempfile
        from pathlib import Path

        from src.evolution.git_tracker import GitTracker

        source = Path(tempfile.mkdtemp(prefix="m6_src_"))
        (source / "main.py").write_text("print('original')", encoding="utf-8")
        repo = Path(tempfile.mkdtemp(prefix="m6_repo_"))
        try:
            tracker = GitTracker(source, repo)
            await tracker.initialize()
            pre_deploy = await tracker.get_current_hash()

            # Simulate a deploy that writes a regression and commits it.
            await tracker.apply_mutation("main.py", "print('REGRESSED')")
            await tracker.snapshot("deploy regression")
            assert (repo / "main.py").read_text(encoding="utf-8") == "print('REGRESSED')"

            engine = SelfEvolutionEngine()
            result = await engine.rollback_deployment(
                {"pre_deploy_hash": pre_deploy}, tracker
            )

            assert result["rolled_back"] is True
            assert "REGRESSED" in result["reverted_diff"]
            assert (repo / "main.py").read_text(encoding="utf-8") == "print('original')"
        finally:
            shutil.rmtree(source, ignore_errors=True)
            shutil.rmtree(repo, ignore_errors=True)

    # ── run_cycle rollback integration tests (M6) ─────────────────────

    @pytest.mark.asyncio
    async def test_run_cycle_post_deploy_failure_triggers_rollback(self) -> None:
        """A failed post-deploy smoke verify reverts the shadow repo (B6).

        Patches generate to a CODE mutation (original_content empty so ab_test
        runs treatment-only) and sequences the sandbox so the post-deploy
        re-execution is the only failing call.
        """
        mock_safety = MagicMock()
        mock_safety.validate = AsyncMock(return_value={"passed": True, "layers": {}})

        mock_sandbox = MagicMock()
        ok = _make_sandbox_result(success=True, duration_seconds=0.1)
        fail = _make_sandbox_result(success=False, exit_code=1, duration_seconds=0.1)
        # sandbox_test (ok) → ab_test treatment-only (ok) → post_deploy_verify (fail)
        mock_sandbox.execute_code = AsyncMock(side_effect=[ok, ok, fail])

        mock_tracker = _make_mock_git_tracker()
        mock_tracker.get_diff = AsyncMock(return_value="diff of deployed change")
        mock_tracker.rollback = AsyncMock(return_value=True)

        mock_persister = _make_mock_persister()
        engine = SelfEvolutionEngine(
            safety_pipeline=mock_safety, persister=mock_persister
        )

        with patch.object(
            engine, "generate",
            new_callable=AsyncMock,
            return_value={
                "mutation_type": MutationType.CODE,
                "description": "code mutation",
                "original_content": "",
                "mutated_content": "x = 1",
                "target_path": "test.py",
                "priority": "high",
                "rationale": "testing rollback",
                "model_used": None,
                "tokens_used": 0,
            },
        ):
            result = await engine.run_cycle(
                execution_history=[],
                failure_patterns=["timeout"],
                sandbox=mock_sandbox,
                git_tracker=mock_tracker,
            )

        assert result["status"] == "rolled_back"
        assert result["deployed"] is False
        assert result["mutations_deployed"] == 0
        # Shadow repo reverted to the pre-deploy hash captured by deploy().
        mock_tracker.rollback.assert_awaited_once_with("predeploy000000")
        # get_diff is now awaited twice: once in deploy() (Phase 3a captures the
        # deployed diff) and once in rollback() (the reverted diff). The most
        # recent call is the rollback, scoped to the pre-deploy hash.
        assert mock_tracker.get_diff.await_count == 2
        assert mock_tracker.get_diff.await_args.kwargs["since_hash"] == "predeploy000000"
        assert result["rollback"]["rolled_back"] is True
        # Persistence recorded the rollback outcome.
        event_types = [c.args[1] for c in mock_persister.record_event.await_args_list]
        assert "rolled_back" in event_types
        assert mock_persister.complete_chain.await_args.args[1] == "rolled_back"
        assert mock_persister.update_mutation_status.await_args.args[1] == "rolled_back"

    @pytest.mark.asyncio
    async def test_run_cycle_verify_failed_when_no_tracker(self) -> None:
        """Smoke failure with no git_tracker → status verify_failed (can't revert)."""
        mock_safety = MagicMock()
        mock_safety.validate = AsyncMock(return_value={"passed": True, "layers": {}})

        mock_sandbox = MagicMock()
        ok = _make_sandbox_result(success=True, duration_seconds=0.1)
        fail = _make_sandbox_result(success=False, exit_code=1, duration_seconds=0.1)
        mock_sandbox.execute_code = AsyncMock(side_effect=[ok, ok, fail])

        mock_persister = _make_mock_persister()
        engine = SelfEvolutionEngine(
            safety_pipeline=mock_safety, persister=mock_persister
        )

        with patch.object(
            engine, "generate",
            new_callable=AsyncMock,
            return_value={
                "mutation_type": MutationType.CODE,
                "description": "code mutation",
                "original_content": "",
                "mutated_content": "x = 1",
                "target_path": "test.py",
                "priority": "high",
                "rationale": "testing verify_failed",
                "model_used": None,
                "tokens_used": 0,
            },
        ):
            result = await engine.run_cycle(
                execution_history=[],
                failure_patterns=["timeout"],
                sandbox=mock_sandbox,
                git_tracker=None,
            )

        assert result["status"] == "verify_failed"
        assert result["deployed"] is False
        assert result["mutations_deployed"] == 0
        assert mock_persister.complete_chain.await_args.args[1] == "verify_failed"

    @pytest.mark.asyncio
    async def test_run_cycle_smoke_pass_still_deploys_code(self) -> None:
        """A CODE mutation whose post-deploy smoke passes still deploys."""
        mock_safety = MagicMock()
        mock_safety.validate = AsyncMock(return_value={"passed": True, "layers": {}})
        mock_sandbox = _make_mock_sandbox(success=True, duration_seconds=0.1)
        mock_tracker = _make_mock_git_tracker()

        engine = SelfEvolutionEngine(safety_pipeline=mock_safety)

        with patch.object(
            engine, "generate",
            new_callable=AsyncMock,
            return_value={
                "mutation_type": MutationType.CODE,
                "description": "code mutation",
                "original_content": "",
                "mutated_content": "x = 1",
                "target_path": "test.py",
                "priority": "high",
                "rationale": "testing happy path",
                "model_used": None,
                "tokens_used": 0,
            },
        ):
            result = await engine.run_cycle(
                execution_history=[],
                failure_patterns=["timeout"],
                sandbox=mock_sandbox,
                git_tracker=mock_tracker,
            )

        assert result["status"] == "deployed"
        assert result["deployed"] is True
        assert result["smoke_result"]["passed"] is True
