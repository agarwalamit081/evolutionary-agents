"""Tests for src.evolution.report — human-readable evolution reports."""

from __future__ import annotations

from src.evolution.report import generate_report
from src.graph.enums import MutationType


def _deployed_cycle_result() -> dict:
    """Create a sample cycle result dict for a successful deployment."""
    return {
        "status": "deployed",
        "deployed": True,
        "proposal": {
            "mutation_type": MutationType.PROMPT,
            "description": "Address recurring failure patterns via prompt refinement",
            "target_path": "evolution/prompt_improvements.json",
            "rationale": "Added 2 prompt improvements addressing: json, timeout",
            "priority": "high",
            "patterns": ["invalid JSON from LLM", "timeout on tool call"],
            "model_used": None,
        },
        "validation": {
            "passed": True,
            "safety_result": {
                "layers": {
                    "syntax": {"passed": True},
                    "static_analysis": {"passed": True},
                    "security": {"passed": True},
                    "imports": {"passed": True},
                    "behavioral": {"passed": True},
                    "sandbox": {"passed": True},
                    "semantic": {"passed": True},
                },
            },
        },
        "sandbox_result": {
            "passed": True,
            "note": "non-code mutation (prompt), sandbox skipped",
        },
        "ab_result": {
            "is_significant": True,
            "note": "non-code mutation (prompt), A/B test skipped",
        },
        "deployment": {
            "deployed": True,
            "generation": 1,
            "commit_hash": "abcdef1234567890",
            "target_path": "evolution/prompt_improvements.json",
        },
        "mutations_proposed": 1,
        "mutations_deployed": 1,
    }


def _rejected_cycle_result() -> dict:
    """Create a sample cycle result for a rejected mutation."""
    return {
        "status": "validation_failed",
        "deployed": False,
        "proposal": {
            "mutation_type": MutationType.CODE,
            "description": "Optimize execute node",
            "target_path": "evolution/code_analysis.json",
            "rationale": "Code improvement analysis",
            "priority": "medium",
        },
        "validation": {
            "passed": False,
            "reason": "Failed safety layers: security",
        },
        "sandbox_result": {},
        "ab_result": {},
        "deployment": {},
        "mutations_proposed": 1,
        "mutations_deployed": 0,
    }


class TestGenerateReport:
    """Tests for the generate_report function."""

    def test_report_contains_header(self) -> None:
        """Report includes the header with generation and trigger."""
        result = generate_report(_deployed_cycle_result(), generation=1)
        assert "EVOLUTION REPORT" in result
        assert "Generation: 1" in result
        assert "reflection_recommended" in result

    def test_report_shows_opportunity(self) -> None:
        """Report includes the opportunity description."""
        result = generate_report(_deployed_cycle_result(), generation=1)
        assert "OPPORTUNITY" in result
        assert "prompt" in result.lower()
        assert "Address recurring" in result

    def test_report_shows_mutation_details(self) -> None:
        """Report includes mutation generation details."""
        result = generate_report(_deployed_cycle_result(), generation=1)
        assert "MUTATION GENERATED" in result
        assert "heuristic" in result
        assert "evolution/prompt_improvements.json" in result

    def test_report_shows_validation_passed(self) -> None:
        """Report shows validation passed for successful mutations."""
        result = generate_report(_deployed_cycle_result(), generation=1)
        assert "VALIDATION" in result
        assert "Passed" in result

    def test_report_shows_validation_failed(self) -> None:
        """Report shows validation failure reason."""
        result = generate_report(_rejected_cycle_result(), generation=1)
        assert "VALIDATION" in result
        assert "Failed" in result
        assert "security" in result

    def test_report_shows_sandbox_skipped(self) -> None:
        """Report shows sandbox skip note for non-code mutations."""
        result = generate_report(_deployed_cycle_result(), generation=1)
        assert "SANDBOX" in result
        assert "sandbox skipped" in result

    def test_report_shows_ab_test_skipped(self) -> None:
        """Report shows A/B test skip note."""
        result = generate_report(_deployed_cycle_result(), generation=1)
        assert "A/B TEST" in result
        assert "skipped" in result

    def test_report_shows_deployment_success(self) -> None:
        """Report shows successful deployment with commit hash."""
        result = generate_report(_deployed_cycle_result(), generation=1)
        assert "DEPLOYMENT" in result
        assert "Deployed" in result
        assert "abcdef12" in result

    def test_report_shows_deployment_rejection(self) -> None:
        """Report shows rejection reason."""
        result = generate_report(_rejected_cycle_result(), generation=1)
        assert "DEPLOYMENT" in result
        assert "Rejected" in result

    def test_report_shows_future_run_effect(self) -> None:
        """Report describes effect on future runs for deployed mutations."""
        result = generate_report(_deployed_cycle_result(), generation=1)
        assert "EFFECT ON FUTURE RUNS" in result
        assert "warm memory" in result

    def test_report_shows_no_effect_for_rejected(self) -> None:
        """Report shows no effect for rejected mutations."""
        result = generate_report(_rejected_cycle_result(), generation=1)
        assert "EFFECT ON FUTURE RUNS" in result
        assert "not deployed" in result.lower() or "None" in result

    def test_report_contains_timestamp(self) -> None:
        """Report includes a UTC timestamp."""
        result = generate_report(_deployed_cycle_result(), generation=1)
        assert "UTC" in result

    def test_report_is_multiline(self) -> None:
        """Report produces a multi-line string."""
        result = generate_report(_deployed_cycle_result(), generation=1)
        assert len(result.splitlines()) > 10

    def test_report_with_sandbox_execution_details(self) -> None:
        """Report shows execution time when sandbox was run."""
        cycle = _deployed_cycle_result()
        cycle["sandbox_result"] = {
            "passed": True,
            "sandbox_result": {
                "success": True,
                "exit_code": 0,
                "stdout": "ok",
                "stderr": "",
                "duration_seconds": 0.42,
                "timed_out": False,
            },
        }
        result = generate_report(cycle, generation=1)
        assert "0.42s" in result

    def test_report_with_ab_test_comparison(self) -> None:
        """Report shows control vs treatment comparison."""
        cycle = _deployed_cycle_result()
        cycle["ab_result"] = {
            "is_significant": True,
            "control_result": {"success": True, "duration_seconds": 0.3},
            "treatment_result": {"success": True, "duration_seconds": 0.15},
        }
        result = generate_report(cycle, generation=1)
        assert "control" in result
        assert "treatment" in result
