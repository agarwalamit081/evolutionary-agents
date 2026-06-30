"""SI-2 — ``--score-spec`` CLI flag (``main._run_score_spec``).

Scores on-disk deliverables against a golden ``GoalSpec`` via the recomputation
engine (``run_checks``) — the same anti-fabrication ground truth ``--eval`` and
the verify node use. This is a host-side, key-free op: structural / execution /
golden probes recompute from deliverable content; the Oracle (LLM-judge) check
is skipped without a gateway.

These tests cover the CLI *wiring* (exit-code contract + output table), NOT the
recomputation engine itself (already covered in ``tests/test_eval``):
  - unknown spec id        → exit 2  (lists available ids)
  - known spec, no files   → exit 1  (probes cannot recompute → FAIL)
  - the output renders the OVERALL SCORE + per-check PASS/FAIL rows
"""

from __future__ import annotations

import click.testing

import main as main_mod
from main import _evidence_str
from src.eval.golden import GOLDEN_SPECS


def _a_spec_with_checks() -> str:
    """First spec id that actually declares checks (skips trivial-pass specs)."""
    for spec_id, spec in GOLDEN_SPECS.items():
        if spec.checks:
            return spec_id
    raise AssertionError("no GoalSpec with checks found in GOLDEN_SPECS")


class TestEvidenceStr:
    """``_evidence_str`` renders evidence compactly without secrets."""

    def test_falsy_is_empty(self) -> None:
        assert _evidence_str(None) == ""
        assert _evidence_str("") == ""
        assert _evidence_str([]) == ""
        assert _evidence_str({}) == ""

    def test_dict_is_compact_json(self) -> None:
        out = _evidence_str({"rows": 4, "ok": True})
        assert out.startswith("{") and out.endswith("}")
        assert '"ok": true' in out
        assert "rows" in out

    def test_list_is_compact_json(self) -> None:
        assert _evidence_str([1, 2, 3]) == "[1, 2, 3]"

    def test_scalar_is_str(self) -> None:
        assert _evidence_str("file missing") == "file missing"
        assert _evidence_str(42) == "42"


class TestScoreSpecCli:
    """``--score-spec`` exit-code contract + output table."""

    def test_unknown_spec_returns_2(self) -> None:
        result = click.testing.CliRunner().invoke(
            main_mod.main, ["--score-spec", "does-not-exist-xyz"]
        )
        assert result.exit_code == 2
        assert "Unknown spec id: does-not-exist-xyz" in result.output
        # Lists at least one available spec id so the operator can self-correct.
        assert "Available:" in result.output

    def test_known_spec_with_no_deliverables_fails_with_1(self) -> None:
        """No deliverable files → probes cannot recompute → FAIL → exit 1."""
        spec_id = _a_spec_with_checks()
        result = click.testing.CliRunner().invoke(
            main_mod.main, ["--score-spec", spec_id]
        )
        assert result.exit_code == 1
        assert "OVERALL SCORE" in result.output
        # A known spec with checks but no files must surface at least one FAIL.
        assert "FAIL" in result.output
        assert spec_id in result.output

    def test_unknown_spec_exits_before_scoring(self) -> None:
        """The unknown-spec branch exits before any scoring / agent run."""
        result = click.testing.CliRunner().invoke(
            main_mod.main, ["--score-spec", "nope"]
        )
        # Exit 2 (not the agent's normal 0/None path) proves the early exit.
        assert result.exit_code == 2
        assert "Available:" in result.output
        # No scoring table was rendered → it bailed at the lookup, not mid-run.
        assert "OVERALL SCORE" not in result.output
