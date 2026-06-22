"""Regression tests for the verify-node deliverable format-validity guard.

A present, non-empty deliverable that does not parse as its declared format —
e.g. an agent that overwrites ``results/q01/normalized.csv`` with a
verification report — must be flagged ``malformed`` by ``_check_deliverables``
and must force ``is_complete=False``, exactly like a missing or empty
deliverable. This is the deterministic backstop for the battery-04 q1
clobbering failure: the file passed the existence check (present + non-empty)
but was not valid CSV, so verify rubber-stamped it complete.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.graph.enums import Confidence
from src.graph.factory import initial_state
from src.graph.models import PlanStep
from src.graph.nodes.verify import (
    _check_deliverables,
    _classify_deliverable_format,
    verify_node,
)


def _install_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the shared settings source at isolated tmp results/workspace roots.

    Mirrors the harness in ``test_verify_grounding.py``: ``_resolve_deliverable``
    resolves candidates via ``src.tools._paths`` which reads
    ``src.config.settings.get_settings``, so that is what we patch.
    """
    results_root = tmp_path / "results"
    workspace_root = tmp_path / "workspace"
    results_root.mkdir(parents=True, exist_ok=True)
    workspace_root.mkdir(parents=True, exist_ok=True)
    fake_settings = SimpleNamespace(
        agent=SimpleNamespace(
            results_root=str(results_root), workspace_root=str(workspace_root)
        )
    )
    monkeypatch.setattr("src.config.settings.get_settings", lambda: fake_settings)
    return results_root


# ─── _classify_deliverable_format (pure logic) ───────────────────────


class TestClassifyDeliverableFormat:
    """The format classifier is the core of the guard."""

    def test_csv_report_text_is_malformed(self, tmp_path: Path) -> None:
        """The q1 regression: report prose (no comma in first row) in a .csv."""
        p = tmp_path / "normalized.csv"
        p.write_text(
            "VERIFICATION REPORT - E-Commerce Event Normalization\n"
            "=====================================================\n\n"
            "STEP 10/11: File Existence and Content Verification\n",
            encoding="utf-8",
        )
        reason = _classify_deliverable_format(p)
        assert reason is not None
        assert "not tabular CSV" in reason

    def test_csv_single_prose_line_is_malformed(self, tmp_path: Path) -> None:
        p = tmp_path / "out.csv"
        p.write_text("just one prose line with no delimiter", encoding="utf-8")
        assert _classify_deliverable_format(p) is not None

    def test_csv_single_column_data_passes(self, tmp_path: Path) -> None:
        """A single-column CSV (header + one value per row) is RFC-4180 valid.

        Regression (covbench evo-demo): primes_demo.csv = header 'prime' +
        15 primes was wrongly rejected as 'not tabular CSV' (the old check
        required >=2 fields on the first row), looping verify->replan forever.
        """
        p = tmp_path / "primes_demo.csv"
        p.write_text(
            "prime\n2\n3\n5\n7\n11\n13\n17\n19\n23\n29\n31\n37\n41\n43\n47\n",
            encoding="utf-8",
        )
        assert _classify_deliverable_format(p) is None

    def test_csv_single_value_passes(self, tmp_path: Path) -> None:
        """A two-row single-column CSV (header + one value) is valid."""
        p = tmp_path / "count.csv"
        p.write_text("count\n42\n", encoding="utf-8")
        assert _classify_deliverable_format(p) is None

    def test_valid_multicolumn_csv_passes(self, tmp_path: Path) -> None:
        p = tmp_path / "normalized.csv"
        p.write_text(
            "event_id,event_type,amount\n1,order,99.99\n2,return,75.0\n",
            encoding="utf-8",
        )
        assert _classify_deliverable_format(p) is None

    def test_valid_json_passes(self, tmp_path: Path) -> None:
        p = tmp_path / "summary.json"
        p.write_text('{"rows": 8, "ok": true}', encoding="utf-8")
        assert _classify_deliverable_format(p) is None

    def test_invalid_json_is_malformed(self, tmp_path: Path) -> None:
        p = tmp_path / "summary.json"
        p.write_text("not json at all {{{ no closing", encoding="utf-8")
        reason = _classify_deliverable_format(p)
        assert reason is not None
        assert "invalid JSON" in reason

    def test_valid_jsonl_passes(self, tmp_path: Path) -> None:
        p = tmp_path / "events.jsonl"
        p.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
        assert _classify_deliverable_format(p) is None

    def test_jsonl_with_bad_line_is_malformed(self, tmp_path: Path) -> None:
        p = tmp_path / "events.jsonl"
        p.write_text('{"a": 1}\nnot json\n{"a": 3}\n', encoding="utf-8")
        reason = _classify_deliverable_format(p)
        assert reason is not None
        assert "not valid JSON" in reason

    def test_markdown_and_python_are_not_checked(self, tmp_path: Path) -> None:
        """Free-form text and code artifacts are skipped (no false positives)."""
        md = tmp_path / "report.md"
        md.write_text("# Report\nno structure at all", encoding="utf-8")
        assert _classify_deliverable_format(md) is None
        py = tmp_path / "tool.py"
        py.write_text("def f():\n    return 1\n", encoding="utf-8")
        assert _classify_deliverable_format(py) is None

    def test_empty_file_is_not_format_flagged(self, tmp_path: Path) -> None:
        """0-byte files are the size check's job, not the format check's."""
        p = tmp_path / "empty.csv"
        p.write_text("", encoding="utf-8")
        assert _classify_deliverable_format(p) is None


# ─── _check_deliverables integration (4-tuple) ───────────────────────


class TestCheckDeliverablesMalformed:
    def test_clobbered_csv_lands_in_malformed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        results_root = _install_roots(monkeypatch, tmp_path)
        (results_root / "q01").mkdir(parents=True, exist_ok=True)
        (results_root / "q01" / "normalized.csv").write_text(
            "VERIFICATION REPORT - E-Commerce\nSTEP 10 verification\n",
            encoding="utf-8",
        )

        evidence, missing, empty, malformed = _check_deliverables(
            ["results/q01/normalized.csv"]
        )

        assert missing == []
        assert empty == []
        assert len(malformed) == 1
        assert "normalized.csv" in malformed[0]
        assert "MALFORMED" in evidence

    def test_valid_csv_stays_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        results_root = _install_roots(monkeypatch, tmp_path)
        (results_root / "q01").mkdir(parents=True, exist_ok=True)
        (results_root / "q01" / "normalized.csv").write_text(
            "event_id,event_type\n1,order\n", encoding="utf-8"
        )

        evidence, missing, empty, malformed = _check_deliverables(
            ["results/q01/normalized.csv"]
        )

        assert missing == []
        assert empty == []
        assert malformed == []
        assert "Present:" in evidence

    def test_malformed_blocks_present_classification(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A malformed file must NOT also appear in the Present list."""
        results_root = _install_roots(monkeypatch, tmp_path)
        (results_root / "summary.json").write_text(
            "VERIFICATION REPORT not json", encoding="utf-8"
        )

        evidence, _missing, _empty, malformed = _check_deliverables(
            ["summary.json"]
        )

        assert len(malformed) == 1
        assert "Present:" not in evidence


# ─── verify_node integration (heuristic path, gateway=None) ──────────


def _state_with_csv_deliverable(file_path: str) -> dict:
    """Build a minimal otherwise-complete state declaring one CSV deliverable.

    Everything is set to "complete" (all steps done, no errors, high
    confidence) so the ONLY thing that can flip ``is_complete`` is the
    deliverable's format validity — isolating the guard.
    """
    state = initial_state("Normalize the events and save the output", "thread-q1")
    step = PlanStep(
        id="fw1",
        description=f"Write {file_path}",
        tool_name="file_writer",
        tool_input={"file_path": file_path, "content": "payload"},
        status="completed",
        result="wrote file",
    )
    state["plan_steps"] = [step]
    state["completed_steps"] = [step]
    state["current_step_index"] = 1
    state["errors"] = []
    state["confidence"] = Confidence.HIGH
    return state


class TestVerifyNodeFormatGuard:
    async def test_clobbered_csv_forces_incomplete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The q1 regression: a clobbered CSV keeps the run incomplete."""
        results_root = _install_roots(monkeypatch, tmp_path)
        (results_root / "q01").mkdir(parents=True, exist_ok=True)
        (results_root / "q01" / "normalized.csv").write_text(
            "VERIFICATION REPORT\nno commas here\njust prose lines\n",
            encoding="utf-8",
        )
        state = _state_with_csv_deliverable("results/q01/normalized.csv")

        result = await verify_node(state)  # gateway=None → heuristic path

        assert result["is_complete"] is False
        assert any("normalized.csv" in e for e in result.get("errors", []))

    async def test_valid_csv_can_complete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sanity: a valid CSV does not trip the format guard."""
        results_root = _install_roots(monkeypatch, tmp_path)
        (results_root / "q01").mkdir(parents=True, exist_ok=True)
        (results_root / "q01" / "normalized.csv").write_text(
            "event_id,event_type\n1,order\n", encoding="utf-8"
        )
        state = _state_with_csv_deliverable("results/q01/normalized.csv")

        result = await verify_node(state)

        assert result["is_complete"] is True
