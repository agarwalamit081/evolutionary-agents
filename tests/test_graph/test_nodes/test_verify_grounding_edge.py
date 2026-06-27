"""Verify grounding fail-safety edges through the production ``verify_node`` gate.

The pure helpers (``_check_deliverables``, ``_classify_deliverable_format``,
``_placeholder_leak_reason``, ``_spot_check_cited_paths`` O3 provenance) each
have dedicated suites (``test_verify_grounding.py``, ``test_verify_format_guard
.py``, ``test_verify_o3_provenance.py``). This file closes the spec's named
edges that are NOT covered there:

* a CITED ``results/<file>`` path that does NOT exist on disk →
  ``_spot_check_cited_paths`` emits the "Cited deliverable paths not found on
  disk" advisory (check (a) in the spec; the missing-cited-path branch is the
  one spot-check warning the O3 suite never exercises directly).
* through ``verify_node`` itself: a goal that names a deliverable which is
  MISSING / EMPTY / MALFORMED does NOT mark the run complete (re-plan);
  a well-formed goal deliverable lets a clean run complete.
* a ``.md``/``.txt`` shipped with ≥2 unsubstituted ``{var}``/``{%tag%}``
  template placeholders is MALFORMED at the gate level (the verify_node
  ``_goal_deliverables_satisfied`` returns False, blocking force-complete).
* a malformed ``.csv`` (prose dump) / ``.json`` (invalid) / ``.jsonl`` (bad
  line) deliverable is rejected at the gate level (goal NOT satisfied).
* a well-formed ``.csv``/``.json``/``.jsonl``/``.md`` deliverable is accepted.

The end-to-end drive through ``verify_node`` (no gateway → heuristic path)
proves the helpers compose into the production completion discipline, not just
that each fires in isolation. No src/ file is modified.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.graph.enums import Confidence, GoalStatus, Phase, TaskComplexity
from src.graph.factory import initial_state
from src.graph.models import Goal, PlanStep
from src.graph.nodes.verify import (
    _classify_deliverable_format,
    _goal_deliverables_satisfied,
    _spot_check_cited_paths,
    verify_node,
)


# ─── shared tmp-root installer ───────────────────────────────────────────


def _install_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the shared settings source at isolated tmp results/workspace roots
    and pin CWD there (mirrors ``test_verify_grounding._install_roots``)."""
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
    monkeypatch.chdir(tmp_path)
    return results_root


def _goal_state(goal_text: str, *, thread: str = "thread-edge") -> dict[str, Any]:
    """A state whose goal names a deliverable, all steps done, HIGH confidence.

    The goal carries a ``Goal`` object so ``_extract_goal_deliverables`` reads
    its text + success criteria; ``current_step_index`` past the plan and no
    errors so the ONLY completion blocker is the deliverable check.
    """
    state = initial_state(goal_text, thread, 10)
    goal = Goal(
        text=goal_text,
        complexity=TaskComplexity.SIMPLE,
        success_criteria=["the deliverable described in the goal is on disk"],
        status=GoalStatus.ACTIVE,
    )
    state["current_goal"] = goal
    state["plan_steps"] = [
        PlanStep(
            id="s1", description="do the work", tool_name="file_writer",
            tool_input={}, status="completed", result="done",
        )
    ]
    state["completed_steps"] = [
        PlanStep(
            id="s1", description="do the work", tool_name="file_writer",
            tool_input={}, status="completed", result="done",
        )
    ]
    state["current_step_index"] = 1  # all steps done
    state["errors"] = []
    state["confidence"] = Confidence.HIGH
    return state


# ─── _spot_check_cited_paths: a CITED path that does NOT exist ───────────


class TestSpotCheckMissingCitedPath:
    """The one spot-check warning the O3 suite leaves uncovered: a deliverable
    that cites ``results/<file>`` where <file> is NOT on disk. Advisory-only —
    it never forces incomplete, but it surfaces the fabrication signal so the
    verifier LLM can weigh it."""

    def test_missing_cited_path_emits_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_roots(monkeypatch, tmp_path)  # results/ exists but is empty
        deliverable = (
            "Wrote the reconciliation report and saved the data to "
            "results/ghost_report.csv."
        )

        out = _spot_check_cited_paths(deliverable, "")

        assert "Cited deliverable paths not found on disk" in out
        assert "results/ghost_report.csv" in out

    def test_present_cited_path_no_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        results = _install_roots(monkeypatch, tmp_path)
        (results / "real_report.csv").write_text("id,v\n1,a\n", encoding="utf-8")
        deliverable = "Saved the data to results/real_report.csv."

        assert _spot_check_cited_paths(deliverable, "") == ""

    def test_missing_cited_path_is_advisory_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The warning text references discovery, not a hard failure — the
        spot-check is defense-in-depth and never touches the completion
        verdict (asserted structurally: it returns a string, the caller only
        interpolates it into the prompt)."""
        _install_roots(monkeypatch, tmp_path)
        out = _spot_check_cited_paths("see results/ghost.csv", "")
        assert "not found" in out
        # Advisory phrasing — never a verdict.
        assert "is_complete" not in out


# ─── verify_node: missing / empty / malformed deliverable → NOT complete ──


class TestVerifyNodeRejectsDefectiveDeliverable:
    """Through the production gate: a goal whose named deliverable is missing,
    empty, or malformed must NOT mark the run complete (the safe behavior is
    to re-plan, not declare success)."""

    @pytest.mark.asyncio
    async def test_missing_goal_deliverable_not_complete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_roots(monkeypatch, tmp_path)  # report.md deliberately absent
        state = _goal_state("Write the audit report to results/report.md")

        result = await verify_node(state)  # no gateway → heuristic path

        assert result["is_complete"] is False
        assert result["phase"] == Phase.EXECUTE
        assert any("report.md" in e for e in result.get("missing_deliverables", []))

    @pytest.mark.asyncio
    async def test_empty_goal_deliverable_not_complete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        results = _install_roots(monkeypatch, tmp_path)
        (results / "report.md").write_text("", encoding="utf-8")  # 0 bytes
        state = _goal_state("Write the audit report to results/report.md")

        result = await verify_node(state)

        assert result["is_complete"] is False
        assert result["phase"] == Phase.EXECUTE

    @pytest.mark.asyncio
    async def test_malformed_csv_deliverable_not_complete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A .csv deliverable clobbered with report prose is malformed → the
        goal is not satisfied → the run does not force-complete."""
        results = _install_roots(monkeypatch, tmp_path)
        # Multi-word prose lines, no delimiter → "rows are prose, not tabular".
        (results / "out.csv").write_text(
            "VERIFICATION REPORT\nStep one completed\nNo delimiter here\n",
            encoding="utf-8",
        )
        state = _goal_state("Reconcile the data and write results/out.csv")

        # Sanity: the classifier flags it.
        assert _classify_deliverable_format(results / "out.csv") is not None
        satisfied, _paths = _goal_deliverables_satisfied(state)
        assert satisfied is False

        result = await verify_node(state)
        assert result["is_complete"] is False

    @pytest.mark.asyncio
    async def test_malformed_json_deliverable_not_complete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        results = _install_roots(monkeypatch, tmp_path)
        (results / "summary.json").write_text("{not valid json}", encoding="utf-8")
        state = _goal_state("Write the summary to results/summary.json")

        assert _classify_deliverable_format(results / "summary.json") is not None
        satisfied, _ = _goal_deliverables_satisfied(state)
        assert satisfied is False

        result = await verify_node(state)
        assert result["is_complete"] is False

    @pytest.mark.asyncio
    async def test_malformed_jsonl_deliverable_not_complete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        results = _install_roots(monkeypatch, tmp_path)
        # One valid + one invalid JSON line → "1/2 lines are not valid JSON".
        (results / "events.jsonl").write_text(
            '{"id": 1}\n{bad line}\n', encoding="utf-8"
        )
        state = _goal_state("Write the event stream to results/events.jsonl")

        assert _classify_deliverable_format(results / "events.jsonl") is not None
        satisfied, _ = _goal_deliverables_satisfied(state)
        assert satisfied is False

        result = await verify_node(state)
        assert result["is_complete"] is False


# ─── placeholder-leak .md/.txt is malformed at the gate level ────────────


class TestPlaceholderLeakMalformedAtGate:
    """A ``.md``/``.txt`` shipped with ≥2 unsubstituted ``{var}``/``{%tag%}``
    template placeholders is malformed → goal not satisfied → no force-complete."""

    @pytest.mark.asyncio
    async def test_leaky_markdown_blocks_completion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        results = _install_roots(monkeypatch, tmp_path)
        leaky = (
            "# Report\n"
            "Score: {uniqueness_pct}% across {total_rows} rows.\n"
        )
        (results / "quality_report.md").write_text(leaky, encoding="utf-8")
        state = _goal_state("Write the quality report to results/quality_report.md")

        assert _classify_deliverable_format(results / "quality_report.md") is not None
        satisfied, _ = _goal_deliverables_satisfied(state)
        assert satisfied is False

        result = await verify_node(state)
        assert result["is_complete"] is False

    @pytest.mark.asyncio
    async def test_leaky_txt_with_jinja_tag_blocks_completion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        results = _install_roots(monkeypatch, tmp_path)
        leaky = "Summary\n{% block content %}\n{{ value }}\n"
        (results / "notes.txt").write_text(leaky, encoding="utf-8")
        state = _goal_state("Write the notes to results/notes.txt")

        satisfied, _ = _goal_deliverables_satisfied(state)
        assert satisfied is False

        result = await verify_node(state)
        assert result["is_complete"] is False

    def test_single_incidental_brace_is_not_malformed(self, tmp_path: Path) -> None:
        """One stray {var} token in prose is below the ≥2 threshold → clean."""
        edge = tmp_path / "report.md"
        edge.write_text("Use the {placeholder} syntax in your template.", encoding="utf-8")
        assert _classify_deliverable_format(edge) is None


# ─── well-formed deliverable → accepted (goal satisfied, run completes) ───


class TestWellFormedDeliverableAccepted:
    @pytest.mark.asyncio
    async def test_well_formed_md_completes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        results = _install_roots(monkeypatch, tmp_path)
        (results / "report.md").write_text(
            "# Audit Report\nFully rendered. No template residue.\n",
            encoding="utf-8",
        )
        state = _goal_state("Write the audit report to results/report.md")

        satisfied, _ = _goal_deliverables_satisfied(state)
        assert satisfied is True

        result = await verify_node(state)
        # Goal satisfied + all steps done + no errors + HIGH confidence → complete.
        assert result["is_complete"] is True
        assert result["phase"] == Phase.COMPLETE

    @pytest.mark.asyncio
    async def test_well_formed_csv_completes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        results = _install_roots(monkeypatch, tmp_path)
        (results / "out.csv").write_text(
            "id,value\n1,10\n2,20\n3,30\n", encoding="utf-8"
        )
        state = _goal_state("Write the reconciled data to results/out.csv")

        satisfied, _ = _goal_deliverables_satisfied(state)
        assert satisfied is True

        result = await verify_node(state)
        assert result["is_complete"] is True

    @pytest.mark.asyncio
    async def test_well_formed_json_completes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        results = _install_roots(monkeypatch, tmp_path)
        (results / "summary.json").write_text(
            '{"overall": 98.5, "rows": 19}', encoding="utf-8"
        )
        state = _goal_state("Write the summary to results/summary.json")

        satisfied, _ = _goal_deliverables_satisfied(state)
        assert satisfied is True

        result = await verify_node(state)
        assert result["is_complete"] is True

    @pytest.mark.asyncio
    async def test_well_formed_jsonl_completes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        results = _install_roots(monkeypatch, tmp_path)
        (results / "events.jsonl").write_text(
            '{"id": 1}\n{"id": 2}\n{"id": 3}\n', encoding="utf-8"
        )
        state = _goal_state("Write the event stream to results/events.jsonl")

        satisfied, _ = _goal_deliverables_satisfied(state)
        assert satisfied is True

        result = await verify_node(state)
        assert result["is_complete"] is True

    @pytest.mark.asyncio
    async def test_single_column_csv_is_well_formed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A single-column CSV (header + one value per line) is RFC-4180-valid
        and must NOT be flagged malformed."""
        results = _install_roots(monkeypatch, tmp_path)
        (results / "primes.csv").write_text(
            "prime\n2\n3\n5\n7\n11\n", encoding="utf-8"
        )
        state = _goal_state("Write the primes to results/primes.csv")

        satisfied, _ = _goal_deliverables_satisfied(state)
        assert satisfied is True
