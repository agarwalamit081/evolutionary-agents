"""Unit tests for the deliverable-grounding functions in verify.py.

These cover the pure filesystem-logic helpers that independently verify the
agent's declared deliverables exist on disk:

* ``_extract_deliverable_paths(state)`` — pulls declared deliverable path
  strings out of state (authoritative ``file_writer`` inputs + phrasal cues).
* ``_check_deliverables(paths)`` — inspects the filesystem relative to the
  shared results/workspace roots and returns ``(evidence_text, missing,
  empty)``.

These functions previously had ZERO direct tests — ``test_verify.py`` exercises
them only indirectly through ``verify_node``. This file closes that gap with
deterministic, hermetic tests (no LLM, ``tmp_path``-isolated roots).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.graph.enums import GoalStatus, Phase
from src.graph.factory import initial_state
from src.graph.models import PlanStep
from src.graph.nodes.verify import (
    _check_deliverables,
    _classify_deliverable_format,
    _enforce_deliverables,
    _extract_deliverable_paths,
    _extract_goal_deliverables,
    _force_complete_on_evidence,
    _goal_deliverables_satisfied,
    _placeholder_leak_reason,
)


def _install_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    """Point the shared settings source at isolated tmp results/workspace roots.

    ``verify._resolve_deliverable`` resolves candidates via the shared path
    resolver (``src.tools._paths``), which reads ``src.config.settings.get_settings``
    — the single source — so that is what we patch (not a per-module symbol).
    Routing both roots at a tmp dir keeps these tests hermetic and matches the
    resolver's ``project_root = results_root.parent`` invariant.
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
    # ``_resolve_deliverable`` appends CWD-relative candidate paths as a last
    # resort. Pin CWD to tmp so those candidates resolve under the isolated root
    # too (otherwise a leftover real-project results/q02/scorecard.json from a
    # prior run is found and a deliberately-omitted deliverable reads as present).
    monkeypatch.chdir(tmp_path)
    return str(results_root)


def _state_with_file_writer(file_path: str) -> dict:
    """Build a minimal state declaring a single ``file_writer`` deliverable."""
    state = initial_state(f"Save the deliverable to {file_path}", "thread-ground")
    step = PlanStep(
        id="fw1",
        description=f"Write deliverable to {file_path}",
        tool_name="file_writer",
        tool_input={"file_path": file_path, "content": "payload"},
        status="completed",
        result="wrote file",
    )
    state["plan_steps"] = [step]
    state["completed_steps"] = [step]
    return state


class TestCheckDeliverables:
    """``_check_deliverables`` partitions declared paths into present/missing/empty."""

    def test_declared_deliverable_present_nonempty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A declared file that exists and is non-empty → neither missing nor empty."""
        results_root = _install_roots(monkeypatch, tmp_path)
        Path(results_root, "report.md").write_text("the answer", encoding="utf-8")

        evidence_text, missing, empty, _malformed = _check_deliverables(["report.md"])

        assert missing == []
        assert empty == []
        assert "Present:" in evidence_text
        assert "report.md" in evidence_text

    def test_declared_deliverable_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A declared file absent from disk → listed in ``missing``, not ``empty``."""
        _install_roots(monkeypatch, tmp_path)

        evidence_text, missing, empty, _malformed = _check_deliverables(["ghost_report.md"])

        assert missing == ["ghost_report.md"]
        assert empty == []
        assert "MISSING:" in evidence_text
        assert "ghost_report.md" in evidence_text

    def test_declared_deliverable_exists_but_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A declared file with 0 bytes → listed in ``empty``, not ``missing``."""
        results_root = _install_roots(monkeypatch, tmp_path)
        Path(results_root, "empty.md").write_text("", encoding="utf-8")

        evidence_text, missing, empty, _malformed = _check_deliverables(["empty.md"])

        assert missing == []
        assert len(empty) == 1
        assert empty[0].startswith("empty.md")
        assert "EMPTY/INCOMPLETE:" in evidence_text
        assert "empty.md" in evidence_text

    def test_no_deliverables_declared(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty path list → no crash, both lists empty, default evidence text."""
        _install_roots(monkeypatch, tmp_path)

        evidence_text, missing, empty, _malformed = _check_deliverables([])

        assert missing == []
        assert empty == []
        assert evidence_text == "No concrete deliverable paths were declared or detected."

    def test_mixed_present_missing_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Three declared deliverables (present, missing, empty) partition correctly."""
        results_root = _install_roots(monkeypatch, tmp_path)
        Path(results_root, "present.md").write_text("payload", encoding="utf-8")
        Path(results_root, "blank.md").write_text("", encoding="utf-8")

        evidence_text, missing, empty, _malformed = _check_deliverables(
            ["present.md", "absent.md", "blank.md"]
        )

        assert missing == ["absent.md"]
        assert len(empty) == 1
        assert empty[0].startswith("blank.md")
        # The present one appears in evidence but in neither problem list.
        assert not any("present.md" in m for m in missing)
        assert not any("present.md" in e for e in empty)
        assert "Present:" in evidence_text
        assert "MISSING:" in evidence_text
        assert "EMPTY/INCOMPLETE:" in evidence_text

    def test_multiple_deliverables_count_match(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """N declared deliverables, all present + non-empty → no false missing/empty."""
        results_root = _install_roots(monkeypatch, tmp_path)
        paths = []
        for i in range(5):
            name = f"out_{i}.md"
            Path(results_root, name).write_text(f"body {i}", encoding="utf-8")
            paths.append(name)

        _evidence_text, missing, empty, _malformed = _check_deliverables(paths)

        assert missing == []
        assert empty == []

    def test_results_prefixed_path_resolves(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A path declared with a redundant ``results/`` prefix de-nests correctly."""
        results_root = _install_roots(monkeypatch, tmp_path)
        # file_writer lands at results_root/nested.md; declaring it as
        # "results/nested.md" must still resolve (double-nest is stripped).
        Path(results_root, "nested.md").write_text("x", encoding="utf-8")

        _evidence_text, missing, empty, _malformed = _check_deliverables(["results/nested.md"])

        assert missing == []
        assert empty == []


class TestExtractDeliverablePaths:
    """``_extract_deliverable_paths`` pulls the right path strings out of state."""

    def test_picks_up_file_writer_paths(self) -> None:
        """Authoritative ``file_writer`` ``tool_input.file_path`` is extracted."""
        state = _state_with_file_writer("results/report.md")
        assert _extract_deliverable_paths(state) == ["results/report.md"]

    def test_picks_up_save_to_cue_from_goal(self) -> None:
        """A phrasal ``save ... to <path>`` cue in the goal text is extracted."""
        state = initial_state(
            "Analyze the data and export the summary to results/summary.md",
            "thread-cue",
        )
        state["plan_steps"] = []
        state["completed_steps"] = []
        paths = _extract_deliverable_paths(state)
        assert "results/summary.md" in paths

    def test_no_deliverables_declared_returns_empty(self) -> None:
        """A state with no deliverable claims → empty list."""
        state = initial_state("Think about the problem carefully.", "thread-none")
        state["plan_steps"] = [
            PlanStep(
                id="s1",
                description="Reason about the question",
                tool_name=None,
                tool_input={},
                status="pending",
                result="",
            )
        ]
        state["completed_steps"] = []
        assert _extract_deliverable_paths(state) == []

    def test_extract_then_check_roundtrip_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end: extract from state, then check on disk (present path)."""
        results_root = _install_roots(monkeypatch, tmp_path)
        Path(results_root, "out.md").write_text("payload", encoding="utf-8")
        state = _state_with_file_writer("out.md")

        paths = _extract_deliverable_paths(state)
        _evidence_text, missing, empty, _malformed = _check_deliverables(paths)

        assert paths == ["out.md"]
        assert missing == []
        assert empty == []

    def test_extract_then_check_roundtrip_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end: extract from state, then check on disk (missing path)."""
        _install_roots(monkeypatch, tmp_path)
        state = _state_with_file_writer("results/never_written.md")

        paths = _extract_deliverable_paths(state)
        _evidence_text, missing, empty, _malformed = _check_deliverables(paths)

        assert paths == ["results/never_written.md"]
        assert missing == ["results/never_written.md"]
        assert empty == []

    def test_dotfile_placeholder_write_is_excluded(self) -> None:
        """A ``file_writer`` write of a VCS placeholder (.gitkeep) is NOT a
        deliverable. battery-04 q2: the agent wrote results/q02/.gitkeep (0
        bytes) to create the output dir; treating it as a declared deliverable
        flagged it empty and looped verify→plan until MAX_ITERATIONS despite the
        real deliverables being present + eval scoring 1.0. Both the
        authoritative file_writer write AND any phrasal cue for the dotfile are
        rejected by the dotfile-basename gate.
        """
        # Goal text deliberately also names the dotfile so the _SAVE_TO_RE cue
        # path is exercised alongside the file_writer path — both must drop it.
        state = _state_with_file_writer("results/q02/.gitkeep")
        assert _extract_deliverable_paths(state) == []

    def test_dotfile_alongside_real_deliverable_roundtrip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A .gitkeep placeholder + a real deliverable on disk: only the real
        file is declared, and the disk check shows no empty/missing output.
        Regression for the q2 verify infinite-loop.
        """
        results_root = _install_roots(monkeypatch, tmp_path)
        Path(results_root, "q02").mkdir()
        Path(results_root, "q02", ".gitkeep").write_text("", encoding="utf-8")
        Path(results_root, "q02", "scorecard.json").write_text(
            '{"overall": 92.1}', encoding="utf-8"
        )
        placeholder_step = PlanStep(
            id="gk",
            description="Create results/q02 dir",
            tool_name="file_writer",
            tool_input={"file_path": "results/q02/.gitkeep", "content": ""},
            status="completed",
            result="wrote placeholder",
        )
        real_step = PlanStep(
            id="sc",
            description="Write scorecard to results/q02/scorecard.json",
            tool_name="file_writer",
            tool_input={
                "file_path": "results/q02/scorecard.json",
                "content": '{"overall": 92.1}',
            },
            status="completed",
            result="wrote scorecard",
        )
        state = initial_state(
            "Write the scorecard to results/q02/scorecard.json", "thread-dotfile"
        )
        state["plan_steps"] = [placeholder_step, real_step]
        state["completed_steps"] = [placeholder_step, real_step]

        paths = _extract_deliverable_paths(state)
        assert paths == ["results/q02/scorecard.json"]  # .gitkeep excluded
        _evidence_text, missing, empty, _malformed = _check_deliverables(paths)
        assert missing == []
        assert empty == []


class TestGoalDeliverableSufficiencyFH:
    """F-h: when the GOAL's named deliverables are present + non-empty, plan-
    declared INTERMEDIATE artifacts (a script the agent intended to write, an
    input file) are advisory, not blockers. verify checks presence; the Phase-3
    eval_enforce layer checks correctness. Regression for the q2 verify→plan
    loop that ran to MAX_ITERATIONS despite the goal deliverables being present.
    """

    _GOAL = (
        "Audit results/q01/normalized.csv, write the report to "
        "results/q02/quality_report.md, and save the scorecard to "
        "results/q02/scorecard.json."
    )

    def _state(self, goal_text: str) -> dict:
        state = initial_state(goal_text, "thread-fh")
        # _force_complete_on_evidence reads plan completion + errors from state.
        state["plan_steps"] = [
            PlanStep(
                id="s1",
                description="step 1",
                tool_name=None,
                tool_input={},
                status=GoalStatus.COMPLETED,
                result="",
            ),
            PlanStep(
                id="s2",
                description="step 2",
                tool_name=None,
                tool_input={},
                status=GoalStatus.COMPLETED,
                result="",
            ),
        ]
        state["current_step_index"] = 2  # all executed
        state["errors"] = []
        return state

    def test_goal_satisfied_when_both_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        results_root = _install_roots(monkeypatch, tmp_path)
        Path(results_root, "q02").mkdir(parents=True)
        Path(results_root, "q02", "quality_report.md").write_text("# report", encoding="utf-8")
        Path(results_root, "q02", "scorecard.json").write_text('{"x": 1}', encoding="utf-8")

        satisfied, goal_paths = _goal_deliverables_satisfied(self._state(self._GOAL))

        assert satisfied is True
        joined = " ".join(goal_paths)
        assert "quality_report.md" in joined
        assert "scorecard.json" in joined

    def test_goal_not_satisfied_when_one_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        results_root = _install_roots(monkeypatch, tmp_path)
        Path(results_root, "q02").mkdir(parents=True)
        Path(results_root, "q02", "quality_report.md").write_text("# report", encoding="utf-8")
        # scorecard.json deliberately NOT written.

        satisfied, goal_paths = _goal_deliverables_satisfied(self._state(self._GOAL))

        assert satisfied is False
        assert len(goal_paths) == 2

    def test_goal_false_when_goal_names_no_deliverable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_roots(monkeypatch, tmp_path)

        satisfied, goal_paths = _goal_deliverables_satisfied(
            self._state("Think carefully about the data quality.")
        )

        assert satisfied is False
        assert goal_paths == []

    def test_enforce_preserves_verdict_when_goal_satisfied(self) -> None:
        """Goal satisfied; only an INTERMEDIATE artifact is missing → must NOT
        force incomplete or append blocking errors."""
        result = {"is_complete": True, "phase": Phase.COMPLETE, "final_output": "ok"}

        out = _enforce_deliverables(
            result,
            deliverable_paths=["audit_quality.py", "results/q02/quality_report.md"],
            deliverable_problems=["audit_quality.py"],
            goal_satisfied=True,
        )

        assert out["is_complete"] is True  # preserved, not forced False
        assert "errors" not in out  # advisory gaps never become blocking errors

    def test_enforce_forces_incomplete_when_goal_not_satisfied(self) -> None:
        """A missing GOAL deliverable still hard-enforces incomplete."""
        result = {"is_complete": True, "phase": Phase.COMPLETE, "final_output": "ok"}

        out = _enforce_deliverables(
            result,
            deliverable_paths=["results/q02/quality_report.md"],
            deliverable_problems=["results/q02/quality_report.md"],
            goal_satisfied=False,
        )

        assert out["is_complete"] is False  # prior hard-enforce preserved
        assert "errors" in out

    def test_force_complete_when_goal_satisfied_and_intermediate_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The q2 scenario: goal deliverables present, planned intermediate
        script (audit_quality.py) absent, pessimistic LLM → force COMPLETE."""
        results_root = _install_roots(monkeypatch, tmp_path)
        Path(results_root, "q02").mkdir(parents=True)
        Path(results_root, "q02", "quality_report.md").write_text("# report", encoding="utf-8")
        Path(results_root, "q02", "scorecard.json").write_text('{"x": 1}', encoding="utf-8")
        result = {"is_complete": False, "phase": Phase.EXECUTE, "final_output": ""}

        out = _force_complete_on_evidence(
            result,
            self._state(self._GOAL),
            deliverable_paths=[
                "audit_quality.py",
                "results/q02/quality_report.md",
                "results/q02/scorecard.json",
            ],
            deliverable_problems=["audit_quality.py"],  # intermediate missing
            goal_satisfied=True,
        )

        assert out["is_complete"] is True  # forced complete despite intermediate gap
        assert out["phase"] == Phase.COMPLETE

    def test_force_complete_declined_when_goal_not_satisfied(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Goal deliverable missing + not satisfied → must NOT force-complete
        (prior behavior preserved — no rubber-stamping an unfinished run)."""
        _install_roots(monkeypatch, tmp_path)  # goal deliverables NOT on disk
        result = {"is_complete": False, "phase": Phase.EXECUTE, "final_output": ""}

        out = _force_complete_on_evidence(
            result,
            self._state(self._GOAL),
            deliverable_paths=["results/q02/quality_report.md"],
            deliverable_problems=["results/q02/quality_report.md"],
            goal_satisfied=False,
        )

        assert out["is_complete"] is False

    def test_force_complete_when_goal_satisfied_despite_stale_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """F-h.2: stale ``state.errors`` from EARLIER verify cycles must not
        block force-completion once the GOAL's deliverables are present.

        ``state.errors`` accumulates via operator.add and is never cleared, so it
        still holds "verify: deliverable not present" entries recorded before the
        agent wrote the deliverable. Without the goal-satisfied bypass of the
        errors guard, an objectively-complete goal loops verify→replan until the
        iteration hard-cap (battery-04 q2 — the 3rd verify saw both deliverables
        on disk yet re-planned because of a stale scorecard-absence error). The
        goal path requires only that all steps executed; eval_enforce checks
        correctness separately.
        """
        results_root = _install_roots(monkeypatch, tmp_path)
        Path(results_root, "q02").mkdir(parents=True)
        Path(results_root, "q02", "quality_report.md").write_text("# report", encoding="utf-8")
        Path(results_root, "q02", "scorecard.json").write_text('{"x": 1}', encoding="utf-8")
        state = self._state(self._GOAL)
        # Stale error left over from the earlier verify when the file was missing.
        state["errors"] = ["verify: deliverable not present — results/q02/scorecard.json"]
        result = {"is_complete": False, "phase": Phase.EXECUTE, "final_output": ""}

        out = _force_complete_on_evidence(
            result,
            state,
            deliverable_paths=[
                "results/q02/quality_report.md",
                "results/q02/scorecard.json",
            ],
            deliverable_problems=[],  # both present now
            goal_satisfied=True,
        )

        assert out["is_complete"] is True
        assert out["phase"] == Phase.COMPLETE

    def test_force_complete_blocked_by_errors_when_goal_not_satisfied(self) -> None:
        """Non-goal path unchanged: errors still block the optimistic clamp, so a
        genuinely-incomplete run (no concrete goal deliverable to anchor on) is
        not rubber-stamped complete just because steps executed."""
        result = {"is_complete": False, "phase": Phase.EXECUTE, "final_output": ""}
        state = initial_state("Reason about the dataset quality.", "thread-err")
        state["plan_steps"] = []
        state["errors"] = ["a real runtime error"]

        out = _force_complete_on_evidence(
            result,
            state,
            deliverable_paths=[],
            deliverable_problems=[],
            goal_satisfied=False,
        )

        assert out["is_complete"] is False


class TestGoalDeliverableExtractionFI:
    """F-i: ``_extract_goal_deliverables`` must handle inflected save-verbs
    (``writes``/``saving``/``stored``/``outputs``/``generates`` — ``\\bwrite\\b``
    has no word boundary inside the word and matched NONE of them) AND deliverable
    LISTS joined under one verb (``write X to a.md and b.json``). Before the fix
    the q2 goal ("writes ... to quality_report.md and scorecard.json") extracted
    ``[]`` → ``goal_satisfied`` was always False → F-h/F-h.2 never fired → the
    run looped to MAX_ITERATIONS despite both deliverables being on disk.
    """

    def test_inflected_writes_verb_extracts(self) -> None:
        """Third-person ``writes`` (inflected) → deliverable extracted."""
        state = initial_state(
            "The agent writes the report to results/out/report.md", "thread-fi1"
        )
        paths = _extract_goal_deliverables(state)
        assert "results/out/report.md" in paths

    def test_and_joined_deliverable_list(self) -> None:
        """Two deliverables under one verb, joined by ``and`` → both extracted."""
        state = initial_state(
            "writes a reproducible-methodology Markdown report to "
            "results/q02/quality_report.md and results/q02/scorecard.json",
            "thread-fi2",
        )
        paths = _extract_goal_deliverables(state)
        assert "results/q02/quality_report.md" in paths
        assert "results/q02/scorecard.json" in paths

    def test_comma_and_joined_deliverable_list(self) -> None:
        """Comma-separated list with an Oxford ``and`` → all three extracted."""
        state = initial_state(
            "Save the outputs to results/a.md, results/b.json, and results/c.csv",
            "thread-fi3",
        )
        paths = _extract_goal_deliverables(state)
        assert {"results/a.md", "results/b.json", "results/c.csv"} <= set(paths)

    def test_inflected_outputs_and_saving(self) -> None:
        """``outputs`` and ``saving`` (other inflections) also extract."""
        state = initial_state(
            "The pipeline outputs the manifest to results/m.json and saves the log "
            "to results/l.txt",
            "thread-fi4",
        )
        paths = _extract_goal_deliverables(state)
        assert "results/m.json" in paths
        assert "results/l.txt" in paths

    def test_base_form_still_works(self) -> None:
        """Regression: the original base-form ``write``/``save`` cues still match."""
        state = initial_state(
            "write the summary to results/summary.md and save the data to results/data.csv",
            "thread-fi5",
        )
        paths = _extract_goal_deliverables(state)
        assert "results/summary.md" in paths
        assert "results/data.csv" in paths

    def test_input_verb_inflection_still_excluded(self) -> None:
        """An inflected INPUT verb (``reads``/``loads``) excludes its file from the
        deliverable set, so an input named in the goal isn't mistaken for an output."""
        state = initial_state(
            "Reads results/q01/source.csv and writes the report to results/out.md",
            "thread-fi6",
        )
        paths = _extract_goal_deliverables(state)
        assert "results/out.md" in paths
        assert "results/q01/source.csv" not in paths


class TestPlaceholderLeakGuardFJ:
    """battery-04 q2 F-j: a present, non-empty text/markdown deliverable that
    ships with unsubstituted template placeholders is malformed and must not be
    rubber-stamped complete.

    Observed live: ``results/q02/quality_report.md`` completed with 13 prose
    tokens like ``{uniqueness_pct}`` / ``{total_rows}`` — the LLM wrote a
    ``.format()`` template but the codegen never rendered it. The verify LLM
    flagged it (60%), yet F-h force-complete overrode the verdict on objective
    evidence (presence + non-empty) because ``_classify_deliverable_format``
    skipped free-form text entirely. F-j closes that hole: text/markdown is now
    scanned for placeholder residue (code spans stripped first so legitimate
    f-strings / JSON / Jinja shown *inside* code are not flagged)."""

    LEAKY_REPORT = (
        "# Quality Report\n"
        "## Uniqueness\n"
        "**Score: {uniqueness_pct}%**\n"
        "### Evidence\n"
        "- Total rows: {total_rows}\n"
        "- Duplicate event_ids: {dup_event_ids}\n"
    )

    CLEAN_REPORT = (
        "# Quality Report\n"
        "## Uniqueness\n"
        "**Score: 98.5%**\n"
        "### Evidence\n"
        "- Total rows: 19\n"
        "- Duplicate event_ids: none\n"
        "```python\n"
        "print(f'Completeness: {completeness}%')  # legit code, not a leak\n"
        "```\n"
    )

    def test_leaky_report_flagged(self, tmp_path: Path) -> None:
        leaky = tmp_path / "quality_report.md"
        leaky.write_text(self.LEAKY_REPORT)
        reason = _placeholder_leak_reason(leaky)
        assert reason is not None
        assert "unsubstituted template placeholders" in reason
        assert "uniqueness_pct" in reason

    def test_clean_report_not_flagged(self, tmp_path: Path) -> None:
        clean = tmp_path / "quality_report.md"
        clean.write_text(self.CLEAN_REPORT)
        # The only {var} tokens are inside a fenced code block — stripped.
        assert _placeholder_leak_reason(clean) is None

    def test_classify_routes_markdown_through_leak_check(self, tmp_path: Path) -> None:
        """``_classify_deliverable_format`` returns the leak reason for .md/.txt."""
        leaky = tmp_path / "report.md"
        leaky.write_text("Score: {score_a}% and {score_b}%")
        reason = _classify_deliverable_format(leaky)
        assert reason is not None and "placeholder" in reason

    def test_classify_clean_markdown_returns_none(self, tmp_path: Path) -> None:
        clean = tmp_path / "report.md"
        clean.write_text("Score: 99% — fully rendered report.")
        assert _classify_deliverable_format(clean) is None

    def test_single_incidental_brace_below_threshold(self, tmp_path: Path) -> None:
        """One stray ``{placeholder}`` token in prose is not a template failure."""
        edge = tmp_path / "report.md"
        edge.write_text("Use the {placeholder} syntax in your template.")
        assert _placeholder_leak_reason(edge) is None

    def test_jinja_double_brace_residue_flagged(self, tmp_path: Path) -> None:
        jinja = tmp_path / "report.md"
        jinja.write_text("Score: {{ uniqueness }}% across {{ total }} rows.")
        reason = _placeholder_leak_reason(jinja)
        assert reason is not None

    def test_non_text_suffix_not_scanned(self, tmp_path: Path) -> None:
        """A .csv/.json file is never run through the placeholder heuristic —
        its own format parser governs (placeholders inside structured data are
        the format validator's job, and ``{var}`` in JSON is a parse error)."""
        struct = tmp_path / "data.json"
        struct.write_text('{"note": "uses {var} syntax"}')
        # JSON with {var} inside a string is still valid JSON -> not malformed.
        assert _classify_deliverable_format(struct) is None

    def test_check_deliverables_marks_leaky_md_malformed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        results_root = Path(_install_roots(monkeypatch, tmp_path))
        (results_root / "q02").mkdir(parents=True, exist_ok=True)
        (results_root / "q02" / "quality_report.md").write_text(self.LEAKY_REPORT)

        _evidence, _missing, _empty, malformed = _check_deliverables(
            ["results/q02/quality_report.md"]
        )
        assert len(malformed) == 1
        assert "placeholder" in malformed[0]

    def test_check_deliverables_passes_clean_md(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        results_root = Path(_install_roots(monkeypatch, tmp_path))
        (results_root / "q02").mkdir(parents=True, exist_ok=True)
        (results_root / "q02" / "quality_report.md").write_text(self.CLEAN_REPORT)

        _evidence, missing, empty, malformed = _check_deliverables(
            ["results/q02/quality_report.md"]
        )
        assert not missing and not empty and not malformed

    def test_goal_satisfied_false_when_report_leaks(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Load-bearing F-j/F-h.2 linkage: a leaky goal-named report makes
        ``_goal_deliverables_satisfied`` return False, so F-h force-complete is
        blocked and the agent retries until the report is fully rendered."""
        results_root = Path(_install_roots(monkeypatch, tmp_path))
        (results_root / "q02").mkdir(parents=True, exist_ok=True)
        (results_root / "q02" / "quality_report.md").write_text(self.LEAKY_REPORT)
        (results_root / "q02" / "scorecard.json").write_text(
            '{"scores": {"uniqueness": 98.5}}'
        )

        state = initial_state(
            "writes a report to results/q02/quality_report.md "
            "and results/q02/scorecard.json",
            "thread-fj-link",
        )
        satisfied, goal_paths = _goal_deliverables_satisfied(state)
        assert "results/q02/quality_report.md" in goal_paths
        assert satisfied is False
