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

from src.graph.factory import initial_state
from src.graph.models import PlanStep
from src.graph.nodes.verify import _check_deliverables, _extract_deliverable_paths


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

        evidence_text, missing, empty = _check_deliverables(["report.md"])

        assert missing == []
        assert empty == []
        assert "Present:" in evidence_text
        assert "report.md" in evidence_text

    def test_declared_deliverable_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A declared file absent from disk → listed in ``missing``, not ``empty``."""
        _install_roots(monkeypatch, tmp_path)

        evidence_text, missing, empty = _check_deliverables(["ghost_report.md"])

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

        evidence_text, missing, empty = _check_deliverables(["empty.md"])

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

        evidence_text, missing, empty = _check_deliverables([])

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

        evidence_text, missing, empty = _check_deliverables(
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

        _evidence_text, missing, empty = _check_deliverables(paths)

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

        _evidence_text, missing, empty = _check_deliverables(["results/nested.md"])

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
        _evidence_text, missing, empty = _check_deliverables(paths)

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
        _evidence_text, missing, empty = _check_deliverables(paths)

        assert paths == ["results/never_written.md"]
        assert missing == ["results/never_written.md"]
        assert empty == []
