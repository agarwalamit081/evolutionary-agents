"""Tests for the state helpers in ``src.graph.state``.

Focus: ``current_blocking_errors`` — the gate both the heuristic verify and
heuristic reflect read instead of the never-cleared ``errors`` accumulator
(``operator.add`` reducer). The ``errors`` list grows for the whole run and is
never cleared, so a ``"verify: deliverable not present — {path}"`` entry stamped
on an early pass would otherwise block completion forever (complex-arxiv-stats-4
looped to the iteration cap with all deliverables on disk because the stale entry
kept ``bool(errors)`` True). ``current_blocking_errors`` drops a
deliverable-error whose path is no longer in the fresh ``missing_deliverables``
view, while keeping genuinely-still-missing and all non-deliverable errors.
"""

from __future__ import annotations

from src.graph.state import current_blocking_errors

_PREFIX = "verify: deliverable not present — "


class TestCurrentBlockingErrors:
    """Resolution semantics for the heuristic completion gate."""

    def test_no_errors_returns_empty(self) -> None:
        assert current_blocking_errors(None) == []
        assert current_blocking_errors([]) == []

    def test_non_deliverable_errors_are_always_kept(self) -> None:
        errors = ["execute: tool x failed", "reflect: low confidence"]
        assert current_blocking_errors(errors) == errors

    def test_deliverable_error_resolved_when_no_longer_missing(self) -> None:
        # The file flagged on an early pass is now present → dropped.
        errors = [
            f"{_PREFIX}results/run1/foo.csv",
            "execute: transient retry recovered",
        ]
        # missing_deliverables is empty (verify's fresh view: nothing absent).
        assert current_blocking_errors(errors, []) == ["execute: transient retry recovered"]

    def test_deliverable_error_kept_when_still_missing(self) -> None:
        # The flagged deliverable is STILL absent → the error is genuine.
        path = "results/run1/bar.json"
        errors = [f"{_PREFIX}{path}"]
        assert current_blocking_errors(errors, [path]) == [f"{_PREFIX}{path}"]

    def test_mixed_resolved_kept_and_non_deliverable(self) -> None:
        errors = [
            f"{_PREFIX}results/r/resolved.csv",   # resolved (now on disk)
            f"{_PREFIX}results/r/still_missing.json",  # genuinely still missing
            "execute: rate limit hit",            # non-deliverable, always kept
        ]
        result = current_blocking_errors(
            errors, ["results/r/still_missing.json"]
        )
        assert result == [
            f"{_PREFIX}results/r/still_missing.json",
            "execute: rate limit hit",
        ]

    def test_regression_complex_arxiv_loop_unblocked(self) -> None:
        """The exact complex-arxiv-stats-4 failure mode.

        The accumulator holds stale deliverable-errors from early passes (files
        written later), but the fresh ``missing_deliverables`` view is empty — so
        the gate must clear and the run completes heuristically instead of
        looping to the iteration cap.
        """
        accumulated = [
            f"{_PREFIX}results/arxiv5/summary.json",
            f"{_PREFIX}results/arxiv5/stats.csv",
        ]
        assert current_blocking_errors(accumulated, []) == []
