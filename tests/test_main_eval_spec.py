"""Phase 10 — battery run_id → golden GoalSpec resolution (``main._resolve_eval_spec_id``).

A ``--run-id q01`` CLI run must resolve to the ``battery04_q01`` spec so the
verify node runs its correctness checks (and the eval store records a score),
WITHOUT forcing run_id to ``battery04_q01`` (which would isolate q1's
deliverables where q2/q3/q4 could not recall them via ``resolve_existing``).
The short form keeps writes under ``results/q01/`` while still opting into eval.
"""

from __future__ import annotations

import main as main_mod


class TestResolveEvalSpecId:
    def test_short_form_resolves_to_battery_spec(self) -> None:
        """``--run-id q01`` → ``battery04_q01`` (clean results/q01/ layout)."""
        assert main_mod._resolve_eval_spec_id("q01") == "battery04_q01"

    def test_all_four_battery_queries_resolve(self) -> None:
        assert main_mod._resolve_eval_spec_id("q01") == "battery04_q01"
        assert main_mod._resolve_eval_spec_id("q02") == "battery04_q02"
        assert main_mod._resolve_eval_spec_id("q03") == "battery04_q03"
        assert main_mod._resolve_eval_spec_id("q04") == "battery04_q04"

    def test_long_form_passes_through(self) -> None:
        """The explicit spec id is accepted directly too."""
        assert main_mod._resolve_eval_spec_id("battery04_q01") == "battery04_q01"

    def test_unknown_run_id_returns_none(self) -> None:
        """An ordinary run_id with no matching spec is untouched (no eval)."""
        assert main_mod._resolve_eval_spec_id("ordinary-run-xyz") is None

    def test_none_returns_none(self) -> None:
        """No run_id → no spec (a plain goal run, eval-free)."""
        assert main_mod._resolve_eval_spec_id(None) is None
