"""Phase 10 — battery run_id → golden GoalSpec resolution (``src.runner._resolve_eval_spec_id``).

A ``--run-id q01`` CLI run must resolve to the ``battery04_q01`` spec so the
verify node runs its correctness checks (and the eval store records a score),
WITHOUT forcing run_id to ``battery04_q01`` (which would isolate q1's
deliverables where q2/q3/q4 could not recall them via ``resolve_existing``).
The short form keeps writes under ``results/q01/`` while still opting into eval.
"""

from __future__ import annotations

import src.runner as runner


class TestResolveEvalSpecId:
    def test_short_form_resolves_to_battery_spec(self) -> None:
        """``--run-id q01`` → ``battery04_q01`` (clean results/q01/ layout)."""
        assert runner._resolve_eval_spec_id("q01") == "battery04_q01"

    def test_all_four_battery_queries_resolve(self) -> None:
        assert runner._resolve_eval_spec_id("q01") == "battery04_q01"
        assert runner._resolve_eval_spec_id("q02") == "battery04_q02"
        assert runner._resolve_eval_spec_id("q03") == "battery04_q03"
        assert runner._resolve_eval_spec_id("q04") == "battery04_q04"

    def test_long_form_passes_through(self) -> None:
        """The explicit spec id is accepted directly too."""
        assert runner._resolve_eval_spec_id("battery04_q01") == "battery04_q01"

    def test_unknown_run_id_returns_none(self) -> None:
        """An ordinary run_id with no matching spec is untouched (no eval)."""
        assert runner._resolve_eval_spec_id("ordinary-run-xyz") is None

    def test_date_suffix_strips_to_spec(self) -> None:
        """Nightly ``-YYYYMMDD`` suffix maps back to the spec for scoring."""
        assert (
            runner._resolve_eval_spec_id("battery04_q01-20260622")
            == "battery04_q01"
        )
        assert runner._resolve_eval_spec_id("q01-20260703") == "battery04_q01"

    def test_generation_tag_plus_date_strips_to_spec(self) -> None:
        """Multi-generation curve: ``-gen{N}-YYYYMMDD`` must still resolve.

        Regression: the self-improvement G0→G1→G2 curve enqueues each generation
        under ``{spec_id}-gen{N}-YYYYMMDD``. If the resolver left ``-gen0`` on,
        ``_resolve_eval_spec_id`` returned None and the verify node silently
        skipped eval scoring — the generation curve would have NO score signal.
        """
        assert (
            runner._resolve_eval_spec_id("q01-gen0-20260703") == "battery04_q01"
        )
        assert (
            runner._resolve_eval_spec_id("q01-gen1-20260703") == "battery04_q01"
        )
        assert (
            runner._resolve_eval_spec_id("q01-gen2-20260703") == "battery04_q01"
        )
        # Long-form spec + generation tag + date.
        assert (
            runner._resolve_eval_spec_id("battery04_q01-gen2-20260703")
            == "battery04_q01"
        )
        # A probe spec (underscores, no internal hyphen) also resolves.
        assert (
            runner._resolve_eval_spec_id("probe_create_tool-gen0-20260703")
            == "probe_create_tool"
        )

    def test_non_gen_hyphen_segment_is_preserved(self) -> None:
        """Only a ``-gen<N>`` tag is dropped — a stray hyphen segment is kept.

        ``my-spec-20260703`` has no ``gen<N>`` tag, so the resolver returns the
        date-stripped ``my-spec`` which matches no spec → None (no false match).
        """
        assert runner._resolve_eval_spec_id("my-spec-20260703") is None

    def test_none_returns_none(self) -> None:
        """No run_id → no spec (a plain goal run, eval-free)."""
        assert runner._resolve_eval_spec_id(None) is None
