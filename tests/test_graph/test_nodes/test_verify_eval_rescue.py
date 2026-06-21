"""F-e: correctness checks rescue an incomplete run once deliverables appear.

``_run_correctness_checks`` used to early-return unless eval was enabled AND
the verdict was complete, so a run oscillating below ``is_complete`` was never
eval-rescued — its checks, which could guide the re-plan, only fired at
completion. The fix runs the checks the FIRST time deliverables appear on disk
on an incomplete verify (once per run, gated by ``eval_rescue_incomplete``),
folding the failing-check reasons into the re-plan as advisory feedback WITHOUT
forcing completion. These tests lock both halves: an early incomplete verify
with nothing on disk is still untouched, and an incomplete verify WITH a
deliverable on disk is rescued exactly once.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.graph.nodes.verify import _run_correctness_checks


def _fake_settings(results: Path, *, eval_enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(
        agent=SimpleNamespace(results_root=str(results), workspace_root=str(results / "ws")),
        eval=SimpleNamespace(
            eval_enabled=eval_enabled,
            eval_enforce=False,
            eval_llm_judge_enabled=False,
            eval_canary_min_score=0.8,
            eval_store_enabled=False,
            eval_rescue_incomplete=True,
        ),
    )


def _patch_settings(monkeypatch: pytest.MonkeyPatch, results: Path, *, eval_enabled: bool) -> None:
    fake = _fake_settings(results, eval_enabled=eval_enabled)
    # verify reads its own bound get_settings for the eval flag; checks.py /
    # _resolve_deliverable read src.config.settings.get_settings for results_root.
    # Patch both so deliverable resolution lands in the tmp tree.
    monkeypatch.setattr("src.graph.nodes.verify.get_settings", lambda: fake)
    monkeypatch.setattr("src.config.settings.get_settings", lambda: fake)


class TestRunCorrectnessChecksFEGate:
    @pytest.mark.asyncio
    async def test_incomplete_verdict_skips_checks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # F-e rescue gate: an INCOMPLETE verdict with NOTHING on disk yet is
        # untouched — there is genuinely nothing to check. (paths=[] → no
        # deliverable resolves on disk → rescue does not fire.)
        results = tmp_path / "results"
        results.mkdir(parents=True, exist_ok=True)
        _patch_settings(monkeypatch, results, eval_enabled=True)
        result = {"is_complete": False, "final_output": "still working"}
        state: dict[str, Any] = {"eval_goal_spec_id": "battery04_q01"}
        out = await _run_correctness_checks(result, state, [], None)
        assert out is result  # unchanged, same object
        assert "eval_correctness_score" not in out

    @pytest.mark.asyncio
    async def test_eval_disabled_skips_checks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        results = tmp_path / "results"
        results.mkdir(parents=True, exist_ok=True)
        _patch_settings(monkeypatch, results, eval_enabled=False)
        result = {"is_complete": True, "final_output": "done"}
        out = await _run_correctness_checks(result, {}, [], None)
        assert out is result
        assert "eval_correctness_score" not in out

    @pytest.mark.asyncio
    async def test_complete_verdict_with_spec_runs_checks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Positive control: the branch IS reachable when complete + enabled +
        # a spec is registered, so the skip tests above aren't passing vacuously.
        from src.eval.models import CheckConfig, GoalSpec

        results = tmp_path / "results"
        target = results / "t" / "out.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("payload")
        _patch_settings(monkeypatch, results, eval_enabled=True)

        spec = GoalSpec(
            spec_id="b04_fe_pos",
            name="b04_fe_pos",
            description="positive control",
            goal_text="produce results/t/out.txt",
            category="test",
            expected_deliverables=["results/t/out.txt"],
            success_criteria=[],
            checks=[
                CheckConfig(
                    check_type="golden",
                    name="exists",
                    params={"assertions": [{"kind": "exists", "deliverable": "results/t/out.txt"}]},
                )
            ],
        )
        monkeypatch.setattr("src.eval.golden.lookup_goal_spec", lambda sid: spec if sid == "b04_fe_pos" else None)

        # Stub the durable store so the positive control does not write a row to
        # the live eval_results table (the write is non-fatal in production).
        class _NoopStore:
            async def record_correctness(self, *_args: Any, **_kwargs: Any) -> None:
                return None

        monkeypatch.setattr("src.eval.store.EvalStore", _NoopStore)

        result = {"is_complete": True, "final_output": "done"}
        state: dict[str, Any] = {
            "eval_goal_spec_id": "b04_fe_pos",
            "thread_id": "fe-pos",
            "iteration_count": 0,
            "max_iterations": 60,
        }
        out = await _run_correctness_checks(result, state, ["results/t/out.txt"], None)
        assert "eval_correctness_score" in out
        assert out["eval_correctness_passed"] is True

    @pytest.mark.asyncio
    async def test_incomplete_with_deliverable_runs_rescue(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # F-e core: an INCOMPLETE verdict WITH a deliverable on disk + a spec is
        # RESCUED — the checks run, the score is folded in, the failing-check
        # reasons surface as advisory feedback, but is_complete stays False
        # (production-safe: rescue never forces completion).
        from src.eval.models import CheckConfig, GoalSpec

        results = tmp_path / "results"
        target = results / "t" / "out.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("payload")
        _patch_settings(monkeypatch, results, eval_enabled=True)

        # A spec with a check that FAILS (asserts a golden payload that differs)
        # so eval_enforce's advisory path is exercised.
        spec = GoalSpec(
            spec_id="b04_fe_rescue",
            name="b04_fe_rescue",
            description="rescue case",
            goal_text="produce results/t/out.txt",
            category="test",
            expected_deliverables=["results/t/out.txt"],
            success_criteria=[],
            checks=[
                CheckConfig(
                    check_type="golden",
                    name="golden_payload",
                    params={
                        "assertions": [
                            {
                                "kind": "equals",
                                "deliverable": "results/t/out.txt",
                                "expected": "WRONG_PAYLOAD",
                            }
                        ]
                    },
                )
            ],
        )
        monkeypatch.setattr(
            "src.eval.golden.lookup_goal_spec",
            lambda sid: spec if sid == "b04_fe_rescue" else None,
        )
        # verify binds get_settings at module level — patch BOTH bindings so the
        # eval gate opens with eval_enforce=True (rescue advisory path) and the
        # checks / _resolve_deliverable see the same tmp-rooted fake settings.
        fake = SimpleNamespace(
            agent=SimpleNamespace(
                results_root=str(results), workspace_root=str(results / "ws")
            ),
            eval=SimpleNamespace(
                eval_enabled=True,
                eval_enforce=True,
                eval_llm_judge_enabled=False,
                eval_canary_min_score=0.8,
                eval_store_enabled=False,
                eval_rescue_incomplete=True,
            ),
        )
        monkeypatch.setattr("src.graph.nodes.verify.get_settings", lambda: fake)
        monkeypatch.setattr("src.config.settings.get_settings", lambda: fake)

        class _NoopStore:
            async def record_correctness(self, *_args: Any, **_kwargs: Any) -> None:
                return None

        monkeypatch.setattr("src.eval.store.EvalStore", _NoopStore)

        result = {"is_complete": False, "final_output": "still working"}
        state: dict[str, Any] = {
            "eval_goal_spec_id": "b04_fe_rescue",
            "thread_id": "fe-rescue",
            "iteration_count": 1,
            "max_iterations": 60,
            "eval_rescue_attempted": False,
            "eval_attempt_id": "at-1",
        }
        out = await _run_correctness_checks(result, state, ["results/t/out.txt"], None)

        # Checks ran (score folded in) and the once-guard flipped.
        assert "eval_correctness_score" in out
        assert out["eval_rescue_attempted"] is True
        # Rescue is advisory: it must NOT force completion.
        assert out.get("is_complete") is False
        # The failing-check reason is surfaced into the re-plan feedback.
        assert any("correctness checks" in e for e in out.get("errors", []))

    @pytest.mark.asyncio
    async def test_rescue_skipped_after_first_attempt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Once-per-run bound: a second incomplete verify with deliverables on
        # disk must NOT re-run the (LLM-judge-bearing) checks. The once-guard
        # (state.eval_rescue_attempted=True) short-circuits so the judge fires
        # at most ~once extra per run.
        results = tmp_path / "results"
        target = results / "t" / "out.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("payload")
        _patch_settings(monkeypatch, results, eval_enabled=True)

        result = {"is_complete": False, "final_output": "still working"}
        state: dict[str, Any] = {
            "eval_goal_spec_id": "battery04_q01",
            "eval_rescue_attempted": True,  # already rescued this run
        }
        out = await _run_correctness_checks(result, state, ["results/t/out.txt"], None)
        assert out is result
        assert "eval_correctness_score" not in out

    @pytest.mark.asyncio
    async def test_rescue_disabled_by_setting(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # EVAL_RESCUE_INCOMPLETE=false restores the original F-e behavior: even
        # with deliverables on disk, an incomplete verify is never rescued.
        results = tmp_path / "results"
        target = results / "t" / "out.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("payload")
        monkeypatch.setattr(
            "src.graph.nodes.verify.get_settings",
            lambda: SimpleNamespace(
                agent=SimpleNamespace(
                    results_root=str(results), workspace_root=str(results / "ws")
                ),
                eval=SimpleNamespace(
                    eval_enabled=True,
                    eval_enforce=False,
                    eval_llm_judge_enabled=False,
                    eval_canary_min_score=0.8,
                    eval_store_enabled=False,
                    eval_rescue_incomplete=False,  # opt-out
                ),
            ),
        )

        result = {"is_complete": False, "final_output": "still working"}
        out = await _run_correctness_checks(
            result, {"eval_goal_spec_id": "battery04_q01"}, ["results/t/out.txt"], None
        )
        assert out is result
        assert "eval_correctness_score" not in out
