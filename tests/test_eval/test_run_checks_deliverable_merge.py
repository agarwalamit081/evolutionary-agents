"""Regression: run_checks merges spec.expected_deliverables into the list.

An execution probe resolves its inputs from ``_DELIVERABLES``, which is built
from the ``deliverables`` list passed to ``run_checks``. In the live verify
path that list is the run's *declared* deliverables (``_extract_deliverables``),
which can omit a spec-named file the agent wrote but failed to list cleanly
(battery-04 q05: agent declared ``reconciled.csv`` but not
``integrity_report.json``/``audit.json``). Before the fix, that omission made
the probes for those files vacuously unresolvable → a false-negative FAIL on
otherwise-correct output (q05 scored 0.62 on data that scores 1.0 when the
spec's own expected deliverables are visible).

The fix: ``run_checks`` merges ``spec.expected_deliverables`` (deduped,
spec-first) so a check always sees every deliverable the spec names — the eval
must not depend on the agent's self-reported deliverable list for WHAT TO CHECK.

Hermetic: roots monkeypatched to an isolated tmp tree. No LLM, no DB.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.eval.checks import run_checks
from src.eval.models import CheckConfig, GoalSpec


def _patch_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    results = tmp_path / "results"
    workspace = tmp_path / "workspace"
    results.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)
    fake = SimpleNamespace(
        agent=SimpleNamespace(results_root=str(results), workspace_root=str(workspace)),
        eval=SimpleNamespace(
            eval_enabled=False,
            eval_enforce=False,
            eval_llm_judge_enabled=False,
            eval_canary_min_score=0.8,
            eval_store_enabled=False,
        ),
    )
    monkeypatch.setattr("src.config.settings.get_settings", lambda: fake)
    return results


def _spec() -> GoalSpec:
    """Minimal spec: one execution probe that reads a spec-named deliverable."""
    return GoalSpec(
        spec_id="merge_regr",
        name="merge_regr",
        description="regression for deliverable merge",
        goal_text="produce results/r/report.json with a count",
        category="test",
        expected_deliverables=["results/r/report.json", "results/r/data.csv"],
        success_criteria=[],
        checks=[
            CheckConfig(
                check_type="execution",
                name="report_has_count",
                params={
                    "code": (
                        "import json as _j, sys\n"
                        "p = next((x for x in _DELIVERABLES if x.endswith('report.json')), '')\n"
                        "if not p:\n"
                        "    print('no report.json'); sys.exit(1)\n"
                        "d = _j.load(open(p))\n"
                        "if int(d.get('count', -1)) != 7:\n"
                        "    print('wrong count'); sys.exit(1)\n"
                        "print('ok: count is 7')\n"
                    ),
                    "timeout": 20,
                },
            )
        ],
    )


class TestRunChecksDeliverableMerge:
    @pytest.mark.asyncio
    async def test_partial_declared_list_still_resolves(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The bug: agent declared ONLY data.csv, omitting report.json (which the
        # execution probe needs). Before the fix the probe saw an empty match and
        # failed with "no report.json". The spec's expected_deliverables must
        # make report.json resolvable regardless of the run's declared list.
        results = _patch_roots(monkeypatch, tmp_path)
        (results / "r").mkdir(parents=True, exist_ok=True)
        (results / "r" / "report.json").write_text('{"count": 7}')
        (results / "r" / "data.csv").write_text("a,b\n1,2\n")

        partial_declared = ["results/r/data.csv"]  # report.json intentionally omitted
        res = await run_checks(
            _spec(), partial_declared, {"thread_id": "regr", "iteration_count": 1, "max_iterations": 60}
        )
        assert res.passed is True
        assert res.overall_score == 1.0
        probe = next(c for c in res.checks if c.check_name == "report_has_count")
        assert probe.passed is True

    @pytest.mark.asyncio
    async def test_empty_declared_list_uses_spec_expected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Edge case: the run declares NOTHING (no extracted deliverables), yet the
        # spec knows what to check. The merge must still surface report.json.
        results = _patch_roots(monkeypatch, tmp_path)
        (results / "r").mkdir(parents=True, exist_ok=True)
        (results / "r" / "report.json").write_text('{"count": 7}')

        res = await run_checks(
            _spec(), [], {"thread_id": "regr2", "iteration_count": 1, "max_iterations": 60}
        )
        assert res.passed is True
        probe = next(c for c in res.checks if c.check_name == "report_has_count")
        assert probe.passed is True

    @pytest.mark.asyncio
    async def test_genuinely_missing_file_still_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Negative control: the merge only makes SPEC-NAMED files resolvable; a file
        # that genuinely does not exist on disk still fails. This proves the fix
        # did not turn the probe into a vacuous pass.
        _patch_roots(monkeypatch, tmp_path)  # empty results tree
        res = await run_checks(
            _spec(), [], {"thread_id": "regr3", "iteration_count": 1, "max_iterations": 60}
        )
        assert res.passed is False
        probe = next(c for c in res.checks if c.check_name == "report_has_count")
        assert probe.passed is False
