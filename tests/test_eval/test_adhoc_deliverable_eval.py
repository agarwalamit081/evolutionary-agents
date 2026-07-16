"""Ad-hoc deliverable eval: a run with NO battery GoalSpec still records an eval row.

``verify._run_correctness_checks`` used to early-return at ``spec is None`` for
every ad-hoc run, so fresh/unspecced queries got the LLM-verify narrative but
ZERO ``eval_results`` rows. The fix synthesizes a generic structural
``GoalSpec`` (``spec_id="adhoc-deliverables"``) over the on-disk deliverables
and records one parsed/non-empty check per file — pure observability (never
enforces: a parse hiccup must not loop a real run; verify's own completion gate
already enforces well-formedness).

Hermetic: results/workspace roots monkeypatched to an isolated tmp tree;
``EvalStore.record_correctness`` captured in-process. No LLM, no DB.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.graph.nodes.verify import _run_correctness_checks


# ─── harness helpers ─────────────────────────────────────────────────────


def _install_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    adhoc: bool = True,
    enforce: bool = True,
) -> Path:
    """Point both get_settings binding sites at an isolated tmp tree.

    ``src.tools._paths`` resolves the results root through the module attribute
    (``src.config.settings.get_settings``); ``verify`` bound ``get_settings`` by
    name at import, so both sites must be patched for the eval gate + the path
    resolver to agree on the tmp roots.
    """
    results_root = tmp_path / "results"
    workspace_root = tmp_path / "workspace"
    results_root.mkdir(parents=True, exist_ok=True)
    workspace_root.mkdir(parents=True, exist_ok=True)
    fake = SimpleNamespace(
        agent=SimpleNamespace(
            results_root=str(results_root), workspace_root=str(workspace_root)
        ),
        eval=SimpleNamespace(
            eval_enabled=True,
            eval_enforce=enforce,
            eval_adhoc_deliverables=adhoc,
            eval_rescue_incomplete=True,
            eval_store_enabled=True,
            eval_llm_judge_enabled=False,
            eval_canary_min_score=0.8,
        ),
    )
    monkeypatch.setattr("src.config.settings.get_settings", lambda: fake)
    monkeypatch.setattr("src.graph.nodes.verify.get_settings", lambda: fake)
    monkeypatch.chdir(tmp_path)
    return results_root


def _capture_store(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record every record_correctness call (goal_id, run_id, attempt, result)."""
    calls: list[Any] = []

    async def _record(
        self: Any,
        correctness: Any,
        *,
        goal_id: str,
        run_id: str,
        attempt_id: str | None = None,
        cost_usd: float = 0.0,
        producer_model: str | None = None,
    ) -> int:
        calls.append(
            SimpleNamespace(
                goal_id=goal_id,
                run_id=run_id,
                attempt_id=attempt_id,
                producer_model=producer_model,
                correctness=correctness,
            )
        )
        return len(correctness.checks)

    monkeypatch.setattr("src.eval.store.EvalStore.record_correctness", _record)
    return calls


def _state(thread: str = "api-adhoc-1", attempt: str = "att-1") -> dict[str, Any]:
    return {
        "thread_id": thread,
        "eval_attempt_id": attempt,
        "eval_goal_spec_id": None,  # no battery GoalSpec → ad-hoc path
        "iteration_count": 3,
        "eval_rescue_attempted": False,
    }


def _result_complete() -> dict[str, Any]:
    return {"is_complete": True, "final_output": "done", "errors": []}


# ─── tests ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_adhoc_writes_eval_row_for_on_disk_deliverables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    results = _install_roots(monkeypatch, tmp_path)
    calls = _capture_store(monkeypatch)

    (results / "report.csv").write_text("region,sales\nnorth,100\n", encoding="utf-8")
    (results / "summary.md").write_text("# Summary\nnon-empty body\n", encoding="utf-8")
    (results / "bad.json").write_text("{not valid json", encoding="utf-8")

    deliverables = ["results/report.csv", "results/summary.md", "results/bad.json"]
    out = await _run_correctness_checks(_result_complete(), _state(), deliverables, None)

    # Exactly one eval engagement, attributed to the synthetic ad-hoc spec.
    assert len(calls) == 1
    call = calls[0]
    assert call.goal_id == "adhoc-deliverables"
    assert call.run_id == "api-adhoc-1"
    assert call.attempt_id == "att-1"
    assert call.correctness.spec_id == "adhoc-deliverables"
    assert len(call.correctness.checks) == 3

    # Observability-only: a failing (malformed) check does NOT downgrade
    # completion even with eval_enforce=True.
    assert out["is_complete"] is True

    # The malformed JSON parse-fails; the well-formed CSV/MD pass.
    by_name = {c.check_name: c for c in call.correctness.checks}
    assert by_name["adhoc:bad.json"].passed is False
    assert by_name["adhoc:report.csv"].passed is True
    assert by_name["adhoc:summary.md"].passed is True


@pytest.mark.asyncio
async def test_adhoc_no_row_when_nothing_on_disk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_roots(monkeypatch, tmp_path)
    calls = _capture_store(monkeypatch)

    out = await _run_correctness_checks(
        _result_complete(), _state(), ["results/missing.csv"], None
    )
    # Nothing to check → no eval row, result unchanged.
    assert calls == []
    assert out["is_complete"] is True


@pytest.mark.asyncio
async def test_adhoc_disabled_writes_no_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    results = _install_roots(monkeypatch, tmp_path, adhoc=False)
    calls = _capture_store(monkeypatch)
    (results / "report.csv").write_text("region,sales\nnorth,100\n", encoding="utf-8")

    out = await _run_correctness_checks(
        _result_complete(), _state(), ["results/report.csv"], None
    )
    # Gate off → no synthetic spec, no row, result unchanged.
    assert calls == []
    assert out["is_complete"] is True


@pytest.mark.asyncio
async def test_adhoc_folds_score_and_checks_into_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    results = _install_roots(monkeypatch, tmp_path)
    _capture_store(monkeypatch)
    (results / "report.csv").write_text("region,sales\nnorth,100\n", encoding="utf-8")

    out = await _run_correctness_checks(
        _result_complete(), _state(), ["results/report.csv"], None
    )
    # The aggregate score + per-check breakdown are surfaced on the result for
    # observability (the same shape battery-spec eval produces).
    assert out["eval_correctness_score"] == pytest.approx(1.0)
    assert out["eval_correctness_passed"] is True
    assert len(out["eval_checks"]) == 1
    assert out["eval_checks"][0]["check_name"] == "adhoc:report.csv"


# ─── Phase E, Fix A: cross-file numeric-consistency probe ────────────────
# A prose report shipped alongside a computed data artifact can be parseable +
# non-empty yet still FABRICATE its numbers (battery-04 d-validation:
# sales_report.md contradicted sales_summary.csv). The ad-hoc eval now records a
# cross_file_consistency check so the drift is machine-visible — observability
# only (never enforced: is_complete is never downgraded).

_SALES_CSV = (
    "region,month,total_sales,daily_avg\n"
    "east,2026-01,2323.42,232.34\n"
    "east,2026-02,2725.81,247.8\n"
    "north,2026-01,2005.24,222.8\n"
    "north,2026-02,2666.45,266.64\n"
    "south,2026-01,3230.67,293.7\n"
    "south,2026-02,2624.44,291.6\n"
)
_FABRICATED_REPORT = (
    "# Sales Report\n"
    "East total: $5,049.23. North total: $4,671.69. South total: $5,855.11.\n"
    "Monthly: north-2026-01 $2,226.65; north-2026-02 $2,445.04; "
    "south-2026-01 $2,926.18; south-2026-02 $2,928.93; east-2026-02 avg $272.58.\n"
)


@pytest.mark.asyncio
async def test_adhoc_cross_file_drift_recorded_not_enforced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A fabricated report is detected (failing cross_file_consistency check +
    lower score) but stays observability-only — is_complete is NOT downgraded
    even with eval_enforce=True (a synthetic spec never loops a real run)."""
    results = _install_roots(monkeypatch, tmp_path)  # enforce=True default
    calls = _capture_store(monkeypatch)
    (results / "sales_summary.csv").write_text(_SALES_CSV, encoding="utf-8")
    (results / "sales_report.md").write_text(_FABRICATED_REPORT, encoding="utf-8")
    deliverables = ["results/sales_summary.csv", "results/sales_report.md"]

    out = await _run_correctness_checks(_result_complete(), _state(), deliverables, None)

    assert len(calls) == 1
    checks = {c.check_name: c for c in calls[0].correctness.checks}
    # Two structural checks (one per file) + one cross-file consistency check.
    assert len(calls[0].correctness.checks) == 3
    xf = checks["adhoc:cross_file_consistency"]
    assert xf.passed is False
    assert xf.evidence["drift_count"] == 5
    # The score reflects the drift; passed is False on the aggregate.
    assert out["eval_correctness_passed"] is False
    assert out["eval_correctness_score"] < 1.0
    # Observability-only — never enforced.
    assert out["is_complete"] is True
