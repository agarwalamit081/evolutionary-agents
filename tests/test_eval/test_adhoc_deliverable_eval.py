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
    ) -> int:
        calls.append(
            SimpleNamespace(
                goal_id=goal_id,
                run_id=run_id,
                attempt_id=attempt_id,
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
