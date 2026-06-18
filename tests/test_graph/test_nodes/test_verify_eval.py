"""Verify-node correctness-eval wiring (Phase 3): ``_run_correctness_checks``.

Exercises the integration seam where the verify node runs a registered
GoalSpec's checks and folds the result into the completion decision — without
spinning up the full LLM verify path. The spec is injected via a patched
``lookup_goal_spec`` so the assertions target the wiring logic (no-op gates,
score folding, enforce-downgrade) and not any particular golden spec.

The EvalStore write is disabled in the fake settings (``eval_store_enabled``
False) so no DB is touched.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.graph.enums import Phase
from src.graph.nodes.verify import _run_correctness_checks
from src.eval.models import CheckConfig, GoalSpec


def _settings(
    *,
    results: Path,
    enabled: bool,
    enforce: bool,
) -> object:
    return SimpleNamespace(
        agent=SimpleNamespace(results_root=str(results), workspace_root=str(results)),
        eval=SimpleNamespace(
            eval_enabled=enabled,
            eval_enforce=enforce,
            eval_llm_judge_enabled=False,
            eval_canary_min_score=0.8,
            eval_store_enabled=False,  # keep the suite DB-free
        ),
    )


def _spec(checks: list[CheckConfig]) -> GoalSpec:
    return GoalSpec(spec_id="t", name="t", goal_text="g", checks=checks)


def _patch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    enabled: bool,
    enforce: bool,
    spec: GoalSpec | None,
) -> str:
    results = tmp_path / "results"
    results.mkdir(parents=True, exist_ok=True)
    fake = _settings(results=results, enabled=enabled, enforce=enforce)
    # verify.py binds get_settings at MODULE level (line 11), so the bound name
    # in verify's namespace must be patched directly — patching only
    # src.config.settings.get_settings leaves verify's get_settings pointing at
    # the real settings and the eval gate never opens. The checks + EvalStore
    # import get_settings lazily, so the source-module patch covers those.
    monkeypatch.setattr("src.graph.nodes.verify.get_settings", lambda: fake)
    monkeypatch.setattr("src.config.settings.get_settings", lambda: fake)
    # verify imports lookup_goal_spec lazily inside the function, so patching the
    # source module's binding is what the call sees.
    monkeypatch.setattr("src.eval.golden.lookup_goal_spec", lambda _id: spec)
    return str(results)


class TestRunCorrectnessChecks:
    @pytest.mark.asyncio
    async def test_disabled_is_noop(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch(monkeypatch, tmp_path, enabled=False, enforce=False, spec=_spec([]))
        result = await _run_correctness_checks(
            {"is_complete": True, "final_output": "done"}, {"eval_goal_spec_id": "t"}, [], None
        )
        assert result == {"is_complete": True, "final_output": "done"}

    @pytest.mark.asyncio
    async def test_not_complete_is_noop(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch(monkeypatch, tmp_path, enabled=True, enforce=False, spec=_spec([]))
        result = await _run_correctness_checks(
            {"is_complete": False}, {"eval_goal_spec_id": "t"}, [], None
        )
        assert result == {"is_complete": False}

    @pytest.mark.asyncio
    async def test_no_spec_is_noop(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch(monkeypatch, tmp_path, enabled=True, enforce=False, spec=None)
        result = await _run_correctness_checks(
            {"is_complete": True, "final_output": "done"}, {"eval_goal_spec_id": "t"}, [], None
        )
        assert result == {"is_complete": True, "final_output": "done"}

    @pytest.mark.asyncio
    async def test_records_score_when_complete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch(
            monkeypatch,
            tmp_path,
            enabled=True,
            enforce=False,
            spec=_spec(
                [
                    CheckConfig(
                        check_type="structural",
                        name="present",
                        params={"deliverable": "results/data.csv", "format": "csv", "min_rows": 1},
                    )
                ]
            ),
        )
        Path(root, "data.csv").write_text("a\n1\n", encoding="utf-8")
        result = await _run_correctness_checks(
            {"is_complete": True, "final_output": "done"},
            {"eval_goal_spec_id": "t", "thread_id": "r1"},
            ["results/data.csv"],
            None,
        )
        assert result["is_complete"] is True  # not enforcing → stays complete
        assert result["eval_correctness_passed"] is True
        assert result["eval_correctness_score"] == pytest.approx(1.0)
        assert isinstance(result["eval_checks"], list)
        assert result["eval_checks"][0]["check_name"] == "present"

    @pytest.mark.asyncio
    async def test_enforce_downgrades_on_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Deliverable deliberately absent → structural check fails → enforce
        # downgrades complete→incomplete while iterations remain.
        _patch(
            monkeypatch,
            tmp_path,
            enabled=True,
            enforce=True,
            spec=_spec(
                [
                    CheckConfig(
                        check_type="structural",
                        name="present",
                        params={"deliverable": "results/missing.csv"},
                    )
                ]
            ),
        )
        result = await _run_correctness_checks(
            {"is_complete": True, "final_output": "draft"},
            {
                "eval_goal_spec_id": "t",
                "thread_id": "r1",
                "iteration_count": 3,
                "max_iterations": 60,
            },
            ["results/missing.csv"],
            None,
        )
        assert result["is_complete"] is False
        assert result["phase"] == Phase.EXECUTE
        assert result["eval_correctness_passed"] is False
        assert any("correctness" in str(e) for e in result.get("errors", []))

    @pytest.mark.asyncio
    async def test_enforce_completes_on_final_iteration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # On the last allowed verify (iteration == max - 1) a failing check must
        # NOT downgrade — the run completes regardless so a strict check can
        # never loop past the iteration hard-cap.
        _patch(
            monkeypatch,
            tmp_path,
            enabled=True,
            enforce=True,
            spec=_spec(
                [
                    CheckConfig(
                        check_type="structural",
                        name="present",
                        params={"deliverable": "results/missing.csv"},
                    )
                ]
            ),
        )
        result = await _run_correctness_checks(
            {"is_complete": True, "final_output": "draft"},
            {
                "eval_goal_spec_id": "t",
                "thread_id": "r1",
                "iteration_count": 59,
                "max_iterations": 60,
            },
            ["results/missing.csv"],
            None,
        )
        assert result["is_complete"] is True
        assert result["eval_correctness_passed"] is False  # still recorded


class TestRunCorrectnessChecksRealQ01Spec:
    """End-to-end demonstration of the eval_enforce loop against the REAL q01
    golden spec + a planted known-buggy deliverable (the run-2 tz-naive -1h
    shift). This is the mechanism the live q1 runs could not reach: a *complete*
    verdict over a present-but-semantically-wrong deliverable is caught by the
    instant-preservation check and downgraded → the graph retries. No LLM cost
    (q01 declares no oracle check); the execution checks run in the real sandbox.
    ``lookup_goal_spec`` is deliberately left UNPATCHED so the real spec resolves.
    """

    @staticmethod
    def _patch_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
        results = tmp_path / "results"
        results.mkdir(parents=True, exist_ok=True)
        fake = SimpleNamespace(
            agent=SimpleNamespace(results_root=str(results), workspace_root=str(results)),
            eval=SimpleNamespace(
                eval_enabled=True,
                eval_enforce=True,
                eval_llm_judge_enabled=False,
                eval_canary_min_score=0.8,
                eval_store_enabled=False,  # DB-free
            ),
        )
        # verify binds get_settings at module level — patch both bindings so the
        # eval gate opens and the checks see the same fake settings.
        monkeypatch.setattr("src.graph.nodes.verify.get_settings", lambda: fake)
        monkeypatch.setattr("src.config.settings.get_settings", lambda: fake)
        return str(results)

    @pytest.mark.asyncio
    async def test_enforce_downgrades_on_tz_naive_shift(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = self._patch_settings(monkeypatch, tmp_path)
        q01 = Path(root, "q01")
        q01.mkdir(parents=True, exist_ok=True)
        # Source: a tz-naive event whose UTC instant is 12:00Z.
        Path(q01, "raw_events.jsonl").write_text(
            '{"event_id": "evt_001", "timestamp": "2024-01-15T10:00:00-05:00", "amount": "10"}\n'
            '{"event_id": "evt_002", "timestamp": "2024-01-15T12:00:00", "amount": "20"}\n',
            encoding="utf-8",
        )
        # Deliverable: evt_002 wrongly shifted -1h (the run-2 bug). Valid UTC
        # FORMAT, wrong INSTANT — every format-level check passes; only the
        # instant-preservation check notices.
        Path(q01, "normalized.csv").write_text(
            "event_id,timestamp,amount\n"
            "EVT_001,2024-01-15T15:00:00Z,10.0\n"
            "EVT_002,2024-01-15T11:00:00Z,20.0\n",
            encoding="utf-8",
        )
        Path(q01, "summary.json").write_text(
            '{"record_count": 2, "duplicate_count_removed": 0, '
            '"fields": ["event_id", "timestamp", "amount"]}',
            encoding="utf-8",
        )

        result = await _run_correctness_checks(
            {"is_complete": True, "final_output": "draft deliverables written"},
            {
                "eval_goal_spec_id": "battery04_q01",
                "thread_id": "demo-q01",
                "iteration_count": 3,
                "max_iterations": 60,
            },
            [
                "results/q01/raw_events.jsonl",
                "results/q01/normalized.csv",
                "results/q01/summary.json",
            ],
            None,
        )

        # The instant-preservation check caught the -1h shift → enforce fired.
        assert result["is_complete"] is False
        assert result["phase"] == Phase.EXECUTE
        assert result["eval_correctness_passed"] is False
        by_name = {c["check_name"]: c for c in result["eval_checks"]}
        instant = by_name["q01_timestamp_instant_preservation"]
        assert instant["passed"] is False
        assert "instant changed" in instant["evidence"]["stdout"]
        # Every format-level check passed — proving only the semantic check saw it.
        assert by_name["q01_utc_conformance_probe"]["passed"] is True
        assert by_name["q01_normalized_csv_schema"]["passed"] is True
        assert by_name["q01_record_count_probe"]["passed"] is True
        assert any("correctness" in str(e) for e in result.get("errors", []))
