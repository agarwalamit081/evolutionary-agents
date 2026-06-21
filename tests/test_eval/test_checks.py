"""Correctness checks (Phase 3): structural / golden / execution / oracle.

The structural/golden/execution checks are stdlib + subprocess — deterministic
and hermetic (deliverables rooted at a tmp dir). Oracle is exercised via its
skip paths (judge disabled / no deliverable / no judge available) so the suite
never depends on deepeval/ragas being importable or a funded LLM call.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.eval.checks import (
    ExecutionCheck,
    GoldenCheck,
    OracleCheck,
    StructuralCheck,
    run_checks,
)
from src.eval.models import CheckConfig, GoalSpec


async def _none_judge(text: str, reference: str) -> tuple[float, dict[str, Any]] | None:
    """Stand-in for an unavailable deepeval/ragas judge (returns no signal)."""
    return None


def _patch_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    """Point the shared settings source at isolated tmp roots (mirrors verify)."""
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
    return str(results)


def _spec(checks: list[CheckConfig], spec_id: str = "t") -> GoalSpec:
    return GoalSpec(spec_id=spec_id, name=spec_id, goal_text="g", checks=checks)


class TestStructuralCheck:
    @pytest.mark.asyncio
    async def test_csv_passes_when_fields_and_rows_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        Path(root, "data.csv").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        res = await StructuralCheck().check(
            CheckConfig(
                check_type="structural",
                name="s",
                params={
                    "deliverable": "results/data.csv",
                    "format": "csv",
                    "required_fields": ["a", "b"],
                    "min_rows": 1,
                },
            ),
            [],
            {},
        )
        assert res.passed is True
        assert res.score == 1.0
        assert res.evidence["rows"] == 2

    @pytest.mark.asyncio
    async def test_missing_deliverable_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_roots(monkeypatch, tmp_path)
        res = await StructuralCheck().check(
            CheckConfig(
                check_type="structural",
                name="s",
                params={"deliverable": "results/missing.csv"},
            ),
            [],
            {},
        )
        assert res.passed is False
        assert res.score == 0.0

    @pytest.mark.asyncio
    async def test_missing_required_field_fails_partial_score(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Score = fraction of declared CONDITIONS satisfied. required_fields
        # fails (z missing) but min_rows passes (1 row) → a genuine partial
        # score, while the check as a whole still fails.
        root = _patch_roots(monkeypatch, tmp_path)
        Path(root, "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        res = await StructuralCheck().check(
            CheckConfig(
                check_type="structural",
                name="s",
                params={
                    "deliverable": "results/data.csv",
                    "required_fields": ["a", "z"],
                    "min_rows": 1,
                },
            ),
            [],
            {},
        )
        assert res.passed is False  # z missing
        assert res.score == pytest.approx(0.5)  # 1 of 2 conditions satisfied

    @pytest.mark.asyncio
    async def test_json_parses_and_counts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        Path(root, "s.json").write_text(json.dumps({"rows": 3, "ok": True}), encoding="utf-8")
        res = await StructuralCheck().check(
            CheckConfig(
                check_type="structural",
                name="s",
                params={"deliverable": "results/s.json", "format": "json"},
            ),
            [],
            {},
        )
        assert res.passed is True
        assert {"rows", "ok"} <= set(res.evidence["fields"])


class TestGoldenCheck:
    @pytest.mark.asyncio
    async def test_exists_pass_and_fail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        Path(root, "r.md").write_text("hello UTC", encoding="utf-8")
        ok = await GoldenCheck().check(
            CheckConfig(
                check_type="golden",
                name="g",
                params={"assertions": [{"kind": "exists", "deliverable": "results/r.md"}]},
            ),
            [],
            {},
        )
        assert ok.passed is True
        bad = await GoldenCheck().check(
            CheckConfig(
                check_type="golden",
                name="g",
                params={"assertions": [{"kind": "exists", "deliverable": "results/nope.md"}]},
            ),
            [],
            {},
        )
        assert bad.passed is False

    @pytest.mark.asyncio
    async def test_contains_and_regex(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        Path(root, "r.md").write_text("Retention was 0.42 methodology", encoding="utf-8")
        res = await GoldenCheck().check(
            CheckConfig(
                check_type="golden",
                name="g",
                params={
                    "assertions": [
                        {"kind": "contains", "deliverable": "results/r.md", "value": "Retention"},
                        {"kind": "regex", "deliverable": "results/r.md", "pattern": r"\d+\.\d+"},
                    ]
                },
            ),
            [],
            {},
        )
        assert res.passed is True

    @pytest.mark.asyncio
    async def test_json_path_eq_with_tolerance(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        Path(root, "s.json").write_text(json.dumps({"score": 0.81}), encoding="utf-8")
        res = await GoldenCheck().check(
            CheckConfig(
                check_type="golden",
                name="g",
                params={
                    "assertions": [
                        {
                            "kind": "json_path_eq",
                            "deliverable": "results/s.json",
                            "path": "score",
                            "value": 0.8,
                            "tolerance": 0.05,
                        }
                    ]
                },
            ),
            [],
            {},
        )
        assert res.passed is True

    @pytest.mark.asyncio
    async def test_no_assertions_is_trivial_pass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_roots(monkeypatch, tmp_path)
        res = await GoldenCheck().check(
            CheckConfig(check_type="golden", name="g", params={}), [], {}
        )
        assert res.passed is True
        assert res.score == 1.0


class TestExecutionCheck:
    @pytest.mark.asyncio
    async def test_probe_passes_on_exit_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_roots(monkeypatch, tmp_path)
        res = await ExecutionCheck().check(
            CheckConfig(
                check_type="execution", name="e", params={"code": "print('ok', len(_DELIVERABLES))"}
            ),
            [],
            {},
        )
        assert res.passed is True
        assert "ok" in res.evidence["stdout"]

    @pytest.mark.asyncio
    async def test_probe_fails_on_nonzero_exit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_roots(monkeypatch, tmp_path)
        res = await ExecutionCheck().check(
            CheckConfig(
                check_type="execution",
                name="e",
                params={"code": "import sys; print('bad'); sys.exit(1)"},
            ),
            [],
            {},
        )
        assert res.passed is False
        assert res.evidence["exit_code"] == 1

    @pytest.mark.asyncio
    async def test_no_code_is_an_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_roots(monkeypatch, tmp_path)
        res = await ExecutionCheck().check(
            CheckConfig(check_type="execution", name="e", params={}), [], {}
        )
        assert res.passed is False
        assert "code" in res.error

    @pytest.mark.asyncio
    async def test_probe_can_read_deliverable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        Path(root, "data.csv").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        res = await ExecutionCheck().check(
            CheckConfig(
                check_type="execution",
                name="e",
                params={
                    "code": (
                        "import csv, sys\n"
                        "p = _DELIVERABLES[0]\n"
                        "n = sum(1 for _ in csv.DictReader(open(p)))\n"
                        "if n != 2:\n"
                        "    sys.exit(1)\n"
                        "print(f'rows={n}')\n"
                    )
                },
            ),
            ["results/data.csv"],
            {},
        )
        assert res.passed is True
        assert "rows=2" in res.evidence["stdout"]


class TestQ01UtcConformanceProbe:
    """The q01 UTC-conformance golden check — regression for the empty-timestamp bug.

    The normalizer dropped negative-offset timestamps to empty cells; the file was
    valid CSV with the timestamp column present, so the structural schema check
    passed (score 1.0) on a semantically wrong deliverable. Only this semantic
    probe catches it. Run against the REAL golden-spec config (not a copy) so any
    change to the probe is detected here.
    """

    def _probe_config(self) -> CheckConfig:
        from src.eval.golden import lookup_goal_spec

        spec = lookup_goal_spec("battery04_q01")
        assert spec is not None
        cfg = next((c for c in spec.checks if c.name == "q01_utc_conformance_probe"), None)
        assert cfg is not None, "q01_utc_conformance_probe must exist in the golden spec"
        return cfg

    @pytest.mark.asyncio
    async def test_empty_timestamps_fail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        # The q1 regression: negative-offset timestamps dropped to empty cells.
        Path(root, "normalized.csv").write_text(
            "event_id,timestamp\nevt_001,2024-01-15T11:15:00Z\nevt_002,\nevt_003,\n",
            encoding="utf-8",
        )
        res = await ExecutionCheck().check(self._probe_config(), ["results/normalized.csv"], {})
        assert res.passed is False
        assert "empty timestamp" in res.evidence["stdout"]

    @pytest.mark.asyncio
    async def test_non_utc_offset_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        # Offset-aware but NOT converted to UTC (no trailing Z) — the raw input shape.
        Path(root, "normalized.csv").write_text(
            "event_id,timestamp\nevt_001,2024-01-15T11:15:00-05:00\n",
            encoding="utf-8",
        )
        res = await ExecutionCheck().check(self._probe_config(), ["results/normalized.csv"], {})
        assert res.passed is False
        assert "trailing Z" in res.evidence["stdout"]

    @pytest.mark.asyncio
    async def test_valid_utc_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        Path(root, "normalized.csv").write_text(
            "event_id,timestamp\nevt_001,2024-01-15T16:15:00Z\nevt_002,2024-01-15T09:00:00Z\n",
            encoding="utf-8",
        )
        res = await ExecutionCheck().check(self._probe_config(), ["results/normalized.csv"], {})
        assert res.passed is True


class TestQ04TestResultsCountsProbe:
    """The q04 test_results.json counts check — regression for the failure-stub slip.

    The q4 orchestrator sub-agent wrote ``{"status": "failed", "reason": ...}``
    when it could not import a test module (it invoked the wrong filename —
    ``pytest test_retention_csv`` for a file named ``test_suite.py``). The OLD
    probe tested ``'pass' not in blob and 'fail' not in blob``; ``"failed"``
    contains ``"fail"``, so the stub passed verification and the run declared
    success on a fabricated result. The strengthened probe must reject any
    status=failed/error stub and require concrete INTEGER pass/fail counts.
    Run against the REAL golden-spec config so a weakening is caught here.
    """

    def _probe_config(self) -> CheckConfig:
        from src.eval.golden import lookup_goal_spec

        spec = lookup_goal_spec("battery04_q04")
        assert spec is not None
        cfg = next((c for c in spec.checks if c.name == "q04_test_results_has_counts"), None)
        assert cfg is not None, "q04_test_results_has_counts must exist in the golden spec"
        return cfg

    @pytest.mark.asyncio
    async def test_failure_stub_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        q04 = Path(root, "q04")
        q04.mkdir(parents=True, exist_ok=True)
        # The exact stub the q4 orchestrator produced.
        Path(q04, "test_results.json").write_text(
            json.dumps({"status": "failed", "reason": "No module named 'test_retention_csv'"}),
            encoding="utf-8",
        )
        res = await ExecutionCheck().check(
            self._probe_config(), ["results/q04/test_results.json"], {}
        )
        assert res.passed is False
        assert "stub" in res.evidence["stdout"]

    @pytest.mark.asyncio
    async def test_error_status_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        q04 = Path(root, "q04", "test_results.json")
        q04.parent.mkdir(parents=True, exist_ok=True)
        q04.write_text(json.dumps({"status": "error", "reason": "boom"}), encoding="utf-8")
        res = await ExecutionCheck().check(
            self._probe_config(), ["results/q04/test_results.json"], {}
        )
        assert res.passed is False

    @pytest.mark.asyncio
    async def test_real_counts_pass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        q04 = Path(root, "q04", "test_results.json")
        q04.parent.mkdir(parents=True, exist_ok=True)
        q04.write_text(json.dumps({"passed": 2, "failed": 2}), encoding="utf-8")
        res = await ExecutionCheck().check(
            self._probe_config(), ["results/q04/test_results.json"], {}
        )
        assert res.passed is True

    @pytest.mark.asyncio
    async def test_nested_summary_counts_pass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        q04 = Path(root, "q04", "test_results.json")
        q04.parent.mkdir(parents=True, exist_ok=True)
        # pytest-json-report style nesting under "summary".
        q04.write_text(
            json.dumps({"summary": {"passed": 4, "failed": 0, "total": 4}}),
            encoding="utf-8",
        )
        res = await ExecutionCheck().check(
            self._probe_config(), ["results/q04/test_results.json"], {}
        )
        assert res.passed is True

    @pytest.mark.asyncio
    async def test_only_total_without_pass_fail_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        q04 = Path(root, "q04", "test_results.json")
        q04.parent.mkdir(parents=True, exist_ok=True)
        q04.write_text(json.dumps({"total": 4}), encoding="utf-8")
        res = await ExecutionCheck().check(
            self._probe_config(), ["results/q04/test_results.json"], {}
        )
        assert res.passed is False

    @pytest.mark.asyncio
    async def test_all_zero_counts_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """battery-04 q4 run2: the orchestrator wrote {"passed":0,"failed":0}
        because its test file had a pandas bug so pytest collected nothing. The
        counts are well-formed integers (so the format gate passes), but a suite
        that executed zero tests validated nothing — the probe must reject it so
        eval_enforce downgrades and the agent retries with a runnable suite."""
        root = _patch_roots(monkeypatch, tmp_path)
        q04 = Path(root, "q04", "test_results.json")
        q04.parent.mkdir(parents=True, exist_ok=True)
        q04.write_text(json.dumps({"passed": 0, "failed": 0}), encoding="utf-8")
        res = await ExecutionCheck().check(
            self._probe_config(), ["results/q04/test_results.json"], {}
        )
        assert res.passed is False
        assert "pass/fail verdict" in res.evidence["stdout"]

    @pytest.mark.asyncio
    async def test_overall_status_stub_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """battery-04 q4 qwen false-positive: the orchestrator's test module
        failed to import (collection error), so it wrote a plausible-looking
        error report with ``overall_status: "failed"`` and a fabricated
        ``error_summary`` claiming the q1-q3 deliverables were missing (they
        were not). The OLD probe only inspected ``data.get('status')`` —
        ``overall_status`` slipped past the stub guard, so the run declared eval
        success on a result where ZERO tests executed. The stub guard must scan
        every status-like top-level key."""
        root = _patch_roots(monkeypatch, tmp_path)
        q04 = Path(root, "q04", "test_results.json")
        q04.parent.mkdir(parents=True, exist_ok=True)
        q04.write_text(
            json.dumps(
                {
                    "overall_status": "failed",
                    "summary": {"total_tests": 0, "passed": 0, "failed": 0, "errors": 1},
                    "error_summary": {
                        "primary_error": "Module import failed during test collection",
                        "missing_files": ["results/q01/normalized.csv"],
                    },
                }
            ),
            encoding="utf-8",
        )
        res = await ExecutionCheck().check(
            self._probe_config(), ["results/q04/test_results.json"], {}
        )
        assert res.passed is False
        assert "stub" in res.evidence["stdout"]

    @pytest.mark.asyncio
    async def test_all_errors_without_verdict_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A collection/import failure reports only ``errors`` (setup/collection
        outcomes), never a pass or fail verdict — no test ran to completion. The
        OLD 'tests ran' gate counted errors toward execution, so
        ``{passed:0, failed:0, errors:1}`` passed. The probe must require at
        least one concrete PASS or FAIL verdict. No top-level status stub here,
        so this isolates the verdict gate from the stub guard."""
        root = _patch_roots(monkeypatch, tmp_path)
        q04 = Path(root, "q04", "test_results.json")
        q04.parent.mkdir(parents=True, exist_ok=True)
        q04.write_text(
            json.dumps({"summary": {"passed": 0, "failed": 0, "errors": 1}}),
            encoding="utf-8",
        )
        res = await ExecutionCheck().check(
            self._probe_config(), ["results/q04/test_results.json"], {}
        )
        assert res.passed is False
        assert "pass/fail verdict" in res.evidence["stdout"]



def _spec_check(name: str) -> CheckConfig:
    """Fetch a named check from the REAL golden spec (not a copy)."""
    from src.eval.golden import lookup_goal_spec

    spec = lookup_goal_spec("battery04_q01")
    assert spec is not None
    cfg = next((c for c in spec.checks if c.name == name), None)
    assert cfg is not None, f"{name} must exist in the golden spec"
    return cfg


class TestQ01TimestampInstantPreservation:
    """The semantic gate the format probe cannot be.

    A normalizer must never change an instant. Run-1 dropped negative-offset
    timestamps to empty cells; run-2 shifted tz-naive timestamps by -1h. Both are
    *value* errors invisible to a format-only check; this per-event_id instant
    comparison catches both. Run against the REAL golden-spec config.
    """

    @pytest.mark.asyncio
    async def test_tz_naive_shift_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        Path(root, "raw_events.jsonl").write_text(
            '{"event_id": "evt_001", "timestamp": "2024-01-15T10:00:00-05:00"}\n'
            '{"event_id": "evt_002", "timestamp": "2024-01-15T12:00:00"}\n',
            encoding="utf-8",
        )
        # evt_002 (tz-naive, instant 12:00Z) wrongly shifted to 11:00:00Z — run-2 bug.
        Path(root, "normalized.csv").write_text(
            "event_id,timestamp\nEVT_001,2024-01-15T15:00:00Z\nEVT_002,2024-01-15T11:00:00Z\n",
            encoding="utf-8",
        )
        res = await ExecutionCheck().check(
            _spec_check("q01_timestamp_instant_preservation"),
            ["results/raw_events.jsonl", "results/normalized.csv"],
            {},
        )
        assert res.passed is False
        assert "instant changed" in res.evidence["stdout"]

    @pytest.mark.asyncio
    async def test_empty_timestamp_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        Path(root, "raw_events.jsonl").write_text(
            '{"event_id": "evt_002", "timestamp": "2024-01-15T12:00:00"}\n',
            encoding="utf-8",
        )
        # Negative-offset dropped to an empty cell — run-1 bug.
        Path(root, "normalized.csv").write_text("event_id,timestamp\nEVT_002,\n", encoding="utf-8")
        res = await ExecutionCheck().check(
            _spec_check("q01_timestamp_instant_preservation"),
            ["results/raw_events.jsonl", "results/normalized.csv"],
            {},
        )
        assert res.passed is False
        assert "empty" in res.evidence["stdout"]

    @pytest.mark.asyncio
    async def test_correct_conversion_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        Path(root, "raw_events.jsonl").write_text(
            '{"event_id": "evt_001", "timestamp": "2024-01-15T10:00:00-05:00"}\n'
            '{"event_id": "evt_002", "timestamp": "2024-01-15T12:00:00"}\n',
            encoding="utf-8",
        )
        # Both correctly converted: -05:00 -> 15:00Z; tz-naive -> 12:00Z (unchanged).
        Path(root, "normalized.csv").write_text(
            "event_id,timestamp\nEVT_001,2024-01-15T15:00:00Z\nEVT_002,2024-01-15T12:00:00Z\n",
            encoding="utf-8",
        )
        res = await ExecutionCheck().check(
            _spec_check("q01_timestamp_instant_preservation"),
            ["results/raw_events.jsonl", "results/normalized.csv"],
            {},
        )
        assert res.passed is True


class TestQ01RecordCountProbe:
    """summary.json.record_count must equal the normalized CSV's data-row count."""

    @pytest.mark.asyncio
    async def test_count_mismatch_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        Path(root, "normalized.csv").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        Path(root, "summary.json").write_text('{"record_count": 1}', encoding="utf-8")
        res = await ExecutionCheck().check(
            _spec_check("q01_record_count_probe"),
            ["results/normalized.csv", "results/summary.json"],
            {},
        )
        assert res.passed is False
        assert "record_count mismatch" in res.evidence["stdout"]

    @pytest.mark.asyncio
    async def test_count_match_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        Path(root, "normalized.csv").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        Path(root, "summary.json").write_text('{"record_count": 2}', encoding="utf-8")
        res = await ExecutionCheck().check(
            _spec_check("q01_record_count_probe"),
            ["results/normalized.csv", "results/summary.json"],
            {},
        )
        assert res.passed is True


def _q02_check(name: str) -> CheckConfig:
    """Fetch a named check from the REAL battery04_q02 golden spec."""
    from src.eval.golden import lookup_goal_spec

    spec = lookup_goal_spec("battery04_q02")
    assert spec is not None
    cfg = next((c for c in spec.checks if c.name == name), None)
    assert cfg is not None, f"{name} must exist in the battery04_q02 spec"
    return cfg


class TestQ02ReportNoPlaceholderLeak:
    """battery-04 q2 F-j: the harness must fail a report that ships with
    unsubstituted template placeholders in prose.

    Without this probe the harness scored quality_report.md 1.000 while it
    carried 13 ``{uniqueness_pct}``-style tokens (present, non-empty, and the
    lenient methodology regex matched). Run against the REAL golden-spec
    config; fenced/inline code is stripped first so f-strings in code don't
    false-positive."""

    LEAKY = (
        "# Quality Report\n"
        "**Score: {uniqueness_pct}%**\n"
        "- Total rows: {total_rows}\n"
        "- Duplicate event_ids: {dup_event_ids}\n"
    )
    CLEAN = (
        "# Quality Report\n"
        "**Score: 98.5%**\n"
        "- Total rows: 19\n"
        "```python\n"
        "print(f'{completeness}%')  # legit code, not a leak\n"
        "```\n"
    )

    @pytest.mark.asyncio
    async def test_leaky_report_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        Path(root, "quality_report.md").write_text(self.LEAKY, encoding="utf-8")
        res = await ExecutionCheck().check(
            _q02_check("q02_report_no_placeholder_leak"),
            ["results/quality_report.md"],
            {},
        )
        assert res.passed is False
        assert "placeholder leak" in res.evidence["stdout"]

    @pytest.mark.asyncio
    async def test_clean_report_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        Path(root, "quality_report.md").write_text(self.CLEAN, encoding="utf-8")
        res = await ExecutionCheck().check(
            _q02_check("q02_report_no_placeholder_leak"),
            ["results/quality_report.md"],
            {},
        )
        assert res.passed is True


class TestOracleCheck:
    @pytest.mark.asyncio
    async def test_skipped_when_judge_disabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)  # eval_llm_judge_enabled=False
        Path(root, "d.md").write_text("content", encoding="utf-8")
        res = await OracleCheck().check(
            CheckConfig(
                check_type="oracle", name="o", params={"deliverable": "results/d.md"}
            ),
            [],
            {},
            gateway=AsyncMock(),
        )
        assert res.skipped is True
        assert "disabled" in res.evidence["reason"]

    @pytest.mark.asyncio
    async def test_skipped_when_no_deliverable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Re-patch with the judge enabled but no deliverable on disk.
        results = tmp_path / "results"
        results.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(
            "src.config.settings.get_settings",
            lambda: SimpleNamespace(
                agent=SimpleNamespace(results_root=str(results), workspace_root=str(results)),
                eval=SimpleNamespace(eval_llm_judge_enabled=True, eval_canary_min_score=0.8),
            ),
        )
        res = await OracleCheck().check(
            CheckConfig(
                check_type="oracle", name="o", params={"deliverable": "results/absent.md"}
            ),
            [],
            {},
            gateway=None,
        )
        assert res.skipped is True

    @pytest.mark.asyncio
    async def test_skipped_when_no_judge_available(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        results = tmp_path / "results"
        results.mkdir(parents=True, exist_ok=True)
        (results / "d.md").write_text("some text", encoding="utf-8")
        monkeypatch.setattr(
            "src.config.settings.get_settings",
            lambda: SimpleNamespace(
                agent=SimpleNamespace(results_root=str(results), workspace_root=str(results)),
                eval=SimpleNamespace(eval_llm_judge_enabled=True, eval_canary_min_score=0.8),
            ),
        )
        # Force both optional judges unavailable and no gateway → skip.
        monkeypatch.setattr(OracleCheck, "_judge_deepeval", staticmethod(_none_judge))
        monkeypatch.setattr(OracleCheck, "_judge_ragas", staticmethod(_none_judge))
        res = await OracleCheck().check(
            CheckConfig(
                check_type="oracle", name="o", params={"deliverable": "results/d.md"}
            ),
            [],
            {},
            gateway=None,
        )
        assert res.skipped is True
        assert "no judge" in res.evidence["reason"]


class TestRunChecks:
    @pytest.mark.asyncio
    async def test_aggregates_pass_and_fail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        Path(root, "ok.csv").write_text("a\n1\n", encoding="utf-8")
        spec = _spec(
            [
                CheckConfig(
                    check_type="structural",
                    name="present",
                    params={"deliverable": "results/ok.csv", "format": "csv", "min_rows": 1},
                ),
                CheckConfig(
                    check_type="structural",
                    name="absent",
                    params={"deliverable": "results/missing.csv"},
                ),
            ]
        )
        corr = await run_checks(spec, ["results/ok.csv"], {})
        assert corr.passed is False
        assert corr.checks[0].passed is True
        assert corr.checks[1].passed is False
        # mean of 1.0 and 0.0
        assert corr.overall_score == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_skipped_checks_excluded_from_gate_and_mean(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)  # judge disabled → oracle skips
        Path(root, "ok.csv").write_text("a\n1\n", encoding="utf-8")
        spec = _spec(
            [
                CheckConfig(
                    check_type="structural",
                    name="present",
                    params={"deliverable": "results/ok.csv", "format": "csv", "min_rows": 1},
                ),
                CheckConfig(
                    check_type="oracle", name="judge", params={"deliverable": "results/ok.csv"}
                ),
            ]
        )
        corr = await run_checks(spec, ["results/ok.csv"], {}, gateway=None)
        # Only the structural check counts → it passed → gate passes, score 1.0
        assert corr.passed is True
        assert corr.overall_score == pytest.approx(1.0)
        assert any(c.skipped for c in corr.checks)

    @pytest.mark.asyncio
    async def test_no_checks_is_trivial_pass(self) -> None:
        corr = await run_checks(_spec([]), [], {})
        assert corr.passed is True
        assert corr.overall_score == 1.0

    @pytest.mark.asyncio
    async def test_unknown_check_type_fails(self) -> None:
        spec = _spec([CheckConfig(check_type="bogus", name="x", params={})])
        corr = await run_checks(spec, [], {})
        assert corr.passed is False
        assert "unknown check type" in corr.checks[0].error
