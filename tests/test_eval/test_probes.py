"""Unit tests for src/eval/probes.py — the learning probes (cross-run signal).

Two concerns:
- **Registration invariants:** all 3 probes resolve via ``lookup_goal_spec`` and
  live in ``GOLDEN_SPECS``; none are in ``BATTERY04_GOALS`` (mirrors the classify
  canaries — the nightly capability-curve battery must stay unperturbed).
- **Anti-fabrication backbone:** each probe's ``execution`` recompute code PASSES
  on a hand-built CORRECT deliverable and FAILS (sys.exit 1) on a wrong / missing
  one — so a hallucinated or stale deliverable cannot pass. The code strings are
  exec'd with the same ``_DELIVERABLES``/``_RESULTS_ROOT`` injection the real
  ExecutionCheck uses (see src/eval/checks.py), via a controlled namespace.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from src.eval.golden import BATTERY04_GOALS, GOLDEN_SPECS, lookup_goal_spec
from src.eval.probes import LEARNING_PROBES

_PROBE_IDS = [
    "probe_create_tool",
    "probe_reuse_tool",
    "probe_analytics_recall",
    "probe_multi_orchestration",
]


# ─── registration invariants ──────────────────────────────────────────────────


def test_learning_probes_listed_and_unique() -> None:
    ids = [p.spec_id for p in LEARNING_PROBES]
    assert ids == _PROBE_IDS
    assert len(ids) == len(set(ids))


def test_probes_resolve_via_lookup() -> None:
    for pid in _PROBE_IDS:
        spec = lookup_goal_spec(pid)
        assert spec is not None, pid
        assert spec.spec_id == pid


def test_probes_in_golden_specs() -> None:
    for pid in _PROBE_IDS:
        assert pid in GOLDEN_SPECS, pid


def test_probes_not_in_battery04_goals() -> None:
    """Probes are cross-run signals; the nightly battery must run only q01…q09."""
    battery_ids = {g.spec_id for g in BATTERY04_GOALS}
    for pid in _PROBE_IDS:
        assert pid not in battery_ids, f"{pid} leaked into BATTERY04_GOALS"


def test_each_probe_has_the_three_check_kinds() -> None:
    """golden(exists) + structural(format) + execution(recompute) — the q01 shape."""
    for pid in _PROBE_IDS:
        kinds = {c.check_type for c in GOLDEN_SPECS[pid].checks}
        assert kinds == {"golden", "structural", "execution"}, (pid, kinds)


# ─── recompute exec harness (mirrors checks.py's _DELIVERABLES injection) ─────


def _run_recompute(code: str, deliverables: list[str], results_root: str) -> tuple[bool, str]:
    """Exec a probe code string with the injected globals; return (passed, stdout).

    Pass = code ran to completion (no sys.exit-with-nonzero). Fail = SystemExit
    raised with a nonzero code, or any other exception.
    """
    ns: dict[str, Any] = {"__name__": "__recompute__", "_DELIVERABLES": deliverables,
                          "_RESULTS_ROOT": results_root}
    buf = io.StringIO()
    ns["print"] = lambda *a, **_kw: buf.write(" ".join(str(x) for x in a) + "\n")
    try:
        exec(code, ns)  # noqa: S102 — probe code is a fixed string from our own specs
    except SystemExit as exc:
        code_int = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        return (code_int == 0), buf.getvalue()
    except Exception as exc:  # noqa: BLE001 — any failure is a probe failure
        return False, f"{type(exc).__name__}: {exc}\n{buf.getvalue()}"
    return True, buf.getvalue()


def _exec_check(spec_id: str) -> str:
    """Pull the execution check's code string for a probe spec."""
    spec = GOLDEN_SPECS[spec_id]
    checks = [c for c in spec.checks if c.check_type == "execution"]
    assert len(checks) == 1, spec_id
    return checks[0].params["code"]


# ─── probe_create_tool recompute ──────────────────────────────────────────────


def _write_csv(path: Path, rows: list[tuple[str, float]]) -> str:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sku", "price_usd"])
        for sku, v in rows:
            w.writerow([sku, f"{v:.2f}"])
    return str(path)


def test_probe_create_recompute_passes_on_correct(tmp_path: Path) -> None:
    dlv = _write_csv(tmp_path / "normalized.csv", [("A001", 13.75), ("A002", 0.67), ("A003", 6.35)])
    ok, out = _run_recompute(_exec_check("probe_create_tool"), [dlv], str(tmp_path))
    assert ok, out
    assert "ok" in out


def test_probe_create_recompute_fails_on_wrong_value(tmp_path: Path) -> None:
    # A003 should be 6.35 (5*1.27); 99.99 is fabricated.
    dlv = _write_csv(tmp_path / "normalized.csv", [("A001", 13.75), ("A002", 0.67), ("A003", 99.99)])
    ok, out = _run_recompute(_exec_check("probe_create_tool"), [dlv], str(tmp_path))
    assert not ok
    assert "A003" in out  # names the offending sku


def test_probe_create_recompute_fails_on_missing_dedup(tmp_path: Path) -> None:
    # 4 rows (A001 duplicated) → dedup was skipped → row-count check fails.
    dlv = _write_csv(
        tmp_path / "normalized.csv",
        [("A001", 13.75), ("A001", 13.75), ("A002", 0.67), ("A003", 6.35)],
    )
    ok, out = _run_recompute(_exec_check("probe_create_tool"), [dlv], str(tmp_path))
    assert not ok
    assert "deduped rows" in out


def test_probe_create_recompute_fails_when_no_deliverable(tmp_path: Path) -> None:
    ok, out = _run_recompute(_exec_check("probe_create_tool"), [], str(tmp_path))
    assert not ok
    assert "no normalized.csv" in out


# ─── probe_reuse_tool recompute ───────────────────────────────────────────────


def test_probe_reuse_recompute_passes_on_correct(tmp_path: Path) -> None:
    dlv = _write_csv(
        tmp_path / "normalized.csv",
        [("B001", 31.75), ("B002", 50.00), ("B003", 6.70), ("B004", 8.89)],
    )
    ok, out = _run_recompute(_exec_check("probe_reuse_tool"), [dlv], str(tmp_path))
    assert ok, out


def test_probe_reuse_recompute_fails_on_wrong_value(tmp_path: Path) -> None:
    dlv = _write_csv(
        tmp_path / "normalized.csv",
        [("B001", 31.75), ("B002", 50.00), ("B003", 6.70), ("B004", 0.01)],
    )
    ok, _ = _run_recompute(_exec_check("probe_reuse_tool"), [dlv], str(tmp_path))
    assert not ok


# ─── probe_analytics_recall recompute ─────────────────────────────────────────


def _write_json(path: Path, obj: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")
    return str(path)


def test_probe_analytics_recompute_passes_on_correct(tmp_path: Path) -> None:
    dlv = _write_json(
        tmp_path / "stats.json",
        {"count": 7, "sum": 1450, "mean": 207.1429, "median": 200, "min": 50, "max": 400},
    )
    ok, out = _run_recompute(_exec_check("probe_analytics_recall"), [dlv], str(tmp_path))
    assert ok, out
    assert "all stats verified" in out


def test_probe_analytics_recompute_fails_on_wrong_mean(tmp_path: Path) -> None:
    dlv = _write_json(
        tmp_path / "stats.json",
        {"count": 7, "sum": 1450, "mean": 999.0, "median": 200, "min": 50, "max": 400},
    )
    ok, out = _run_recompute(_exec_check("probe_analytics_recall"), [dlv], str(tmp_path))
    assert not ok
    assert "mean" in out


def test_probe_analytics_recompute_is_case_insensitive_on_keys(tmp_path: Path) -> None:
    # The agent may emit Title-Case keys; the _find() helper lowercases both sides.
    dlv = _write_json(
        tmp_path / "stats.json",
        {"Count": 7, "Sum": 1450, "Mean": 207.142857, "Median": 200, "Min": 50, "Max": 400},
    )
    ok, _ = _run_recompute(_exec_check("probe_analytics_recall"), [dlv], str(tmp_path))
    assert ok


# ─── probe_multi_orchestration recompute (Track-1 multi-goal canary) ──────────


def _exec_check_by_name(spec_id: str, check_name: str) -> str:
    """Pull a NAMED execution check's code (probe_multi has 3 execution checks)."""
    spec = GOLDEN_SPECS[spec_id]
    for c in spec.checks:
        if c.check_type == "execution" and c.name == check_name:
            return c.params["code"]
    raise AssertionError(f"{spec_id}/{check_name} not found")


def test_probe_multi_has_three_deliverables_and_complex_category() -> None:
    spec = GOLDEN_SPECS["probe_multi_orchestration"]
    assert spec.category == "complex"
    assert spec.timeout_seconds == 300
    assert len(spec.expected_deliverables) == 3
    # Three execution checks (stats recompute, ranges recompute, summary cross-check).
    exec_checks = [c for c in spec.checks if c.check_type == "execution"]
    assert len(exec_checks) == 3


def test_probe_multi_stats_recompute_passes_on_correct(tmp_path: Path) -> None:
    dlv = _write_json(
        tmp_path / "probe_multi" / "stats.json",
        {"count": 5, "sum": 150, "mean": 30.0, "median": 30, "min": 10, "max": 50},
    )
    ok, out = _run_recompute(
        _exec_check_by_name("probe_multi_orchestration", "probe_multi_stats_recompute"),
        [dlv], str(tmp_path),
    )
    assert ok, out


def test_probe_multi_stats_recompute_fails_on_wrong_mean(tmp_path: Path) -> None:
    dlv = _write_json(
        tmp_path / "probe_multi" / "stats.json",
        {"count": 5, "sum": 150, "mean": 999.0, "median": 30, "min": 10, "max": 50},
    )
    ok, out = _run_recompute(
        _exec_check_by_name("probe_multi_orchestration", "probe_multi_stats_recompute"),
        [dlv], str(tmp_path),
    )
    assert not ok
    assert "mean" in out


def test_probe_multi_ranges_recompute_passes_on_correct(tmp_path: Path) -> None:
    dlv = _write_json(
        tmp_path / "probe_multi" / "ranges.json",
        {"count": 8, "min": 1, "max": 9, "range": 8, "sum": 31},
    )
    ok, _ = _run_recompute(
        _exec_check_by_name("probe_multi_orchestration", "probe_multi_ranges_recompute"),
        [dlv], str(tmp_path),
    )
    assert ok


def test_probe_multi_summary_crosscheck_passes_on_consistent(tmp_path: Path) -> None:
    """All three deliverables internally consistent → cross-check passes."""
    stats = _write_json(
        tmp_path / "probe_multi" / "stats.json",
        {"count": 5, "sum": 150, "mean": 30.0, "median": 30, "min": 10, "max": 50},
    )
    ranges = _write_json(
        tmp_path / "probe_multi" / "ranges.json",
        {"count": 8, "min": 1, "max": 9, "range": 8, "sum": 31},
    )
    summary = _write_json(
        tmp_path / "probe_multi" / "summary.json",
        {"stats_a_mean": 30.0, "range_b": 8, "total_count": 13, "combined_ok": True},
    )
    ok, out = _run_recompute(
        _exec_check_by_name("probe_multi_orchestration", "probe_multi_summary_crosscheck"),
        [stats, ranges, summary], str(tmp_path),
    )
    assert ok, out
    assert "cross-references verified" in out


def test_probe_multi_summary_crosscheck_fails_on_inconsistent_summary(
    tmp_path: Path,
) -> None:
    """The multi-goal collapse discriminator: stats.json is CORRECT (mean=30) but
    summary.json lies (stats_a_mean=99). The cross-check recomputes the expected
    30 FROM stats.json and catches the fabricated/inconsistent summary."""
    stats = _write_json(
        tmp_path / "probe_multi" / "stats.json",
        {"count": 5, "sum": 150, "mean": 30.0, "median": 30, "min": 10, "max": 50},
    )
    ranges = _write_json(
        tmp_path / "probe_multi" / "ranges.json",
        {"count": 8, "min": 1, "max": 9, "range": 8, "sum": 31},
    )
    summary = _write_json(
        tmp_path / "probe_multi" / "summary.json",
        {"stats_a_mean": 99.0, "range_b": 8, "total_count": 13, "combined_ok": True},
    )
    ok, out = _run_recompute(
        _exec_check_by_name("probe_multi_orchestration", "probe_multi_summary_crosscheck"),
        [stats, ranges, summary], str(tmp_path),
    )
    assert not ok
    assert "stats_a_mean" in out


def test_probe_multi_summary_crosscheck_fails_on_missing_upstream(
    tmp_path: Path,
) -> None:
    """A fabricated summary with NO upstream ranges.json → the cross-check cannot
    resolve the range_b reference and fails."""
    stats = _write_json(
        tmp_path / "probe_multi" / "stats.json",
        {"count": 5, "sum": 150, "mean": 30.0, "median": 30, "min": 10, "max": 50},
    )
    summary = _write_json(
        tmp_path / "probe_multi" / "summary.json",
        {"stats_a_mean": 30.0, "range_b": 8, "total_count": 13, "combined_ok": True},
    )
    ok, out = _run_recompute(
        _exec_check_by_name("probe_multi_orchestration", "probe_multi_summary_crosscheck"),
        [stats, summary], str(tmp_path),  # ranges.json deliberately absent
    )
    assert not ok
    assert "probe_multi/ranges.json" in out
