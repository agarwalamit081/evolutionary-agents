"""Ad-hoc cross-file numeric-consistency probe (Phase E, Fix A).

``verify._cross_file_numeric_drift`` catches a prose report that FABRICATES its
numbers against a sibling computed data artifact — the battery-04 d-validation
regression where ``sales_report.md`` shipped correct combined region totals but
contradictory monthly/daily figures vs ``sales_summary.csv`` (LLM-authored prose
not grounded in the computed file). The ad-hoc spec only checks parse +
non-empty, so it rubber-stamped the drift; this probe makes it machine-visible.

Hermetic: results/workspace roots monkeypatched to an isolated tmp tree. No LLM,
no DB. The probe is observability-only (never enforced) — these tests lock the
drift detection + the no-signal skip contract.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.graph.nodes.verify import _cross_file_numeric_drift


# ─── harness ──────────────────────────────────────────────────────────────


def _install_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the path-resolver settings at an isolated tmp results/workspace tree.

    ``_resolve_deliverable`` → ``src.tools._paths.resolve_deliverable`` reads the
    results root through ``src.config.settings.get_settings``, so patching that
    binding is enough for the resolver to find the tmp deliverables.
    """
    results_root = tmp_path / "results"
    workspace_root = tmp_path / "workspace"
    results_root.mkdir(parents=True, exist_ok=True)
    workspace_root.mkdir(parents=True, exist_ok=True)
    fake = SimpleNamespace(
        agent=SimpleNamespace(
            results_root=str(results_root), workspace_root=str(workspace_root)
        )
    )
    monkeypatch.setattr("src.config.settings.get_settings", lambda: fake)
    monkeypatch.setattr("src.graph.nodes.verify.get_settings", lambda: fake)
    monkeypatch.chdir(tmp_path)
    return results_root


# The real battery-04 d-validation data (ground truth the report must match).
_SALES_CSV = (
    "region,month,total_sales,daily_avg\n"
    "east,2026-01,2323.42,232.34\n"
    "east,2026-02,2725.81,247.8\n"
    "north,2026-01,2005.24,222.8\n"
    "north,2026-02,2666.45,266.64\n"
    "south,2026-01,3230.67,293.7\n"
    "south,2026-02,2624.44,291.6\n"
)

# A report whose combined region totals are CORRECT (they equal the recomputed
# per-region sums) but whose monthly/daily figures are FABRICATED — the exact
# shape of the d-validation regression.
_FABRICATED_REPORT = (
    "# Sales Report\n\n"
    "## Combined Totals (correct)\n"
    "- East: $5,049.23\n"
    "- North: $4,671.69\n"
    "- South: $5,855.11\n\n"
    "## Monthly Breakdown (fabricated)\n"
    "| region | month | total | daily_avg |\n"
    "| east | 2026-01 | 2323.42 | 232.34 |\n"
    "| east | 2026-02 | 2725.81 | 272.58 |\n"
    "| north | 2026-01 | 2226.65 | 222.8 |\n"
    "| north | 2026-02 | 2445.04 | 266.64 |\n"
    "| south | 2026-01 | 2926.18 | 293.7 |\n"
    "| south | 2026-02 | 2928.93 | 291.6 |\n"
)

# A fully-grounded report — every figure is a CSV cell or a recomputed region sum.
_GROUNDED_REPORT = (
    "# Sales Report\n"
    "East total: $5,049.23. North total: $4,671.69. South total: $5,855.11.\n"
    "East Jan: 2323.42 (avg 232.34). North Feb: 2666.45 (avg 266.64).\n"
)


# ─── tests ────────────────────────────────────────────────────────────────


def test_drift_none_when_no_prose(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Data-only run → no prose to cross-check → skip (None)."""
    results = _install_roots(monkeypatch, tmp_path)
    (results / "sales.csv").write_text(_SALES_CSV, encoding="utf-8")
    assert _cross_file_numeric_drift(["results/sales.csv"]) is None


def test_drift_none_when_no_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Prose-only run → no data artifact to ground against → skip (None)."""
    results = _install_roots(monkeypatch, tmp_path)
    (results / "report.md").write_text(_FABRICATED_REPORT, encoding="utf-8")
    assert _cross_file_numeric_drift(["results/report.md"]) is None


def test_drift_none_when_no_decimal_claims(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A claim-free summary + data → no signal → skip (None), preserving the
    structural check count (the existing ad-hoc test relies on this skip)."""
    results = _install_roots(monkeypatch, tmp_path)
    (results / "sales.csv").write_text(_SALES_CSV, encoding="utf-8")
    (results / "summary.md").write_text("# Summary\nnon-empty body\n", encoding="utf-8")
    assert _cross_file_numeric_drift(["results/sales.csv", "results/summary.md"]) is None


def test_drift_none_when_data_has_no_numbers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Prose cites figures but the data artifact yields no ground-truth numbers
    (e.g. a categorical-only CSV) → nothing to match against → skip (None)."""
    results = _install_roots(monkeypatch, tmp_path)
    (results / "cats.csv").write_text("region,status\neast,active\n", encoding="utf-8")
    (results / "report.md").write_text("Total: 5000.00\n", encoding="utf-8")
    assert _cross_file_numeric_drift(["results/cats.csv", "results/report.md"]) is None


def test_drift_catches_fabricated_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The d-validation regression: a fabricated report is detected.

    Correct region totals (5049.23/4671.69/5855.11 = recomputed per-region sums;
    2323.42/2725.81/232.34 = CSV cells) MATCH, but the fabricated monthly/daily
    figures (2226.65, 2445.04, 2926.18, 2928.93, 272.58) DRIFT — proving the
    probe does not rubber-stamp a well-structured but dishonest report.
    """
    results = _install_roots(monkeypatch, tmp_path)
    (results / "sales_summary.csv").write_text(_SALES_CSV, encoding="utf-8")
    (results / "sales_report.md").write_text(_FABRICATED_REPORT, encoding="utf-8")

    res = _cross_file_numeric_drift(
        ["results/sales_summary.csv", "results/sales_report.md"]
    )
    assert res is not None
    assert res.check_name == "adhoc:cross_file_consistency"
    assert res.check_type == "cross_file_consistency"
    assert res.passed is False
    drift_values = {d["value"] for d in res.evidence["drift"]}
    assert drift_values == {2226.65, 2445.04, 2926.18, 2928.93, 272.58}
    assert res.evidence["drift_count"] == 5
    assert res.evidence["matched"] >= 5  # the correct totals + cells
    assert res.score < 1.0
    assert res.error  # non-empty reason


def test_drift_passes_grounded_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A report citing only CSV cells + recomputed region sums → no drift → pass.

    Guards against false positives: a legitimate aggregate (a region total that
    is a SUM of rows, not a cell value) must match the recomputed group sum.
    """
    results = _install_roots(monkeypatch, tmp_path)
    (results / "sales_summary.csv").write_text(_SALES_CSV, encoding="utf-8")
    (results / "sales_report.md").write_text(_GROUNDED_REPORT, encoding="utf-8")

    res = _cross_file_numeric_drift(
        ["results/sales_summary.csv", "results/sales_report.md"]
    )
    assert res is not None
    assert res.passed is True
    assert res.evidence["drift_count"] == 0
    assert res.score == pytest.approx(1.0)
    assert res.error == ""


def test_drift_handles_json_data_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A .json list-of-records data artifact also grounds the report (cells +
    recomputed totals/group sums), so a fabricated figure still drifts."""
    results = _install_roots(monkeypatch, tmp_path)
    (results / "sales.json").write_text(
        '[{"region":"east","sales":100.5},{"region":"east","sales":200.5},'
        '{"region":"north","sales":50.25}]',
        encoding="utf-8",
    )
    # 301.0 = east group sum (correct); 999.99 = fabricated.
    (results / "report.md").write_text(
        "East total: 301.0. Bogus: 999.99.\n", encoding="utf-8"
    )
    res = _cross_file_numeric_drift(["results/sales.json", "results/report.md"])
    assert res is not None
    assert res.passed is False
    assert {d["value"] for d in res.evidence["drift"]} == {999.99}


def test_drift_ignores_bare_integers_in_prose(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Bare integers (counts/years/ids) are NOT extracted as claims — a report
    citing '2026' or a row count must not register as drift against the data."""
    results = _install_roots(monkeypatch, tmp_path)
    (results / "sales.csv").write_text(_SALES_CSV, encoding="utf-8")
    (results / "report.md").write_text(
        "Year 2026, 6 rows reviewed. Total: 15575.03.\n", encoding="utf-8"
    )
    # 15575.03 = grand total of total_sales (correct); 2026 and 6 are ignored.
    res = _cross_file_numeric_drift(["results/sales.csv", "results/report.md"])
    assert res is not None
    assert res.passed is True
    assert res.evidence["drift_count"] == 0


def test_drift_malformed_data_file_is_skipped_not_crashed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A malformed data file is parsed best-effort (no crash); if a second valid
    data file grounds the report, the probe still runs over the valid one."""
    results = _install_roots(monkeypatch, tmp_path)
    (results / "bad.csv").write_text("region,sales\nnorth,notanumber\n", encoding="utf-8")
    (results / "good.csv").write_text("region,sales\nnorth,100.5\n", encoding="utf-8")
    (results / "report.md").write_text("Total: 100.5.\n", encoding="utf-8")
    res = _cross_file_numeric_drift(
        ["results/bad.csv", "results/good.csv", "results/report.md"]
    )
    assert res is not None
    assert res.passed is True
