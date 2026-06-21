"""q09 golden-spec probes: good fixtures pass, crafted bad fixtures FAIL.

Front-loaded "catch bugs early" gate (mirrors ``test_q05_q06_probes.py`` and
``test_q07_q08_probes.py``). Every recomputation / anti-fabrication guarantee in the
q09 capability-robustness spec is locked here by driving the REAL spec's check code
(looked up by name in ``GOLDEN_SPECS``) against fixtures that encode the exact
failure mode:

- q09_outliers_recomputed: an anomalies.json whose q1/q3/iqr/fences OR whose outlier
  index SET disagree with an independent linear-interpolation Tukey-IQR recomputation
  from transactions.csv is rejected.
- q09_qa_independent_corroboration: a qa_verification.json whose recomputed outlier
  count / fences / per-row classification disagree with the recomputation, or whose
  sample_verified does not span at least one outlier AND one inlier, is rejected.

Hermetic: roots are monkeypatched to an isolated tmp tree. No LLM, no DB.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.eval.checks import CHECK_REGISTRY, run_checks
from src.eval.golden import GOLDEN_SPECS
from src.eval.models import CheckResult


# --------------------------------------------------------------------------
# Shared harness (self-contained copy of the q07/q08 probe-test helpers)
# --------------------------------------------------------------------------

def _patch_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the shared settings source at an isolated tmp results/workspace."""
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


async def _run_one(spec_id: str, check_name: str, deliverables: list[str]) -> CheckResult:
    """Run a SINGLE named check from the real spec (not a copy)."""
    spec = GOLDEN_SPECS[spec_id]
    cfg = next(c for c in spec.checks if c.name == check_name)
    check = CHECK_REGISTRY[cfg.check_type]
    return await check.check(cfg, deliverables, {})


# --------------------------------------------------------------------------
# q09 fixture builders — Tukey-IQR outlier detection
# --------------------------------------------------------------------------

_Q09_DELIVERABLES = [
    "results/q09/transactions.csv",
    "results/q09/anomalies.json",
    "results/q09/qa_verification.json",
]

_OUTLIER_INDICES = (10, 100, 300, 700, 950)


def _q09_amounts(*, clean: bool = False) -> list[float]:
    """1000 deterministic transaction amounts.

    Bulk tightly clustered 100..110 (so IQR fences sit just outside the cluster) plus
    5 clear high outliers at fixed indices. ``clean=True`` yields a flat column with
    NO outliers (used to exercise the seed-failure guard in the recomputation probes).
    """
    amts: list[float] = []
    for i in range(1000):
        if clean:
            amts.append(100.0)
        elif i in _OUTLIER_INDICES:
            amts.append(5000.0 + i)
        else:
            amts.append(100.0 + (i % 11))
    return amts


def _pct(sorted_vals: list[float], p: float) -> float:
    """Linear-interpolation percentile (numpy.percentile method='linear'). Mirrors
    the probe convention exactly so honest fixtures are recomputed identically."""
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    rank = (p / 100.0) * (n - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(sorted_vals[lo])
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (rank - lo))


def _q09_recompute(amounts: list[float]) -> tuple[float, float, float, float, float, list[int]]:
    svs = sorted(amounts)
    q1 = _pct(svs, 25)
    q3 = _pct(svs, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    oidx = [i for i, v in enumerate(amounts) if v < lower or v > upper]
    return q1, q3, iqr, lower, upper, oidx


def _build_q09(
    root: Path,
    *,
    amounts: list[float] | None = None,
    anomalies: dict | None = None,
    qa: dict | None = None,
) -> Path:
    """Write a q09 fixture. Defaults = honest, internally-consistent artifacts
    (anomalies + qa computed from the data by the correct Tukey-IQR algorithm)."""
    q09 = root / "q09"
    q09.mkdir(parents=True, exist_ok=True)

    amts = amounts if amounts is not None else _q09_amounts()
    lines = ["transaction_id,customer_id,amount,quantity,timestamp"]
    for i, a in enumerate(amts):
        day = (i % 27) + 1
        lines.append(f"t{i},c{i % 50},{a},{i % 5},2026-01-{day:02d}")
    (q09 / "transactions.csv").write_text("\n".join(lines) + "\n")

    q1, q3, iqr, lower, upper, oidx = _q09_recompute(amts)
    an = anomalies if anomalies is not None else {
        "column": "amount",
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_fence": lower,
        "upper_fence": upper,
        "outlier_row_indices": list(oidx),
    }
    (q09 / "anomalies.json").write_text(json.dumps(an))

    if qa is None:
        out_set = set(oidx)
        inliers = [i for i in range(len(amts)) if i not in out_set][:2]
        sample_rows = ([oidx[0]] if oidx else []) + inliers
        sample = [
            {
                "row_index": ri,
                "amount": amts[ri],
                "is_outlier": bool(amts[ri] < lower or amts[ri] > upper),
            }
            for ri in sample_rows
        ]
        qa_doc = {
            "recomputed_outlier_count": len(oidx),
            "fence_lower": lower,
            "fence_upper": upper,
            "sample_verified": sample,
        }
    else:
        qa_doc = qa
    (q09 / "qa_verification.json").write_text(json.dumps(qa_doc))
    return q09


class TestQ09Probes:
    @pytest.mark.asyncio
    async def test_good_fixture_all_checks_pass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        _build_q09(root)
        result = await run_checks(GOLDEN_SPECS["battery04_q09"], _Q09_DELIVERABLES, {})
        failed = [c.check_name for c in result.checks if not c.passed and not c.skipped]
        assert result.passed is True, f"failed checks: {failed}"
        assert result.overall_score == 1.0

    @pytest.mark.asyncio
    async def test_fabricated_fences_are_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # anomalies.json claims a q1 that is 1000 off the recomputed value. The
        # recomputation must catch the fabricated fence.
        root = _patch_roots(monkeypatch, tmp_path)
        amts = _q09_amounts()
        q1, q3, iqr, lower, upper, oidx = _q09_recompute(amts)
        _build_q09(
            root,
            amounts=amts,
            anomalies={
                "column": "amount", "q1": q1 + 1000.0, "q3": q3, "iqr": iqr,
                "lower_fence": lower, "upper_fence": upper,
                "outlier_row_indices": list(oidx),
            },
        )
        res = await _run_one("battery04_q09", "q09_outliers_recomputed", _Q09_DELIVERABLES)
        assert res.passed is False
        assert "q1" in res.evidence.get("stdout", "")

    @pytest.mark.asyncio
    async def test_wrong_outlier_index_set_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # anomalies.json drops one real outlier index. The index-SET recomputation
        # must catch the partial fabrication.
        root = _patch_roots(monkeypatch, tmp_path)
        amts = _q09_amounts()
        q1, q3, iqr, lower, upper, oidx = _q09_recompute(amts)
        _build_q09(
            root,
            amounts=amts,
            anomalies={
                "column": "amount", "q1": q1, "q3": q3, "iqr": iqr,
                "lower_fence": lower, "upper_fence": upper,
                "outlier_row_indices": oidx[:-1],  # drop the last outlier
            },
        )
        res = await _run_one("battery04_q09", "q09_outliers_recomputed", _Q09_DELIVERABLES)
        assert res.passed is False
        assert "outlier_row_indices" in res.evidence.get("stdout", "")

    @pytest.mark.asyncio
    async def test_qa_count_mismatch_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # qa_verification.json claims a wrong recomputed_outlier_count. Fences +
        # samples are honest, so only the count check fails.
        root = _patch_roots(monkeypatch, tmp_path)
        amts = _q09_amounts()
        _, _, _, lower, upper, oidx = _q09_recompute(amts)
        out_set = set(oidx)
        inliers = [i for i in range(len(amts)) if i not in out_set][:2]
        sample = [
            {"row_index": ri, "amount": amts[ri],
             "is_outlier": bool(amts[ri] < lower or amts[ri] > upper)}
            for ri in ([oidx[0]] + inliers)
        ]
        _build_q09(
            root,
            amounts=amts,
            qa={
                "recomputed_outlier_count": len(oidx) + 5,
                "fence_lower": lower, "fence_upper": upper,
                "sample_verified": sample,
            },
        )
        res = await _run_one("battery04_q09", "q09_qa_independent_corroboration", _Q09_DELIVERABLES)
        assert res.passed is False
        assert "recomputed_outlier_count" in res.evidence.get("stdout", "")

    @pytest.mark.asyncio
    async def test_qa_wrong_classification_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # One sample row is a real inlier but is_outlier=True. The per-row
        # recomputation must catch the misclassification.
        root = _patch_roots(monkeypatch, tmp_path)
        amts = _q09_amounts()
        _, _, _, lower, upper, oidx = _q09_recompute(amts)
        out_set = set(oidx)
        inlier = next(i for i in range(len(amts)) if i not in out_set)
        sample = [
            {"row_index": oidx[0], "amount": amts[oidx[0]], "is_outlier": True},
            {"row_index": inlier, "amount": amts[inlier], "is_outlier": True},  # WRONG
            {"row_index": inlier + 1, "amount": amts[inlier + 1], "is_outlier": False},
        ]
        _build_q09(
            root,
            amounts=amts,
            qa={
                "recomputed_outlier_count": len(oidx),
                "fence_lower": lower, "fence_upper": upper,
                "sample_verified": sample,
            },
        )
        res = await _run_one("battery04_q09", "q09_qa_independent_corroboration", _Q09_DELIVERABLES)
        assert res.passed is False
        assert "is_outlier" in res.evidence.get("stdout", "")

    @pytest.mark.asyncio
    async def test_qa_single_class_sample_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # All sample_verified rows are inliers — the QA file fails to demonstrate it
        # can distinguish outliers from inliers (must span both classes).
        root = _patch_roots(monkeypatch, tmp_path)
        amts = _q09_amounts()
        _, _, _, lower, upper, oidx = _q09_recompute(amts)
        out_set = set(oidx)
        only_inliers = [i for i in range(len(amts)) if i not in out_set][:3]
        sample = [
            {"row_index": ri, "amount": amts[ri], "is_outlier": False}
            for ri in only_inliers
        ]
        _build_q09(
            root,
            amounts=amts,
            qa={
                "recomputed_outlier_count": len(oidx),
                "fence_lower": lower, "fence_upper": upper,
                "sample_verified": sample,
            },
        )
        res = await _run_one("battery04_q09", "q09_qa_independent_corroboration", _Q09_DELIVERABLES)
        assert res.passed is False
        assert "span" in res.evidence.get("stdout", "")

    @pytest.mark.asyncio
    async def test_zero_outliers_seed_failure_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A flat `amount` column with zero outliers means the seed-data step failed
        # to inject anomalies. BOTH recomputation probes must reject it (the detector
        # cannot be validated against an anomaly-free column).
        root = _patch_roots(monkeypatch, tmp_path)
        amts = _q09_amounts(clean=True)
        _build_q09(root, amounts=amts)
        for probe in ("q09_outliers_recomputed", "q09_qa_independent_corroboration"):
            res = await _run_one("battery04_q09", probe, _Q09_DELIVERABLES)
            assert res.passed is False, f"{probe} should reject a zero-outlier column"
            assert "no outliers" in res.evidence.get("stdout", ""), probe
