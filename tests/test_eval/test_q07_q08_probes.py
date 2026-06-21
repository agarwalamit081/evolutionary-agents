"""q07/q08 golden-spec probes: good fixtures pass, crafted bad fixtures FAIL.

Front-loaded "catch bugs early" gate (mirrors ``test_q05_q06_probes.py``). Every
recomputation / anti-fabrication / handoff guarantee in the q07/q08 specs is
locked here by driving the REAL spec's check code (looked up by name in
``GOLDEN_SPECS``) against fixtures that encode the exact failure mode:

- q07_constraints_recomputed (O3 crux): a constraint_report whose claimed values
  disagree with an independent recomputation from the raw instance is rejected.
- q07_no_hard_violations: a solution that assigns a task to an engineer lacking
  the required skill is rejected.
- q07_objective_sane: a non-optimizing solution that drops a higher-priority
  ADDABLE task (eligible engineer with spare capacity) while keeping a lower-
  priority one is rejected; a skill-feasible but capacity-BLOCKED task is
  genuinely unassignable and must NOT trip a false positive.
- q07_adversarial_sanity: an adversarial_report with attacks_tried < 2 (or
  violations_found != 0, or hardened != True) is rejected.
- q07_recompute_deterministic: the constraint recompute is reproducible.
- q08_handoff_integrity: a manifest entry whose input_sha256 or derived_value
  does not match an independent recomputation of the upstream artifact is rejected.
- q08_aggregation_complete: a fabricated analysis/synthesis whose totals disagree
  with a recomputation from raw_findings.jsonl is rejected.
- q08_no_placeholder_summary: >= 2 template-placeholder residues are rejected.

Hermetic: roots are monkeypatched to an isolated tmp tree. No LLM, no DB.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.eval.checks import CHECK_REGISTRY, run_checks
from src.eval.golden import GOLDEN_SPECS
from src.eval.models import CheckResult


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
# q07 fixture builders — constraint-satisfaction (task -> engineer assignment)
# --------------------------------------------------------------------------

_Q07_DELIVERABLES = [
    "results/q07/instance.json",
    "results/q07/solution.json",
    "results/q07/constraint_report.json",
    "results/q07/adversarial_report.json",
]

# 6 engineers, 3 skills; total capacity 47 vs total hours 66 (binding).
_Q07_INSTANCE = {
    "engineers": [
        {"id": "e1", "capacity_hours": 10, "skills": ["python", "sql"], "cost_per_hour": 50},
        {"id": "e2", "capacity_hours": 10, "skills": ["python", "frontend"], "cost_per_hour": 60},
        {"id": "e3", "capacity_hours": 8, "skills": ["sql"], "cost_per_hour": 40},
        {"id": "e4", "capacity_hours": 8, "skills": ["frontend"], "cost_per_hour": 45},
        {"id": "e5", "capacity_hours": 6, "skills": ["python", "sql", "frontend"], "cost_per_hour": 70},
        {"id": "e6", "capacity_hours": 5, "skills": ["python"], "cost_per_hour": 55},
    ],
    "tasks": [
        {"id": "t1", "required_hours": 5, "required_skill": "python", "priority": 5},
        {"id": "t2", "required_hours": 5, "required_skill": "python", "priority": 5},
        {"id": "t3", "required_hours": 4, "required_skill": "sql", "priority": 5},
        {"id": "t4", "required_hours": 4, "required_skill": "sql", "priority": 4},
        {"id": "t5", "required_hours": 6, "required_skill": "frontend", "priority": 4},
        {"id": "t6", "required_hours": 6, "required_skill": "frontend", "priority": 3},
        {"id": "t7", "required_hours": 3, "required_skill": "python", "priority": 3},
        {"id": "t8", "required_hours": 3, "required_skill": "sql", "priority": 3},
        {"id": "t9", "required_hours": 3, "required_skill": "frontend", "priority": 2},
        {"id": "t10", "required_hours": 4, "required_skill": "python", "priority": 2},
        {"id": "t11", "required_hours": 2, "required_skill": "sql", "priority": 2},
        {"id": "t12", "required_hours": 2, "required_skill": "frontend", "priority": 1},
        {"id": "t13", "required_hours": 2, "required_skill": "python", "priority": 1},
        {"id": "t14", "required_hours": 5, "required_skill": "sql", "priority": 1},
        {"id": "t15", "required_hours": 5, "required_skill": "frontend", "priority": 1},
        {"id": "t16", "required_hours": 6, "required_skill": "python", "priority": 1},
    ],
}

# A feasible near-optimal assignment (greedy by priority): assigns the 9 highest-
# priority tasks that fit, drops the low-priority tail. objective_value = 34.
_Q07_GOOD_ASSIGNMENTS = [
    {"task_id": "t1", "engineer_id": "e1"},
    {"task_id": "t3", "engineer_id": "e1"},
    {"task_id": "t2", "engineer_id": "e2"},
    {"task_id": "t7", "engineer_id": "e2"},
    {"task_id": "t4", "engineer_id": "e3"},
    {"task_id": "t8", "engineer_id": "e3"},
    {"task_id": "t5", "engineer_id": "e4"},
    {"task_id": "t6", "engineer_id": "e5"},
    {"task_id": "t10", "engineer_id": "e6"},
]
_Q07_GOOD_UNASSIGNED = ["t9", "t11", "t12", "t13", "t14", "t15", "t16"]


def _q07_report(assignments: list[dict], unassigned: list[str]) -> dict:
    """Recompute the constraint_report from the instance + a given solution.

    Used by the good fixture (honest) and the no-hard-violation BAD fixture
    (also honest, so only the targeted check fails).
    """
    eng = {e["id"]: e for e in _Q07_INSTANCE["engineers"]}
    tasks = {t["id"]: t for t in _Q07_INSTANCE["tasks"]}
    seen: set[str] = set()
    skill_mismatch = 0
    cap_used: dict[str, float] = {}
    violations: list[str] = []
    for a in assignments:
        tid, eid = a["task_id"], a["engineer_id"]
        if tid in seen:
            violations.append("dup " + tid)
            continue
        seen.add(tid)
        if tid not in tasks or eid not in eng:
            violations.append("unknown " + tid)
            continue
        if tasks[tid]["required_skill"] not in eng[eid]["skills"]:
            skill_mismatch += 1
            violations.append("skill " + tid)
        cap_used[eid] = cap_used.get(eid, 0.0) + tasks[tid]["required_hours"]
    for eid, used in cap_used.items():
        if used > eng[eid]["capacity_hours"] + 1e-6:
            violations.append("cap " + eid)
    for tid in unassigned:
        if tid in seen:
            violations.append("both " + tid)
    obj = sum(tasks[t]["priority"] for t in seen if t in tasks)
    return {
        "total_tasks": len(tasks),
        "assigned_tasks": len(seen),
        "unassigned_tasks": len(set(unassigned)),
        "hard_violations": len(violations),
        "skill_mismatches": skill_mismatch,
        "capacity_used": {k: round(v, 6) for k, v in cap_used.items()},
        "objective_recomputed": obj,
    }


def _build_q07(
    root: Path,
    *,
    assignments: list[dict] | None = None,
    unassigned: list[str] | None = None,
    report: dict | None = None,
    adversarial: dict | None = None,
) -> Path:
    """Write a q07 fixture. Defaults = the honest good solution + matching report."""
    q07 = root / "q07"
    q07.mkdir(parents=True, exist_ok=True)
    (q07 / "instance.json").write_text(json.dumps(_Q07_INSTANCE))
    asg = assignments if assignments is not None else _Q07_GOOD_ASSIGNMENTS
    una = unassigned if unassigned is not None else _Q07_GOOD_UNASSIGNED
    obj = sum(
        {t["id"]: t for t in _Q07_INSTANCE["tasks"]}[a["task_id"]]["priority"]
        for a in asg
        if a["task_id"] in {t["id"] for t in _Q07_INSTANCE["tasks"]}
    )
    (q07 / "solution.json").write_text(json.dumps({
        "assignments": asg,
        "unassigned": una,
        "objective_value": obj,
    }))
    rep = report if report is not None else _q07_report(asg, una)
    (q07 / "constraint_report.json").write_text(json.dumps(rep))
    adv = adversarial if adversarial is not None else {
        "attacks_tried": 3,
        "violations_found": 0,
        "hardened": True,
        "attacks": [
            "reassign t1 to e3 (lacks python)",
            "overload e5 with an extra task past capacity",
            "duplicate-assign t2",
        ],
    }
    (q07 / "adversarial_report.json").write_text(json.dumps(adv))
    return q07


class TestQ07Probes:
    @pytest.mark.asyncio
    async def test_good_fixture_all_checks_pass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        _build_q07(root)
        result = await run_checks(GOLDEN_SPECS["battery04_q07"], _Q07_DELIVERABLES, {})
        failed = [c.check_name for c in result.checks if not c.passed and not c.skipped]
        assert result.passed is True, f"failed checks: {failed}"
        assert result.overall_score == 1.0

    @pytest.mark.asyncio
    async def test_recompute_deterministic_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        _build_q07(root)
        res = await _run_one("battery04_q07", "q07_recompute_deterministic", _Q07_DELIVERABLES)
        assert res.passed is True, res.evidence
        assert res.evidence["identical"] is True

    @pytest.mark.asyncio
    async def test_fabricated_constraint_report_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # O3 crux: the report CLAIMS all 16 tasks were assigned with 0 violations,
        # but the solution only assigns 9. The recomputation must catch the lie.
        root = _patch_roots(monkeypatch, tmp_path)
        honest = _q07_report(_Q07_GOOD_ASSIGNMENTS, _Q07_GOOD_UNASSIGNED)
        fake = {**honest, "assigned_tasks": 16, "unassigned_tasks": 0, "objective_recomputed": 43}
        _build_q07(root, report=fake)
        res = await _run_one("battery04_q07", "q07_constraints_recomputed", _Q07_DELIVERABLES)
        assert res.passed is False
        assert "assigned_tasks" in res.evidence.get("stdout", "")

    @pytest.mark.asyncio
    async def test_solution_with_skill_violation_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Assign a frontend task to a sql-only engineer -> skill mismatch. The
        # report is kept HONEST (hard_violations=1) so only no_hard_violations
        # is the failing check.
        root = _patch_roots(monkeypatch, tmp_path)
        bad_asg = [{"task_id": "t5", "engineer_id": "e3"}] + [  # t5=frontend -> e3=sql only
            a for a in _Q07_GOOD_ASSIGNMENTS if a["task_id"] != "t5"
        ]
        honest = _q07_report(bad_asg, _Q07_GOOD_UNASSIGNED)
        _build_q07(root, assignments=bad_asg, report=honest)
        res = await _run_one("battery04_q07", "q07_no_hard_violations", _Q07_DELIVERABLES)
        assert res.passed is False
        assert "hard violations" in res.evidence.get("stdout", "")

    @pytest.mark.asyncio
    async def test_non_optimal_solution_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Drop t1 (priority 5, skill-feasible, room available on e1) while keeping
        # t10 (priority 2). The solution is feasible but non-optimizing; the
        # objective-sane optimization property must flag the mis-prioritization.
        root = _patch_roots(monkeypatch, tmp_path)
        nonopt_asg = [a for a in _Q07_GOOD_ASSIGNMENTS if a["task_id"] != "t1"]
        nonopt_una = ["t1"] + _Q07_GOOD_UNASSIGNED
        honest = _q07_report(nonopt_asg, nonopt_una)
        _build_q07(root, assignments=nonopt_asg, unassigned=nonopt_una, report=honest)
        res = await _run_one("battery04_q07", "q07_objective_sane", _Q07_DELIVERABLES)
        assert res.passed is False
        assert "non-optimal" in res.evidence.get("stdout", "")

    @pytest.mark.asyncio
    async def test_capacity_blocked_task_is_not_a_false_positive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # battery-04 q07 false-positive regression: a high-priority task that is
        # skill-feasible but whose ONLY eligible engineer is capacity-saturated
        # is genuinely UNassignable. Dropping it while keeping a lower-priority
        # task on a DIFFERENT engineer is optimal, not a greedy miss — the
        # objective-sane check must NOT flag it. (Pre-fix this tripped every run
        # whose generated instance had a capacity-bound feasible task.)
        root = _patch_roots(monkeypatch, tmp_path)
        q07 = root / "q07"
        q07.mkdir(parents=True, exist_ok=True)
        instance = {
            "engineers": [
                {"id": "e1", "capacity_hours": 10, "skills": ["python"], "cost_per_hour": 50},
                {"id": "e2", "capacity_hours": 10, "skills": ["sql"], "cost_per_hour": 40},
            ],
            "tasks": [
                {"id": "thigh", "required_hours": 10, "required_skill": "python", "priority": 10},
                {"id": "tmid", "required_hours": 10, "required_skill": "python", "priority": 5},
                {"id": "tlow", "required_hours": 5, "required_skill": "sql", "priority": 1},
            ],
        }
        (q07 / "instance.json").write_text(json.dumps(instance))
        # thigh saturates e1 (the only python engineer) -> tmid cannot fit.
        asg = [{"task_id": "thigh", "engineer_id": "e1"}, {"task_id": "tlow", "engineer_id": "e2"}]
        (q07 / "solution.json").write_text(json.dumps({
            "assignments": asg, "unassigned": ["tmid"], "objective_value": 11,
        }))
        (q07 / "constraint_report.json").write_text(json.dumps({
            "total_tasks": 3, "assigned_tasks": 2, "unassigned_tasks": 1,
            "hard_violations": 0, "skill_mismatches": 0,
            "capacity_used": {"e1": 10.0, "e2": 5.0}, "objective_recomputed": 11,
        }))
        (q07 / "adversarial_report.json").write_text(json.dumps({
            "attacks_tried": 2, "violations_found": 0, "hardened": True,
            "attacks": ["reassign thigh to e2 (lacks python)", "duplicate-assign thigh"],
        }))
        res = await _run_one("battery04_q07", "q07_objective_sane", _Q07_DELIVERABLES)
        assert res.passed is True, res.evidence

    @pytest.mark.asyncio
    async def test_weak_adversarial_report_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # attacks_tried < 2 -> the adversarial self-test was not exercised.
        root = _patch_roots(monkeypatch, tmp_path)
        _build_q07(root, adversarial={
            "attacks_tried": 1, "violations_found": 0, "hardened": True, "attacks": ["one"],
        })
        res = await _run_one("battery04_q07", "q07_adversarial_sanity", _Q07_DELIVERABLES)
        assert res.passed is False
        assert "attacks_tried" in res.evidence.get("stdout", "")


# --------------------------------------------------------------------------
# q08 fixture builders — multi-agent handoff pipeline
# --------------------------------------------------------------------------

_Q08_DELIVERABLES = [
    "results/q08/raw_findings.jsonl",
    "results/q08/analysis.json",
    "results/q08/synthesis.json",
    "results/q08/final_report.md",
    "results/q08/handoff_manifest.json",
]

# 15 findings across 3 categories (security/performance/reliability), 5 each.
_Q08_RAW = [
    # security: avg 3.8, total_metric 370 (worst by avg, then total_metric)
    {"id": "s1", "category": "security", "severity": 5, "metric_value": 100, "description": "a"},
    {"id": "s2", "category": "security", "severity": 4, "metric_value": 80, "description": "b"},
    {"id": "s3", "category": "security", "severity": 3, "metric_value": 60, "description": "c"},
    {"id": "s4", "category": "security", "severity": 2, "metric_value": 40, "description": "d"},
    {"id": "s5", "category": "security", "severity": 5, "metric_value": 90, "description": "e"},
    # performance: avg 2.4, total_metric 195
    {"id": "p1", "category": "performance", "severity": 3, "metric_value": 50, "description": "f"},
    {"id": "p2", "category": "performance", "severity": 2, "metric_value": 30, "description": "g"},
    {"id": "p3", "category": "performance", "severity": 4, "metric_value": 70, "description": "h"},
    {"id": "p4", "category": "performance", "severity": 1, "metric_value": 20, "description": "i"},
    {"id": "p5", "category": "performance", "severity": 2, "metric_value": 25, "description": "j"},
    # reliability: avg 3.8, total_metric 280 (loses the tie to security)
    {"id": "r1", "category": "reliability", "severity": 4, "metric_value": 60, "description": "k"},
    {"id": "r2", "category": "reliability", "severity": 5, "metric_value": 80, "description": "l"},
    {"id": "r3", "category": "reliability", "severity": 3, "metric_value": 45, "description": "m"},
    {"id": "r4", "category": "reliability", "severity": 4, "metric_value": 55, "description": "n"},
    {"id": "r5", "category": "reliability", "severity": 3, "metric_value": 40, "description": "o"},
]


def _q08_analysis() -> dict:
    """Per-category aggregates recomputed from _Q08_RAW."""
    cats: dict[str, list[dict]] = {}
    for f in _Q08_RAW:
        cats.setdefault(f["category"], []).append(f)
    out = []
    for c, items in cats.items():
        sevs = [it["severity"] for it in items]
        mets = [it["metric_value"] for it in items]
        top = sorted(items, key=lambda it: (-it["severity"], it["id"]))
        out.append({
            "category": c,
            "count": len(items),
            "avg_severity": round(sum(sevs) / len(sevs), 6),
            "total_metric": round(sum(mets), 6),
            "top_finding_ids": [it["id"] for it in top],
        })
    return {"categories": out, "total_findings": len(_Q08_RAW)}


def _q08_worst_category() -> str:
    agg = _q08_analysis()["categories"]
    best = max(agg, key=lambda c: (c["avg_severity"], c["total_metric"]))
    return best["category"]


def _sha(obj_or_text: str | dict | list) -> str:
    raw = obj_or_text if isinstance(obj_or_text, str) else json.dumps(obj_or_text)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _build_q08(
    root: Path,
    *,
    analysis: dict | None = None,
    synthesis: dict | None = None,
    final_md: str | None = None,
    manifest: dict | None = None,
) -> Path:
    """Write a q08 fixture. Defaults = honest, internally-consistent artifacts."""
    q08 = root / "q08"
    q08.mkdir(parents=True, exist_ok=True)
    (q08 / "raw_findings.jsonl").write_text("\n".join(json.dumps(f) for f in _Q08_RAW) + "\n")

    ana = analysis if analysis is not None else _q08_analysis()
    (q08 / "analysis.json").write_text(json.dumps(ana))
    worst = _q08_worst_category()
    syn = synthesis if synthesis is not None else {
        "total_findings": len(_Q08_RAW),
        "worst_category": worst,
        "worst_avg_severity": next(
            c["avg_severity"] for c in ana["categories"] if c["category"] == worst
        ),
        "category_count": len(ana["categories"]),
        "recommended_actions": ["patch the security findings first", "add reliability tests"],
    }
    (q08 / "synthesis.json").write_text(json.dumps(syn))

    md = final_md if final_md is not None else (
        f"# Incident Report\n\nAcross {len(_Q08_RAW)} findings, the worst-affected "
        f"category is **{worst}**. Recommended actions are prioritized accordingly.\n"
    )
    (q08 / "final_report.md").write_text(md)

    raw_sha = _sha("\n".join(json.dumps(f) for f in _Q08_RAW) + "\n")
    ana_sha = _sha(ana)
    syn_sha = _sha(syn)
    man = manifest if manifest is not None else {
        "stages": [
            {
                "stage": "analysis", "sub_agent": "analyst",
                "input_artifact": "results/q08/raw_findings.jsonl",
                "input_sha256": raw_sha,
                "output_artifact": "results/q08/analysis.json",
                "derived_value": str(len(_Q08_RAW)),
            },
            {
                "stage": "synthesis", "sub_agent": "synthesizer",
                "input_artifact": "results/q08/analysis.json",
                "input_sha256": ana_sha,
                "output_artifact": "results/q08/synthesis.json",
                "derived_value": worst,
            },
            {
                "stage": "orchestrator", "sub_agent": "orchestrator",
                "input_artifact": "results/q08/synthesis.json",
                "input_sha256": syn_sha,
                "output_artifact": "results/q08/final_report.md",
                "derived_value": worst,
            },
        ]
    }
    (q08 / "handoff_manifest.json").write_text(json.dumps(man))
    return q08


class TestQ08Probes:
    @pytest.mark.asyncio
    async def test_good_fixture_all_checks_pass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        _build_q08(root)
        result = await run_checks(GOLDEN_SPECS["battery04_q08"], _Q08_DELIVERABLES, {})
        failed = [c.check_name for c in result.checks if not c.passed and not c.skipped]
        assert result.passed is True, f"failed checks: {failed}"
        assert result.overall_score == 1.0

    @pytest.mark.asyncio
    async def test_fabricated_analysis_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # analysis.json claims security avg_severity = 1.0 (real = 3.8). The raw
        # recomputation must catch the fabricated aggregate.
        root = _patch_roots(monkeypatch, tmp_path)
        ana = _q08_analysis()
        for c in ana["categories"]:
            if c["category"] == "security":
                c["avg_severity"] = 1.0
        _build_q08(root, analysis=ana)
        res = await _run_one("battery04_q08", "q08_aggregation_complete", _Q08_DELIVERABLES)
        assert res.passed is False
        assert "avg_severity" in res.evidence.get("stdout", "")

    @pytest.mark.asyncio
    async def test_broken_handoff_derived_value_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The synthesis stage's derived_value lies ("performance") while the
        # recomputed worst_category from analysis.json is "security". The
        # content-verification must catch it (manifest did not actually parse).
        root = _patch_roots(monkeypatch, tmp_path)
        man = {
            "stages": [
                {
                    "stage": "analysis", "sub_agent": "analyst",
                    "input_artifact": "results/q08/raw_findings.jsonl",
                    "input_sha256": _sha("\n".join(json.dumps(f) for f in _Q08_RAW) + "\n"),
                    "output_artifact": "results/q08/analysis.json",
                    "derived_value": str(len(_Q08_RAW)),
                },
                {
                    "stage": "synthesis", "sub_agent": "synthesizer",
                    "input_artifact": "results/q08/analysis.json",
                    "input_sha256": _sha(_q08_analysis()),
                    "output_artifact": "results/q08/synthesis.json",
                    "derived_value": "performance",  # WRONG (recomputed = security)
                },
                {
                    "stage": "orchestrator", "sub_agent": "orchestrator",
                    "input_artifact": "results/q08/synthesis.json",
                    "input_sha256": _sha({
                        "total_findings": len(_Q08_RAW), "worst_category": "security",
                    }),
                    "output_artifact": "results/q08/final_report.md",
                    "derived_value": "security",
                },
            ]
        }
        _build_q08(root, manifest=man)
        res = await _run_one("battery04_q08", "q08_handoff_integrity", _Q08_DELIVERABLES)
        assert res.passed is False
        assert "derived_value" in res.evidence.get("stdout", "")

    @pytest.mark.asyncio
    async def test_wrong_handoff_sha256_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The analysis stage hashes the WRONG file (claims to have read
        # raw_findings.jsonl but its sha256 does not match).
        root = _patch_roots(monkeypatch, tmp_path)
        man = {
            "stages": [
                {
                    "stage": "analysis", "sub_agent": "analyst",
                    "input_artifact": "results/q08/raw_findings.jsonl",
                    "input_sha256": hashlib.sha256(b"not the raw file").hexdigest(),
                    "output_artifact": "results/q08/analysis.json",
                    "derived_value": str(len(_Q08_RAW)),
                },
                {
                    "stage": "synthesis", "sub_agent": "synthesizer",
                    "input_artifact": "results/q08/analysis.json",
                    "input_sha256": _sha(_q08_analysis()),
                    "output_artifact": "results/q08/synthesis.json",
                    "derived_value": _q08_worst_category(),
                },
                {
                    "stage": "orchestrator", "sub_agent": "orchestrator",
                    "input_artifact": "results/q08/synthesis.json",
                    "input_sha256": _sha({
                        "total_findings": len(_Q08_RAW), "worst_category": "security",
                    }),
                    "output_artifact": "results/q08/final_report.md",
                    "derived_value": "security",
                },
            ]
        }
        _build_q08(root, manifest=man)
        res = await _run_one("battery04_q08", "q08_handoff_integrity", _Q08_DELIVERABLES)
        assert res.passed is False
        assert "sha256" in res.evidence.get("stdout", "")

    @pytest.mark.asyncio
    async def test_placeholder_leak_in_report_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        # {{pending}} / {{another_var}} in the f-string render as literal
        # {pending} / {another_var} (single-brace) — >= 2 template residues.
        _build_q08(root, final_md=(
            f"# Report\n\nThe {len(_Q08_RAW)} findings in {_q08_worst_category()} "
            "need the {{pending}} and {{another_var}} fixed.\n"
        ))
        res = await _run_one("battery04_q08", "q08_no_placeholder_summary", _Q08_DELIVERABLES)
        assert res.passed is False

    @pytest.mark.asyncio
    async def test_report_missing_recomputed_total_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # final_report.md fails to cite the recomputed total — aggregation chain
        # requires the orchestrator to actually carry the number forward.
        root = _patch_roots(monkeypatch, tmp_path)
        _build_q08(root, final_md=(
            f"# Report\n\nThe worst category is {_q08_worst_category()}. "
            "No numbers are mentioned here at all.\n"
        ))
        res = await _run_one("battery04_q08", "q08_aggregation_complete", _Q08_DELIVERABLES)
        assert res.passed is False
        assert "total" in res.evidence.get("stdout", "")
