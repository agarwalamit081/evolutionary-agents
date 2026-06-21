"""q05/q06 golden-spec probes: good fixtures pass, crafted bad fixtures FAIL.

This is the front-loaded "catch bugs early" gate. Every recomputation /
anti-fabrication guarantee in the q05/q06 specs is locked here by driving the
REAL spec's check code (looked up by name in ``GOLDEN_SPECS``) against fixtures
that encode the exact failure mode:

- q05_orphan_cross_check (O3 crux): tool + auditor AGREE on a fabricated orphan
  count; the probe recomputes from the RAW sources and must reject it.
- q05_utc_conformance_reconciled: a non-UTC timestamp cell must fail.
- q05_no_duplicate_customers_survive: a duplicate customer with conflicting
  joined attributes must fail.
- q06_manifest_idempotent: a fabricated ``is_idempotent:true`` whose sha256 does
  not match the on-disk rollup must fail.
- q06_test_results_real_counts: a ``{"status":"failed"}`` stub must fail.
- q06_no_placeholder_summary: >=2 template-placeholder residues must fail.
- q06_v2_has_rolling_avg: a v2 with no rolling-average column must fail.
- q06_utc_dates: an unparseable date cell must fail.

Hermetic: roots are monkeypatched to an isolated tmp tree (mirrors the verify
node's path resolution). No LLM, no DB.
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
# q05 fixture builders
# --------------------------------------------------------------------------

_Q05_DELIVERABLES = [
    "results/q05/customers.jsonl",
    "results/q05/orders.jsonl",
    "results/q05/reconciled.csv",
    "results/q05/integrity_report.json",
    "results/q05/audit.json",
]


def _q05_sources() -> tuple[list[str], list[str]]:
    """Raw customers.jsonl + orders.jsonl with seeded integrity violations.

    Returns (customer_lines, order_lines). 12 distinct canonical customer ids
    (with 2 duplicate customer lines + mixed case), 30 matched orders + 8
    orphans + 2 duplicate order events, tz-naive timestamps, mixed-case FKs.
    """
    names = {f"c{i:03d}": f"Customer {i}" for i in range(1, 13)}
    emails = {f"c{i:03d}": f"c{i:03d}@example.com" for i in range(1, 13)}
    mixed = ["C001", "c002", "C003", "c004", "C005", "c006", "C007", "c008", "C009", "c010", "C011", "c012"]
    cust_lines = [json.dumps({"customer_id": cid, "name": names[cid.lower()], "email": emails[cid.lower()]}) for cid in mixed]
    # 2 duplicate customer lines (duplicate_customers_removed == 2).
    cust_lines.append(json.dumps({"customer_id": "C001", "name": names["c001"], "email": emails["c001"]}))
    cust_lines.append(json.dumps({"customer_id": "c002", "name": names["c002"], "email": emails["c002"]}))

    order_lines: list[str] = []

    def _add(oid: str, fk: str, day: int, rev: float) -> None:
        order_lines.append(json.dumps({"order_id": oid, "customer_id": fk, "timestamp": f"2026-01-0{day} 10:00:00", "revenue": rev}))

    # 30 matched orders cycling the 12 real customers; FKs alternate case.
    for i in range(1, 31):
        key = f"c{((i - 1) % 12) + 1:03d}"
        fk = key.upper() if i % 2 == 0 else key
        _add(f"ord_{i:03d}", fk, ((i - 1) % 5) + 1, round(i * 1.5, 2))
    # 2 duplicate order events (same order_id as ord_001) — must be deduped.
    _add("ord_001", "c001", 1, 1.5)
    _add("ord_001", "C001", 1, 1.5)
    # 8 orphan orders referencing nonexistent customers.
    for i in range(31, 39):
        _add(f"ord_{i:03d}", f"c9{i - 30:02d}", ((i - 1) % 5) + 1, round(i * 1.5, 2))
    return cust_lines, order_lines


def _q05_reconciled_rows(order_lines: list[str]) -> list[dict[str, str]]:
    """The CORRECT reconciliation: matched orders, joined attrs, UTC ts."""
    names = {f"c{i:03d}": f"Customer {i}" for i in range(1, 13)}
    emails = {f"c{i:03d}": f"c{i:03d}@example.com" for i in range(1, 13)}
    cust_ids = {f"c{i:03d}" for i in range(1, 13)}
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    for line in order_lines:
        o = json.loads(line)
        oid = o["order_id"]
        if oid in seen:
            continue
        fk = o["customer_id"].strip().lower()
        if fk not in cust_ids:
            continue  # orphan — not in the joined output
        seen.add(oid)
        ts = o["timestamp"].replace(" ", "T") + "Z"
        rows.append({
            "order_id": oid,
            "customer_id": fk,
            "customer_name": names[fk],
            "customer_email": emails[fk],
            "timestamp": ts,
            "revenue": str(o["revenue"]),
        })
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    import csv as _csv

    if not rows:
        path.write_text("\n")
        return
    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _build_q05_good(root: Path) -> dict:
    """Correct end-to-end q05 fixture. Returns computed counts."""
    q05 = root / "q05"
    q05.mkdir(parents=True, exist_ok=True)
    cust_lines, order_lines = _q05_sources()
    (q05 / "customers.jsonl").write_text("\n".join(cust_lines) + "\n")
    (q05 / "orders.jsonl").write_text("\n".join(order_lines) + "\n")
    rows = _q05_reconciled_rows(order_lines)
    _write_csv(q05 / "reconciled.csv", rows)
    # total unique orders = 38 (30 matched + 8 orphan; dups deduped)
    total = len({json.loads(_line)["order_id"] for _line in order_lines})
    matched = len(rows)
    orphan = 8
    report = {
        "total_orders": total,
        "matched_orders": matched,
        "orphaned_orders": orphan,
        "duplicate_customers_removed": 2,
        "all_timestamps_utc": True,
    }
    (q05 / "integrity_report.json").write_text(json.dumps(report))
    (q05 / "audit.json").write_text(json.dumps({
        "independent_orphan_count": orphan,
        "tool_orphan_count": orphan,
        "matches_tool": True,
    }))
    return report


class TestQ05Probes:
    @pytest.mark.asyncio
    async def test_good_fixture_all_checks_pass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        _build_q05_good(root)
        result = await run_checks(GOLDEN_SPECS["battery04_q05"], _Q05_DELIVERABLES, {})
        failed = [c.check_name for c in result.checks if not c.passed and not c.skipped]
        assert result.passed is True, f"failed checks: {failed}"
        assert result.overall_score == 1.0

    @pytest.mark.asyncio
    async def test_fabricated_orphan_count_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # O3 crux: tool AND auditor agree on a WRONG orphan count; the probe
        # recomputes from raw sources and must catch the fabrication.
        root = _patch_roots(monkeypatch, tmp_path)
        report = _build_q05_good(root)
        q05 = root / "q05"
        fake_orphan = report["orphaned_orders"] + 5
        (q05 / "integrity_report.json").write_text(json.dumps({**report, "orphaned_orders": fake_orphan}))
        (q05 / "audit.json").write_text(json.dumps({
            "independent_orphan_count": fake_orphan,
            "tool_orphan_count": fake_orphan,
            "matches_tool": True,
        }))
        res = await _run_one("battery04_q05", "q05_orphan_cross_check", _Q05_DELIVERABLES)
        assert res.passed is False
        # The recomputation guard fired (not just the auditor-agreement gate).
        assert "recomputed" in res.evidence.get("stdout", "").lower()

    @pytest.mark.asyncio
    async def test_non_utc_timestamp_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        _build_q05_good(root)
        q05 = root / "q05"
        text = (q05 / "reconciled.csv").read_text()
        # Strip the trailing Z on one timestamp → not UTC.
        (q05 / "reconciled.csv").write_text(text.replace("2026-01-01T10:00:00Z", "2026-01-01T10:00:00", 1))
        res = await _run_one("battery04_q05", "q05_utc_conformance_reconciled", _Q05_DELIVERABLES)
        assert res.passed is False

    @pytest.mark.asyncio
    async def test_utc_check_recognizes_date_named_column(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Regression: the agent named the temporal column ``order_date`` (a
        # reasonable name). The UTC probe's column detector must recognize
        # date-/day-named columns (q06-consistent), not just timestamp/_at/time
        # — otherwise correct Z-formatted output fails with "no timestamp column".
        root = _patch_roots(monkeypatch, tmp_path)
        q05 = root / "q05"
        q05.mkdir(parents=True, exist_ok=True)
        (q05 / "reconciled.csv").write_text(
            "order_id,customer_id,order_date,amount\n"
            "ORD-1,alice,2024-12-11T01:03:37Z,10.0\n"
            "ORD-2,bob,2024-06-04T02:24:06Z,20.0\n"
        )
        res = await _run_one("battery04_q05", "q05_utc_conformance_reconciled", _Q05_DELIVERABLES)
        assert res.passed is True
        assert "ok" in res.evidence.get("stdout", "").lower()

    @pytest.mark.asyncio
    async def test_surviving_duplicate_customer_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        _build_q05_good(root)
        q05 = root / "q05"
        import csv as _csv

        with open(q05 / "reconciled.csv", newline="") as f:
            rows = list(_csv.DictReader(f))
        # Inject a conflicting attribute for an existing canonical customer.
        rows.append({**rows[0], "customer_name": "Conflicting Name"})
        with open(q05 / "reconciled.csv", "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        res = await _run_one("battery04_q05", "q05_no_duplicate_customers_survive", _Q05_DELIVERABLES)
        assert res.passed is False


# --------------------------------------------------------------------------
# q06 fixture builders
# --------------------------------------------------------------------------

_Q06_DELIVERABLES = [
    "results/q06/daily_rollup.csv",
    "results/q06/run_manifest.json",
    "results/q06/daily_rollup_v2.csv",
    "results/q06/test_results.json",
    "results/q06/capstone_summary.md",
]


def _build_q06_good(root: Path) -> dict:
    """Correct end-to-end q06 fixture (also writes a q05 reconciled.csv input)."""
    q05 = root / "q05"
    q06 = root / "q06"
    q05.mkdir(parents=True, exist_ok=True)
    q06.mkdir(parents=True, exist_ok=True)
    # q06 reads q05/reconciled.csv as its rollup input.
    recon_rows = [
        {"date": "2026-01-01", "customer_id": "c001", "revenue": "10.0"},
        {"date": "2026-01-01", "customer_id": "c002", "revenue": "5.0"},
        {"date": "2026-01-02", "customer_id": "c001", "revenue": "7.0"},
    ]
    _write_csv(q05 / "reconciled.csv", recon_rows)

    rollup_rows = [
        {"date": "2026-01-01", "order_count": "2", "revenue_sum": "15.0", "unique_customers": "2"},
        {"date": "2026-01-02", "order_count": "1", "revenue_sum": "7.0", "unique_customers": "1"},
    ]
    _write_csv(q06 / "daily_rollup.csv", rollup_rows)

    rollup_bytes = (q06 / "daily_rollup.csv").read_bytes()
    recon_bytes = (q05 / "reconciled.csv").read_bytes()
    manifest = {
        "run_count": 1,
        "input_sha256": hashlib.sha256(recon_bytes).hexdigest(),
        "output_sha256": hashlib.sha256(rollup_bytes).hexdigest(),
        "is_idempotent": True,
    }
    (q06 / "run_manifest.json").write_text(json.dumps(manifest))

    # v2 adds a rolling-average column (more columns than v1).
    v2_rows = [
        {**r, "revenue_7day_avg": "11.0"} for r in rollup_rows
    ]
    _write_csv(q06 / "daily_rollup_v2.csv", v2_rows)

    (q06 / "test_results.json").write_text(json.dumps({"passed": 4, "failed": 0}))
    (q06 / "capstone_summary.md").write_text(
        "# Capstone\n\nThe rollup is idempotent and evolution added a rolling average.\n"
    )
    return manifest


class TestQ06Probes:
    @pytest.mark.asyncio
    async def test_good_fixture_all_checks_pass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        _build_q06_good(root)
        result = await run_checks(GOLDEN_SPECS["battery04_q06"], _Q06_DELIVERABLES, {})
        failed = [c.check_name for c in result.checks if not c.passed and not c.skipped]
        assert result.passed is True, f"failed checks: {failed}"
        assert result.overall_score == 1.0

    @pytest.mark.asyncio
    async def test_idempotency_transform_passes_on_valid_input(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The NEW idempotency family consumed by q06: reference rollup transform
        # over the q05 reconciled input is deterministic + non-empty.
        root = _patch_roots(monkeypatch, tmp_path)
        _build_q06_good(root)
        res = await _run_one("battery04_q06", "q06_rollup_deterministic", _Q06_DELIVERABLES)
        assert res.passed is True, res.evidence
        assert res.evidence["identical"] is True

    @pytest.mark.asyncio
    async def test_fabricated_idempotent_flag_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        manifest = _build_q06_good(root)
        q06 = root / "q06"
        # Claim id_idempotent but report a sha256 that does NOT match the disk file.
        (q06 / "run_manifest.json").write_text(json.dumps({
            **manifest,
            "is_idempotent": True,
            "output_sha256": hashlib.sha256(b"not the rollup").hexdigest(),
        }))
        res = await _run_one("battery04_q06", "q06_manifest_idempotent", _Q06_DELIVERABLES)
        assert res.passed is False

    @pytest.mark.asyncio
    async def test_manifest_input_sha256_matches_via_membership_with_stray(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Regression (harness bug #4): the manifest.input_sha256 honestly covers
        # the REAL results/q05/reconciled.csv, but the agent ALSO re-reconciled
        # into its own subdir (a stray results/q06/q05/reconciled.csv) with
        # DIFFERENT bytes. The OLD first-match probe grabbed whichever
        # reconciled.csv os.walk hit first — non-deterministic, so it could land
        # on the stray and false-reject an honest manifest. The membership fix
        # walks ALL reconciled.csv under _RESULTS_ROOT and passes if the manifest
        # matches ANY; this must PASS.
        root = _patch_roots(monkeypatch, tmp_path)
        manifest = _build_q06_good(root)
        real_recon = root / "q05" / "reconciled.csv"
        assert manifest["input_sha256"] == hashlib.sha256(real_recon.read_bytes()).hexdigest()
        # Stray reconciled.csv under q06's own subdir with different bytes/hash.
        stray = root / "q06" / "q05" / "reconciled.csv"
        stray.parent.mkdir(parents=True, exist_ok=True)
        stray.write_text("date,customer_id,revenue\n2099-12-31,zzz,0.0\n")
        assert hashlib.sha256(stray.read_bytes()).hexdigest() != manifest["input_sha256"]
        res = await _run_one("battery04_q06", "q06_manifest_idempotent", _Q06_DELIVERABLES)
        assert res.passed is True, res.evidence

    @pytest.mark.asyncio
    async def test_manifest_input_sha256_matching_nothing_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Negative control: the membership fix must NOT weaken the
        # anti-fabrication intent. A manifest whose input_sha256 matches NO
        # reconciled.csv anywhere on disk is a fabrication and must still FAIL.
        root = _patch_roots(monkeypatch, tmp_path)
        manifest = _build_q06_good(root)
        q06 = root / "q06"
        (q06 / "run_manifest.json").write_text(json.dumps({
            **manifest,
            "input_sha256": hashlib.sha256(b"not any reconciled.csv on disk").hexdigest(),
        }))
        res = await _run_one("battery04_q06", "q06_manifest_idempotent", _Q06_DELIVERABLES)
        assert res.passed is False
        assert "input_sha256" in res.evidence.get("stdout", "").lower()

    @pytest.mark.asyncio
    async def test_test_results_stub_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        _build_q06_good(root)
        (root / "q06" / "test_results.json").write_text(json.dumps({"status": "failed"}))
        res = await _run_one("battery04_q06", "q06_test_results_real_counts", _Q06_DELIVERABLES)
        assert res.passed is False

    @pytest.mark.asyncio
    async def test_placeholder_leak_in_summary_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        _build_q06_good(root)
        (root / "q06" / "capstone_summary.md").write_text(
            "# Capstone\n\nThe {name} produced {count} rows.\n"
        )
        res = await _run_one("battery04_q06", "q06_no_placeholder_summary", _Q06_DELIVERABLES)
        assert res.passed is False

    @pytest.mark.asyncio
    async def test_v2_without_rolling_avg_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        _build_q06_good(root)
        # Overwrite v2 with the v1 schema (no rolling/avg column).
        import shutil

        shutil.copy(root / "q06" / "daily_rollup.csv", root / "q06" / "daily_rollup_v2.csv")
        res = await _run_one("battery04_q06", "q06_v2_has_rolling_avg", _Q06_DELIVERABLES)
        assert res.passed is False

    @pytest.mark.asyncio
    async def test_unparseable_date_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        _build_q06_good(root)
        _write_csv(root / "q06" / "daily_rollup.csv", [
            {"date": "not-a-date", "order_count": "2", "revenue_sum": "15.0", "unique_customers": "2"},
        ])
        res = await _run_one("battery04_q06", "q06_utc_dates", _Q06_DELIVERABLES)
        assert res.passed is False
