"""Golden GoalSpecs for the Battery-04 validation suite (Phase 3 / Phase 10).

Each spec pairs a goal with machine-verifiable correctness checks. The checks
are deliberately schema-light but real — the exact column/key names an agent
generates aren't known a priori, so the gate asserts what we *can* assert
(existence, format validity, non-emptiness, JSON validity, stable invariants)
rather than a brittle fixed schema. A run associates itself with a spec via
``state["eval_goal_spec_id"]``; ``lookup_goal_spec`` resolves it for the verify
node and the ``--eval`` suite.

The deeper per-deliverable invariants (UTC conformance, zero-duplicates,
retention sums) are exercised end-to-end by query q4's generated pytest suite;
these golden checks are the deterministic regression gate (e.g. the Phase-8
evolution canary) that must not flake on agent output variance.
"""

from __future__ import annotations

from src.eval.models import CheckConfig, GoalSpec


def _battery04_q01() -> GoalSpec:
    """Ingestion + normalization tool → normalized.csv + summary.json."""
    return GoalSpec(
        spec_id="battery04_q01",
        name="battery04_q01",
        description="Ingest + dedupe + canonicalize an e-commerce event JSONL to normalized CSV + summary JSON",
        goal_text=(
            "Generate a denormalized e-commerce event JSONL (orders, returns, page-views) "
            "seeded with inconsistencies (mixed-case IDs, tz-naive timestamps, duplicate "
            "events, malformed rows). Create a tool that ingests, deduplicates, canonicalizes "
            "IDs, coerces types, and converts timestamps to UTC ISO-8601, writing "
            "results/q01/normalized.csv and results/q01/summary.json."
        ),
        category="tool_creation",
        max_iterations=60,
        timeout_seconds=900,
        expected_deliverables=["results/q01/normalized.csv", "results/q01/summary.json"],
        success_criteria=[
            "results/q01/normalized.csv exists and is a valid CSV with >= 1 data row",
            "results/q01/summary.json exists and is valid JSON",
            "No duplicate event rows remain after normalization",
            "All timestamps are UTC ISO-8601",
        ],
        checks=[
            CheckConfig(
                check_type="golden",
                name="q01_deliverables_exist",
                params={
                    "assertions": [
                        {"kind": "exists", "deliverable": "results/q01/normalized.csv"},
                        {"kind": "exists", "deliverable": "results/q01/summary.json"},
                    ]
                },
            ),
            CheckConfig(
                check_type="structural",
                name="q01_normalized_csv_schema",
                params={
                    "deliverable": "results/q01/normalized.csv",
                    "format": "csv",
                    "min_rows": 1,
                },
            ),
            CheckConfig(
                check_type="structural",
                name="q01_summary_json_schema",
                params={
                    "deliverable": "results/q01/summary.json",
                    "format": "json",
                },
            ),
            CheckConfig(
                check_type="execution",
                name="q01_no_duplicates_probe",
                params={
                    "code": (
                        "import csv, json as _j, sys\n"
                        "rows = []\n"
                        "path = next((p for p in _DELIVERABLES if p.endswith('normalized.csv')), '')\n"
                        "if not path:\n"
                        "    print('no normalized.csv'); sys.exit(1)\n"
                        "with open(path, newline='') as _f:\n"
                        "    for _r in csv.DictReader(_f):\n"
                        "        rows.append(tuple(sorted(_r.items())))\n"
                        "dups = len(rows) - len(set(rows))\n"
                        "if dups > 0:\n"
                        "    print(f'{dups} duplicate rows remain'); sys.exit(1)\n"
                        "print(f'ok: {len(rows)} unique rows')\n"
                    ),
                    "timeout": 20,
                },
            ),
            CheckConfig(
                check_type="execution",
                name="q01_utc_conformance_probe",
                params={
                    # Catches the q1 regression: a normalizer that drops
                    # negative-offset timestamps to empty cells (the file is
                    # valid CSV + has the timestamp column, so the structural
                    # schema check passes — only this semantic probe notices
                    # the cells are empty / not UTC). Every timestamp must be
                    # non-empty, end with a trailing Z, and parse as ISO-8601.
                    "code": (
                        "import csv, sys\n"
                        "from datetime import datetime\n"
                        "path = next((p for p in _DELIVERABLES if p.endswith('normalized.csv')), '')\n"
                        "if not path:\n"
                        "    print('no normalized.csv'); sys.exit(1)\n"
                        "bad = []\n"
                        "count = 0\n"
                        "with open(path, newline='') as _f:\n"
                        "    reader = csv.DictReader(_f)\n"
                        "    cols = reader.fieldnames or []\n"
                        "    tcol = next((c for c in cols if 'timestamp' in c.lower() or c.lower() in ('time', 'ts', 'utc')), None)\n"
                        "    if tcol is None:\n"
                        "        print('no timestamp column in normalized.csv'); sys.exit(1)\n"
                        "    for _r in reader:\n"
                        "        count += 1\n"
                        "        val = (_r.get(tcol) or '').strip()\n"
                        "        if not val:\n"
                        "            bad.append('empty timestamp at row ' + str(count)); continue\n"
                        "        if not val.endswith('Z'):\n"
                        "            bad.append(repr(val) + ' at row ' + str(count) + ' has no trailing Z'); continue\n"
                        "        try:\n"
                        "            datetime.fromisoformat(val.replace('Z', '+00:00'))\n"
                        "        except ValueError:\n"
                        "            bad.append(repr(val) + ' at row ' + str(count) + ' not parseable ISO-8601')\n"
                        "if count == 0:\n"
                        "    print('no data rows in normalized.csv'); sys.exit(1)\n"
                        "if bad:\n"
                        "    print('UTC non-conformance: ' + '; '.join(bad[:5])); sys.exit(1)\n"
                        "print('ok: all ' + str(count) + ' timestamps are UTC ISO-8601')\n"
                    ),
                    "timeout": 20,
                },
            ),
            CheckConfig(
                check_type="execution",
                name="q01_timestamp_instant_preservation",
                params={
                    # Catches *value* errors the format-only probe cannot: a
                    # normalizer must never change an instant. We match each
                    # normalized row to its raw source by event_id (case-insensitive),
                    # parse both timestamps to an aware UTC instant (tz-naive raw
                    # timestamps are assumed UTC — the goal's "UTC-assumed" rule),
                    # and require equality. This flags a -1h shift on tz-naive rows
                    # (run-2 bug: 12:00:00 -> 11:00:00Z) AND empty cells (run-1 bug:
                    # negative offsets dropped -> unparseable). Rows with no raw
                    # counterpart (legitimately dropped malformed rows) are skipped,
                    # so the check never false-positives on valid dedup/drops.
                    "code": (
                        "import csv, json as _j, os, sys\n"
                        "from datetime import datetime, timezone\n"
                        "\n"
                        "def _to_utc(s):\n"
                        "    s = (s or '').strip()\n"
                        "    if not s:\n"
                        "        return None\n"
                        "    if s.endswith('Z'):\n"
                        "        s = s[:-1] + '+00:00'\n"
                        "    try:\n"
                        "        dt = datetime.fromisoformat(s)\n"
                        "    except ValueError:\n"
                        "        return None\n"
                        "    if dt.tzinfo is None:\n"
                        "        dt = dt.replace(tzinfo=timezone.utc)\n"
                        "    return dt.astimezone(timezone.utc)\n"
                        "\n"
                        "norm = next((p for p in _DELIVERABLES if p.endswith('normalized.csv')), '')\n"
                        "raw = next((p for p in _DELIVERABLES if p.endswith('raw_events.jsonl')), '')\n"
                        "if not raw and _RESULTS_ROOT:\n"
                        "    for _c in (os.path.join(_RESULTS_ROOT, 'q01', 'raw_events.jsonl'),\n"
                        "               os.path.join(_RESULTS_ROOT, 'raw_events.jsonl')):\n"
                        "        if os.path.exists(_c):\n"
                        "            raw = _c\n"
                        "            break\n"
                        "if not raw and _RESULTS_ROOT:\n"
                        "    for _d, _, _fs in os.walk(_RESULTS_ROOT):\n"
                        "        if 'raw_events.jsonl' in _fs:\n"
                        "            raw = os.path.join(_d, 'raw_events.jsonl')\n"
                        "            break\n"
                        "if not norm:\n"
                        "    print('no normalized.csv'); sys.exit(1)\n"
                        "if not raw:\n"
                        "    print('no raw_events.jsonl found (cannot verify instant preservation)'); sys.exit(1)\n"
                        "\n"
                        "raw_inst = {}\n"
                        "with open(raw) as _f:\n"
                        "    for _line in _f:\n"
                        "        _line = _line.strip()\n"
                        "        if not _line:\n"
                        "            continue\n"
                        "        try:\n"
                        "            _o = _j.loads(_line)\n"
                        "        except ValueError:\n"
                        "            continue\n"
                        "        if not isinstance(_o, dict):\n"
                        "            continue\n"
                        "        eid = str(_o.get('event_id') or _o.get('id') or '').strip().lower()\n"
                        "        ts = _o.get('timestamp') or _o.get('ts') or _o.get('time') or ''\n"
                        "        if not eid:\n"
                        "            continue\n"
                        "        inst = _to_utc(str(ts))\n"
                        "        if inst is not None:\n"
                        "            raw_inst.setdefault(eid, inst)\n"
                        "\n"
                        "bad = []\n"
                        "checked = 0\n"
                        "with open(norm, newline='') as _f:\n"
                        "    reader = csv.DictReader(_f)\n"
                        "    cols = reader.fieldnames or []\n"
                        "    eid_col = next((c for c in cols if c.lower() in ('event_id', 'id')), None)\n"
                        "    tcol = next((c for c in cols if 'timestamp' in c.lower() or c.lower() in ('time', 'ts', 'utc')), None)\n"
                        "    if eid_col is None or tcol is None:\n"
                        "        print('normalized.csv missing event_id or timestamp column'); sys.exit(1)\n"
                        "    for _r in reader:\n"
                        "        eid = str(_r.get(eid_col) or '').strip().lower()\n"
                        "        rinst = raw_inst.get(eid)\n"
                        "        if rinst is None:\n"
                        "            continue\n"
                        "        checked += 1\n"
                        "        ninst = _to_utc(str(_r.get(tcol) or ''))\n"
                        "        if ninst is None:\n"
                        "            bad.append(eid + ': normalized timestamp empty/unparseable')\n"
                        "        elif ninst != rinst:\n"
                        "            bad.append(eid + ': normalized ' + ninst.isoformat().replace('+00:00', 'Z') + ' != raw ' + rinst.isoformat().replace('+00:00', 'Z') + ' (instant changed)')\n"
                        "\n"
                        "if checked == 0:\n"
                        "    print('no normalized rows matched a raw event_id; cannot verify'); sys.exit(1)\n"
                        "if bad:\n"
                        "    print('timestamp instant not preserved: ' + '; '.join(bad[:5])); sys.exit(1)\n"
                        "print('ok: all ' + str(checked) + ' normalized timestamps preserve the raw UTC instant')\n"
                    ),
                    "timeout": 20,
                },
            ),
            CheckConfig(
                check_type="execution",
                name="q01_record_count_probe",
                params={
                    # summary.json.record_count must equal the CSV's data-row count.
                    # Catches the run-2 count bug (summary=24 vs 25 rows).
                    "code": (
                        "import csv, json as _j, sys\n"
                        "norm = next((p for p in _DELIVERABLES if p.endswith('normalized.csv')), '')\n"
                        "summ = next((p for p in _DELIVERABLES if p.endswith('summary.json')), '')\n"
                        "if not norm or not summ:\n"
                        "    print('missing normalized.csv or summary.json'); sys.exit(1)\n"
                        "with open(norm, newline='') as _f:\n"
                        "    rows = sum(1 for _ in csv.DictReader(_f))\n"
                        "try:\n"
                        "    data = _j.load(open(summ))\n"
                        "except ValueError:\n"
                        "    print('summary.json is invalid JSON'); sys.exit(1)\n"
                        "declared = data.get('record_count') if isinstance(data, dict) else None\n"
                        "if declared is None:\n"
                        "    print('summary.json has no record_count'); sys.exit(1)\n"
                        "if int(declared) != rows:\n"
                        "    print('record_count mismatch: summary=' + str(declared) + ' vs csv rows=' + str(rows)); sys.exit(1)\n"
                        "print('ok: record_count ' + str(declared) + ' == csv rows ' + str(rows))\n"
                    ),
                    "timeout": 20,
                },
            ),
        ],
    )


def _battery04_q02() -> GoalSpec:
    """Data-quality auditor sub-agent → quality_report.md + scorecard.json."""
    return GoalSpec(
        spec_id="battery04_q02",
        name="battery04_q02",
        description="Spawn a data-quality auditor sub-agent that scores q01 output and writes a methodology report",
        goal_text=(
            "Spawn a data-quality auditor sub-agent that audits results/q01/normalized.csv against "
            "the source, computes a scorecard (completeness, uniqueness, validity, timeliness, each "
            "0-100 with evidence), flags the top anomalies, and writes a reproducible-methodology "
            "Markdown report to results/q02/quality_report.md and results/q02/scorecard.json."
        ),
        category="multi_agent",
        max_iterations=60,
        timeout_seconds=900,
        expected_deliverables=["results/q02/quality_report.md", "results/q02/scorecard.json"],
        success_criteria=[
            "results/q02/quality_report.md exists and documents the methodology",
            "results/q02/scorecard.json exists, is valid JSON, and has >= 1 dimension score",
        ],
        checks=[
            CheckConfig(
                check_type="golden",
                name="q02_deliverables_exist",
                params={
                    "assertions": [
                        {"kind": "exists", "deliverable": "results/q02/quality_report.md"},
                        {"kind": "exists", "deliverable": "results/q02/scorecard.json"},
                    ]
                },
            ),
            CheckConfig(
                check_type="golden",
                name="q02_report_has_methodology",
                params={
                    "assertions": [
                        {
                            "kind": "regex",
                            "deliverable": "results/q02/quality_report.md",
                            "pattern": "(?i)method|approach|how|step",
                        }
                    ]
                },
            ),
            CheckConfig(
                check_type="execution",
                name="q02_report_no_placeholder_leak",
                params={
                    # F-j: a report that ships with unsubstituted template
                    # placeholders ({var}/{{var}}) in prose is not a usable
                    # deliverable, yet it is present, non-empty, and matches the
                    # lenient methodology regex — so without this probe the
                    # harness scored it 1.000 while quality_report.md carried 13
                    # "{uniqueness_pct}"-style tokens. Fenced/inline code is
                    # stripped first so f-strings/JSON shown inside code don't
                    # false-positive. >=2 distinct residues in prose fails.
                    "code": (
                        "import re, sys\n"
                        "path = next((p for p in _DELIVERABLES "
                        "if p.endswith('quality_report.md')), '')\n"
                        "if not path:\n"
                        "    print('no quality_report.md'); sys.exit(1)\n"
                        "t = open(path, encoding='utf-8', "
                        "errors='replace').read()\n"
                        "t = re.sub(r'```.*?```', '', t, flags=re.DOTALL)\n"
                        "t = re.sub(r'`[^`\\n]*`', '', t)\n"
                        "fmt = set(re.findall("
                        "r'(?<!\\{)\\{([a-z_][a-z0-9_]{2,})\\}(?!\\})', t))\n"
                        "jv = set(re.findall("
                        "r'\\{\\{\\s*([a-zA-Z_][a-zA-Z0-9_.-]*)\\s*\\}\\}', t))\n"
                        "leaks = fmt | jv\n"
                        "if len(leaks) >= 2:\n"
                        "    print('placeholder leak: ' + "
                        "', '.join(sorted(leaks)[:6])); sys.exit(1)\n"
                        "print('ok: no template-placeholder residue in report')\n"
                    ),
                    "timeout": 20,
                },
            ),
            CheckConfig(
                check_type="structural",
                name="q02_scorecard_json_schema",
                params={"deliverable": "results/q02/scorecard.json", "format": "json"},
            ),
            CheckConfig(
                check_type="execution",
                name="q02_scorecard_numeric_probe",
                params={
                    "code": (
                        "import json as _j, sys\n"
                        "def _flatten(o):\n"
                        "    if isinstance(o, dict):\n"
                        "        for _v in o.values():\n"
                        "            yield from _flatten(_v)\n"
                        "    elif isinstance(o, list):\n"
                        "        for _v in o:\n"
                        "            yield from _flatten(_v)\n"
                        "    else:\n"
                        "        yield o\n"
                        "path = next((p for p in _DELIVERABLES if p.endswith('scorecard.json')), '')\n"
                        "if not path:\n"
                        "    print('no scorecard.json'); sys.exit(1)\n"
                        "data = _j.load(open(path))\n"
                        "scores = [v for v in _flatten(data) if isinstance(v, (int, float))]\n"
                        "if not scores:\n"
                        "    print('no numeric scores found'); sys.exit(1)\n"
                        "print(f'ok: {len(scores)} numeric scores')\n"
                    ),
                    "timeout": 20,
                },
            ),
        ],
    )


def _battery04_q03() -> GoalSpec:
    """Cohort retention tool + evolution → retention.csv + churn_flags.csv."""
    return GoalSpec(
        spec_id="battery04_q03",
        name="battery04_q03",
        description="Create a cohort retention tool over q01 data, then evolve it to emit per-cohort churn-risk flags",
        goal_text=(
            "Reuse the q01 normalizer. Create a cohort retention tool (weekly cohorts, retention "
            "matrix) over results/q01/normalized.csv, writing results/q03/retention.csv. Then evolve "
            "that tool (TOOL/CODE mutation via the codegen model + safety + sandbox + post-deploy "
            "verify) to also emit a per-cohort churn-risk flag at a configurable threshold, writing "
            "results/q03/churn_flags.csv."
        ),
        category="tool_creation",
        max_iterations=60,
        timeout_seconds=1200,
        expected_deliverables=["results/q03/retention.csv", "results/q03/churn_flags.csv"],
        success_criteria=[
            "results/q03/retention.csv exists with >= 1 cohort row",
            "results/q03/churn_flags.csv exists (evolved capability)",
            "A tool mutation was deployed during this run",
        ],
        checks=[
            CheckConfig(
                check_type="golden",
                name="q03_deliverables_exist",
                params={
                    "assertions": [
                        {"kind": "exists", "deliverable": "results/q03/retention.csv"},
                        {"kind": "exists", "deliverable": "results/q03/churn_flags.csv"},
                    ]
                },
            ),
            CheckConfig(
                check_type="structural",
                name="q03_retention_csv_schema",
                params={
                    "deliverable": "results/q03/retention.csv",
                    "format": "csv",
                    "min_rows": 1,
                },
            ),
            CheckConfig(
                check_type="structural",
                name="q03_churn_flags_csv_schema",
                params={
                    "deliverable": "results/q03/churn_flags.csv",
                    "format": "csv",
                    "min_rows": 1,
                },
            ),
        ],
    )


def _battery04_q04() -> GoalSpec:
    """Capstone orchestrator sub-agent → test_results.json + executive_summary.md."""
    return GoalSpec(
        spec_id="battery04_q04",
        name="battery04_q04",
        description="Spawn an orchestrator sub-agent that builds+runs a pytest suite over q1-q3 and writes an executive summary",
        goal_text=(
            "Spawn an orchestrator sub-agent that builds and runs a pytest suite validating q1-q3 "
            "outputs (schema invariants, retention sums within tolerance, no nulls in required "
            "fields, UTC conformance) in the sandbox, writing results/q04/test_results.json, then "
            "writes an executive Markdown synthesizing q1-q3 into business KPIs to "
            "results/q04/executive_summary.md."
        ),
        category="multi_agent",
        max_iterations=60,
        timeout_seconds=1200,
        expected_deliverables=["results/q04/test_results.json", "results/q04/executive_summary.md"],
        success_criteria=[
            "results/q04/test_results.json exists and records pass/fail counts",
            "results/q04/executive_summary.md exists and is non-empty",
        ],
        checks=[
            CheckConfig(
                check_type="golden",
                name="q04_deliverables_exist",
                params={
                    "assertions": [
                        {"kind": "exists", "deliverable": "results/q04/test_results.json"},
                        {"kind": "exists", "deliverable": "results/q04/executive_summary.md"},
                    ]
                },
            ),
            CheckConfig(
                check_type="structural",
                name="q04_test_results_json_schema",
                params={"deliverable": "results/q04/test_results.json", "format": "json"},
            ),
            CheckConfig(
                check_type="execution",
                name="q04_test_results_has_counts",
                params={
                    "code": (
                        "import json as _j, sys\n"
                        "path = next((p for p in _DELIVERABLES if p.endswith('test_results.json')), '')\n"
                        "if not path:\n"
                        "    print('no test_results.json'); sys.exit(1)\n"
                        "data = _j.load(open(path))\n"
                        "blob = _j.dumps(data).lower()\n"
                        "if 'pass' not in blob and 'fail' not in blob:\n"
                        "    print('no pass/fail counts recorded'); sys.exit(1)\n"
                        "print('ok: test results record pass/fail')\n"
                    ),
                    "timeout": 20,
                },
            ),
        ],
    )


# Registry keyed by spec_id — the verify node + --eval suite resolve a run's spec
# via state["eval_goal_spec_id"] → lookup_goal_spec(spec_id).
GOLDEN_SPECS: dict[str, GoalSpec] = {
    spec.spec_id: spec
    for spec in (
        _battery04_q01(),
        _battery04_q02(),
        _battery04_q03(),
        _battery04_q04(),
    )
}

# Ordered suite for `python main.py --eval`.
BATTERY04_GOALS: list[GoalSpec] = [
    GOLDEN_SPECS["battery04_q01"],
    GOLDEN_SPECS["battery04_q02"],
    GOLDEN_SPECS["battery04_q03"],
    GOLDEN_SPECS["battery04_q04"],
]


def lookup_goal_spec(spec_id: str | None) -> GoalSpec | None:
    """Resolve a GoalSpec by id, or None if unregistered / empty."""
    if not spec_id:
        return None
    return GOLDEN_SPECS.get(spec_id)
