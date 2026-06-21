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
            "results/q04/executive_summary.md. Write the test file to results/q04/ and invoke "
            "pytest by its EXACT written path (e.g. pytest results/q04/test_q1_q3.py) — the "
            "filename passed to pytest must match the file you wrote. results/q04/test_results.json "
            "MUST record concrete integer pass/fail counts parsed from pytest's summary line "
            "(e.g. {\"passed\": N, \"failed\": M}); a {\"status\": \"failed\"} or {\"status\": "
            "\"error\"} placeholder is INVALID — if a test module fails to import, the filename "
            "or path is wrong: fix it and re-run so the recorded file always carries real counts."
        ),
        category="multi_agent",
        max_iterations=60,
        timeout_seconds=1200,
        expected_deliverables=["results/q04/test_results.json", "results/q04/executive_summary.md"],
        success_criteria=[
            "results/q04/test_results.json exists and records concrete integer pass/fail counts",
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
                    # Reject failure/error STUBS (e.g. {"status": "failed", ...}) —
                    # the previous 'pass'/'fail' substring test passed such stubs
                    # because "failed" contains "fail". Require concrete INTEGER
                    # pass/fail counts at the top level OR under a summary/totals
                    # node (pytest-json-report shape), so a run that never actually
                    # executed its suite cannot pass verification.
                    "code": (
                        "import json as _j, sys\n"
                        "path = next((p for p in _DELIVERABLES if p.endswith('test_results.json')), '')\n"
                        "if not path:\n"
                        "    print('no test_results.json'); sys.exit(1)\n"
                        "try:\n"
                        "    data = _j.load(open(path))\n"
                        "except Exception as _e:\n"
                        "    print('unparseable test_results.json:', _e); sys.exit(1)\n"
                        "# battery-04 q4 qwen: a failure stub can hide under ANY status-like\n"
                        "# top-level key, not just 'status'. qwen wrote {overall_status:\n"
                        "# 'failed', summary:{passed:0,failed:0,errors:1}, ...} and the old\n"
                        "# data.get('status') guard never fired. Scan every common key.\n"
                        "_STUB = {'failed', 'error', 'fail', 'errored', 'crashed', 'aborted'}\n"
                        "_status = ''\n"
                        "if isinstance(data, dict):\n"
                        "    for _k in ('status', 'overall_status', 'result', 'outcome'):\n"
                        "        _v = str(data.get(_k, '')).lower().strip()\n"
                        "        if _v in _STUB:\n"
                        "            _status = _v\n"
                        "            break\n"
                        "if _status:\n"
                        "    print('rejected: test_results.json is a failure stub (' + _status + ')'); sys.exit(1)\n"
                        "_COUNT_KEYS = ('passed', 'failed', 'errors', 'skipped', 'total', 'collected', 'pass', 'fail')\n"
                        "def _ints(d):\n"
                        "    out = {}\n"
                        "    if isinstance(d, dict):\n"
                        "        for k, v in d.items():\n"
                        "            if k in _COUNT_KEYS:\n"
                        "                try:\n"
                        "                    f = float(v)\n"
                        "                except (TypeError, ValueError):\n"
                        "                    continue\n"
                        "                if f == int(f):\n"
                        "                    out[k] = int(f)\n"
                        "    return out\n"
                        "counts = _ints(data)\n"
                        "if not counts and isinstance(data, dict):\n"
                        "    for _sub in (data.get('summary'), data.get('totals'), data.get('stats')):\n"
                        "        counts = _ints(_sub)\n"
                        "        if counts:\n"
                        "            break\n"
                        "if not counts:\n"
                        "    print('no integer pass/fail counts recorded'); sys.exit(1)\n"
                        "_pass = any(k in counts for k in ('passed', 'pass'))\n"
                        "_fail = any(k in counts for k in ('failed', 'fail', 'errors'))\n"
                        "if not (_pass or _fail):\n"
                        "    print('counts present but neither pass nor fail recorded'); sys.exit(1)\n"
                        "# battery-04 q4 run2 wrote {\"passed\":0,\"failed\":0} (a pandas .dt.tz\n"
                        "# bug meant pytest collected nothing); qwen later wrote {passed:0,\n"
                        "# failed:0, errors:1} — a collection/import FAILURE dressed up as a\n"
                        "# result. Both validated nothing. 'errors' is a setup/collection\n"
                        "# outcome, NOT a test that ran to a verdict, and total/collected\n"
                        "# alone don't prove execution either. Require at least one concrete\n"
                        "# PASS or FAIL verdict so a stub, a hollow 0/0, AND an all-error\n"
                        "# collection failure all fail verification.\n"
                        "_verdicts = counts.get('passed',0)+counts.get('pass',0)+counts.get('failed',0)+counts.get('fail',0)\n"
                        "if _verdicts <= 0:\n"
                        "    print('no tests reached a pass/fail verdict (only errors/zero — suite did not execute)'); sys.exit(1)\n"
                        "print('ok: test results record counts ' + _j.dumps(counts))\n"
                    ),
                    "timeout": 20,
                },
            ),
        ],
    )


def _battery04_q05() -> GoalSpec:
    """Multi-source reconciliation + referential integrity + independent auditor.

    New dimension vs q01–q04: referential integrity across TWO related sources
    (orders→customers) plus a *cross-agent agreement* gate — an auditor
    sub-agent re-derives the orphan count independently, and the execution probe
    recomputes it a third time from the RAW sources. Three independent counts
    must agree, so an agent that fabricates ``integrity_report.json`` numbers
    (the O3 "demo-data" failure mode) is caught deterministically.
    """
    return GoalSpec(
        spec_id="battery04_q05",
        name="battery04_q05",
        description="Reconcile two related datasets (orders→customers) with referential-integrity violations; auditor cross-checks the orphan count",
        goal_text=(
            "Generate two related synthetic datasets: results/q05/customers.jsonl "
            "(>=12 records with duplicate and mixed-case customer_id values) and "
            "results/q05/orders.jsonl (>=40 records), SEED with referential-integrity "
            "violations: orders referencing nonexistent customers (orphans), mixed-case "
            "foreign keys that must canonicalize to a real customer, tz-naive timestamps, "
            "and duplicate order events. CREATE a reconciliation tool that (a) "
            "canonicalizes customer_id on both sides (lowercase, trimmed), (b) "
            "deduplicates customers and orders, (c) coerces all timestamps to UTC "
            "ISO-8601 with a trailing Z suffix (e.g. 2024-06-10T16:00:00Z — emit Z, "
            "NOT a +00:00 offset), (d) JOINs orders to customers on canonical "
            "customer_id, and "
            "(e) detects orphan orders. It writes results/q05/reconciled.csv (one row "
            "per matched order with customer fields joined) and results/q05/integrity_report.json "
            "with EXACTLY these keys: total_orders, matched_orders, orphaned_orders, "
            "duplicate_customers_removed, all_timestamps_utc (bool). Then SPAWN an "
            "auditor sub-agent that INDEPENDENTLY reads the RAW customers.jsonl + "
            "orders.jsonl, re-derives the orphan count by its own logic (no reuse of the "
            "reconciliation tool), and writes results/q05/audit.json with EXACTLY: "
            "independent_orphan_count (int), tool_orphan_count (copied from "
            "integrity_report.json), matches_tool (bool). Both orphan counts must agree."
        ),
        category="multi_agent",
        max_iterations=60,
        timeout_seconds=1200,
        expected_deliverables=[
            "results/q05/customers.jsonl",
            "results/q05/orders.jsonl",
            "results/q05/reconciled.csv",
            "results/q05/integrity_report.json",
            "results/q05/audit.json",
        ],
        success_criteria=[
            "results/q05/reconciled.csv exists with >= 1 joined order row",
            "results/q05/integrity_report.json exists with the 5 named keys",
            "Tool, auditor, and independent recompute all agree on the orphan count",
            "All timestamps in reconciled.csv are UTC ISO-8601 with a trailing Z",
        ],
        checks=[
            CheckConfig(
                check_type="golden",
                name="q05_deliverables_exist",
                params={
                    "assertions": [
                        {"kind": "exists", "deliverable": "results/q05/customers.jsonl"},
                        {"kind": "exists", "deliverable": "results/q05/orders.jsonl"},
                        {"kind": "exists", "deliverable": "results/q05/reconciled.csv"},
                        {"kind": "exists", "deliverable": "results/q05/integrity_report.json"},
                        {"kind": "exists", "deliverable": "results/q05/audit.json"},
                    ]
                },
            ),
            CheckConfig(
                check_type="structural",
                name="q05_reconciled_csv_schema",
                params={"deliverable": "results/q05/reconciled.csv", "format": "csv", "min_rows": 1},
            ),
            CheckConfig(
                check_type="structural",
                name="q05_integrity_report_schema",
                params={"deliverable": "results/q05/integrity_report.json", "format": "json"},
            ),
            CheckConfig(
                check_type="execution",
                name="q05_integrity_keys",
                params={
                    "code": (
                        "import json as _j, sys\n"
                        "report = next((p for p in _DELIVERABLES if p.endswith('integrity_report.json')), '')\n"
                        "if not report:\n"
                        "    print('no integrity_report.json'); sys.exit(1)\n"
                        "data = _j.load(open(report))\n"
                        "required = ['total_orders', 'matched_orders', 'orphaned_orders', "
                        "'duplicate_customers_removed', 'all_timestamps_utc']\n"
                        "missing = [k for k in required if k not in data]\n"
                        "if missing:\n"
                        "    print('missing keys: ' + ', '.join(missing)); sys.exit(1)\n"
                        "if not isinstance(data.get('all_timestamps_utc'), bool):\n"
                        "    print('all_timestamps_utc is not a bool'); sys.exit(1)\n"
                        "to = int(data['total_orders']); mt = int(data['matched_orders']); orph = int(data['orphaned_orders'])\n"
                        "if mt < 0 or orph < 0 or to < 0:\n"
                        "    print('negative count'); sys.exit(1)\n"
                        "if mt + orph > to + 5:\n"
                        "    print('matched+orphaned exceeds total_orders'); sys.exit(1)\n"
                        "print('ok: integrity_report keys present and sane')\n"
                    ),
                    "timeout": 20,
                },
            ),
            CheckConfig(
                check_type="execution",
                name="q05_no_duplicate_customers_survive",
                params={
                    # A duplicate customer that survived reconciliation would show
                    # the SAME canonical customer_id with CONFLICTING joined
                    # attribute values (name/email/...) across order rows. If every
                    # canonical id maps to exactly one attribute signature, the
                    # duplicates were collapsed. Requires the join to actually carry
                    # customer attribute columns (the goal mandates this).
                    "code": (
                        "import csv, sys\n"
                        "from collections import defaultdict\n"
                        "path = next((p for p in _DELIVERABLES if p.endswith('reconciled.csv')), '')\n"
                        "if not path:\n"
                        "    print('no reconciled.csv'); sys.exit(1)\n"
                        "groups = defaultdict(set)\n"
                        "with open(path, newline='') as _f:\n"
                        "    reader = csv.DictReader(_f)\n"
                        "    cols = reader.fieldnames or []\n"
                        "    cid_col = next((c for c in cols if 'customer_id' in c.lower() "
                        "or c.lower() in ('customerid', 'cust_id', 'cid')), None)\n"
                        "    if cid_col is None:\n"
                        "        print('no customer_id column in reconciled.csv'); sys.exit(1)\n"
                        "    attr_cols = [c for c in cols if c.lower().startswith('customer') and c != cid_col]\n"
                        "    if not attr_cols:\n"
                        "        print('no customer attribute columns joined (customer fields missing)'); sys.exit(1)\n"
                        "    for _r in reader:\n"
                        "        cid = str(_r.get(cid_col, '')).strip().lower()\n"
                        "        if not cid:\n"
                        "            continue\n"
                        "        sig = tuple(str(_r.get(a, '')).strip().lower() for a in attr_cols)\n"
                        "        groups[cid].add(sig)\n"
                        "inconsistent = {c: s for c, s in groups.items() if len(s) > 1}\n"
                        "if inconsistent:\n"
                        "    _ex = next(iter(inconsistent))\n"
                        "    print('customer ' + _ex + ' has ' + str(len(inconsistent[_ex])) "
                        "+ ' conflicting attribute sets (duplicate not collapsed)'); sys.exit(1)\n"
                        "print('ok: ' + str(len(groups)) + ' distinct customers, attributes consistent')\n"
                    ),
                    "timeout": 20,
                },
            ),
            CheckConfig(
                check_type="execution",
                name="q05_utc_conformance_reconciled",
                params={
                    # Every timestamp cell in the joined output must be non-empty,
                    # end with a trailing Z, and parse as ISO-8601 (the q01 pattern).
                    "code": (
                        "import csv, sys\n"
                        "from datetime import datetime\n"
                        "path = next((p for p in _DELIVERABLES if p.endswith('reconciled.csv')), '')\n"
                        "if not path:\n"
                        "    print('no reconciled.csv'); sys.exit(1)\n"
                        "bad = []; count = 0\n"
                        "with open(path, newline='') as _f:\n"
                        "    reader = csv.DictReader(_f)\n"
                        "    cols = reader.fieldnames or []\n"
                        "    tcols = [c for c in cols if 'timestamp' in c.lower() "
                        "or 'date' in c.lower() or 'day' in c.lower() "
                        "or c.lower().endswith('_at') or c.lower().endswith('time') "
                        "or c.lower() in ('ts', 'utc')]\n"
                        "    if not tcols:\n"
                        "        print('no timestamp column in reconciled.csv'); sys.exit(1)\n"
                        "    for _r in reader:\n"
                        "        for tcol in tcols:\n"
                        "            count += 1\n"
                        "            val = (_r.get(tcol) or '').strip()\n"
                        "            if not val or not val.endswith('Z'):\n"
                        "                bad.append(repr(val) + ' no trailing Z'); continue\n"
                        "            try:\n"
                        "                datetime.fromisoformat(val.replace('Z', '+00:00'))\n"
                        "            except ValueError:\n"
                        "                bad.append(repr(val) + ' not ISO-8601')\n"
                        "if count == 0:\n"
                        "    print('no timestamp cells in reconciled.csv'); sys.exit(1)\n"
                        "if bad:\n"
                        "    print('UTC non-conformance: ' + '; '.join(bad[:5])); sys.exit(1)\n"
                        "print('ok: all ' + str(count) + ' reconciled timestamps are UTC ISO-8601')\n"
                    ),
                    "timeout": 20,
                },
            ),
            CheckConfig(
                check_type="execution",
                name="q05_orphan_cross_check",
                params={
                    # O3 anti-fabrication crux. THREE independent orphan counts must
                    # agree: (1) integrity_report.orphaned_orders (the tool), (2)
                    # audit.independent_orphan_count + audit.matches_tool (the
                    # sub-agent), and (3) a recomputation right here from the RAW
                    # orders.jsonl + customers.jsonl (canonicalize the FK, dedupe
                    # orders by order_id, set-difference). An agent that writes a
                    # plausible-but-fabricated orphan number fails #3.
                    "code": (
                        "import json as _j, sys\n"
                        "report = next((p for p in _DELIVERABLES if p.endswith('integrity_report.json')), '')\n"
                        "audit = next((p for p in _DELIVERABLES if p.endswith('audit.json')), '')\n"
                        "orders = next((p for p in _DELIVERABLES if p.endswith('orders.jsonl')), '')\n"
                        "customers = next((p for p in _DELIVERABLES if p.endswith('customers.jsonl')), '')\n"
                        "if not (report and audit and orders and customers):\n"
                        "    print('missing a required deliverable'); sys.exit(1)\n"
                        "rep = _j.load(open(report)); aud = _j.load(open(audit))\n"
                        "tool_orphan = int(rep.get('orphaned_orders', -1))\n"
                        "ind_orphan = int(aud.get('independent_orphan_count', -1))\n"
                        "if aud.get('matches_tool') is not True:\n"
                        "    print('audit.matches_tool is not True'); sys.exit(1)\n"
                        "if tool_orphan != ind_orphan:\n"
                        "    print('tool ' + str(tool_orphan) + ' != auditor ' + str(ind_orphan)); sys.exit(1)\n"
                        "# Independent recomputation from the RAW sources.\n"
                        "cust_ids = set()\n"
                        "for _line in open(customers):\n"
                        "    _line = _line.strip()\n"
                        "    if not _line:\n"
                        "        continue\n"
                        "    try:\n"
                        "        _o = _j.loads(_line)\n"
                        "    except ValueError:\n"
                        "        continue\n"
                        "    if isinstance(_o, dict):\n"
                        "        cid = str(_o.get('customer_id') or _o.get('id') or '').strip().lower()\n"
                        "        if cid:\n"
                        "            cust_ids.add(cid)\n"
                        "seen = set(); recomputed = 0; total = 0\n"
                        "for _line in open(orders):\n"
                        "    _line = _line.strip()\n"
                        "    if not _line:\n"
                        "        continue\n"
                        "    try:\n"
                        "        _o = _j.loads(_line)\n"
                        "    except ValueError:\n"
                        "        continue\n"
                        "    if not isinstance(_o, dict):\n"
                        "        continue\n"
                        "    oid = str(_o.get('order_id') or _o.get('id') or '').strip()\n"
                        "    key = oid or str(_o)\n"
                        "    if key in seen:\n"
                        "        continue\n"
                        "    seen.add(key)\n"
                        "    total += 1\n"
                        "    fk = str(_o.get('customer_id') or '').strip().lower()\n"
                        "    if fk and fk not in cust_ids:\n"
                        "        recomputed += 1\n"
                        "if recomputed != tool_orphan:\n"
                        "    print('recomputed orphan ' + str(recomputed) + ' != tool ' + str(tool_orphan)); sys.exit(1)\n"
                        "print('ok: orphan count ' + str(tool_orphan) + ' agreed by tool, auditor, and independent recompute (of ' + str(total) + ' orders)')\n"
                    ),
                    "timeout": 20,
                },
            ),
            CheckConfig(
                check_type="execution",
                name="q05_join_cardinality_sane",
                params={
                    "code": (
                        "import json as _j, sys\n"
                        "report = next((p for p in _DELIVERABLES if p.endswith('integrity_report.json')), '')\n"
                        "if not report:\n"
                        "    print('no integrity_report.json'); sys.exit(1)\n"
                        "data = _j.load(open(report))\n"
                        "to = int(data.get('total_orders', 0)); mt = int(data.get('matched_orders', 0))\n"
                        "if mt < 1:\n"
                        "    print('matched_orders < 1 (no join occurred)'); sys.exit(1)\n"
                        "if mt > to + 5:\n"
                        "    print('matched_orders ' + str(mt) + ' exceeds total_orders ' + str(to)); sys.exit(1)\n"
                        "print('ok: matched_orders ' + str(mt) + ' within [1, total ' + str(to) + ']')\n"
                    ),
                    "timeout": 20,
                },
            ),
        ],
    )


def _battery04_q06() -> GoalSpec:
    """Capstone: idempotent rollup + evolution + sub-agent tests + determinism.

    New dimension vs q01–q05: idempotency / determinism. The rollup tool must be
    reproducible (byte-identical across runs) and self-attest via a sha256
    manifest the probe cross-checks; a second ``idempotency`` check re-runs a
    reference transform twice. Then the tool is EVOLVED (q03 pattern) to v2 and
    an orchestrator sub-agent generates+runs a pytest suite (q04 pattern). All
    three prior dimensions compose here, plus the new determinism axis.
    """
    return GoalSpec(
        spec_id="battery04_q06",
        name="battery04_q06",
        description="Idempotent daily-rollup over q05 reconciled data, evolve to v2 (rolling avg), orchestrator sub-agent generates+runs a determinism pytest suite",
        goal_text=(
            "Reuse the q05 reconciliation tool. Read results/q05/reconciled.csv (if "
            "absent, read results/q05/customers.jsonl + orders.jsonl and reconcile "
            "inline first). CREATE an IDEMPOTENT daily-rollup tool that groups orders "
            "by UTC calendar date and writes results/q06/daily_rollup.csv with columns "
            "including order_count, revenue_sum, unique_customers for each date. The "
            "tool must be deterministic: running it twice over the same input must "
            "produce byte-identical output. The tool itself writes results/q06/run_manifest.json "
            "recording run_count (int, starts at 1), input_sha256 (sha256 of the "
            "reconciled input bytes), output_sha256 (sha256 of the rollup CSV it "
            "wrote), and is_idempotent (bool) which the tool sets by re-running its "
            "own transform in-memory and comparing the output hash. Then EVOLVE that "
            "rollup tool (TOOL/CODE mutation via the codegen model + safety + sandbox "
            "+ post-deploy verify) to ALSO emit a 7-day rolling revenue average "
            "column, writing results/q06/daily_rollup_v2.csv (if evolution cannot "
            "deploy within budget, still write daily_rollup_v2.csv by any deterministic "
            "means). Then SPAWN an orchestrator sub-agent that builds and runs a "
            "pytest suite validating BOTH rollups by EXACT written path: (a) sha256 "
            "of daily_rollup.csv is stable across two runs (idempotency), (b) "
            "revenue_sum tolerance checks, (c) no nulls in required columns, (d) UTC "
            "dates. Write the suite to results/q06/ and invoke pytest by its EXACT "
            "path. results/q06/test_results.json MUST record concrete integer "
            "pass/fail counts parsed from pytest (e.g. {\"passed\": N, \"failed\": M}); "
            "a {\"status\": \"failed\"} stub is INVALID. Also write "
            "results/q06/capstone_summary.md synthesizing the idempotency + evolution outcome."
        ),
        category="multi_agent",
        max_iterations=70,
        timeout_seconds=1500,
        expected_deliverables=[
            "results/q06/daily_rollup.csv",
            "results/q06/run_manifest.json",
            "results/q06/daily_rollup_v2.csv",
            "results/q06/test_results.json",
            "results/q06/capstone_summary.md",
        ],
        success_criteria=[
            "results/q06/daily_rollup.csv exists with >= 1 date row",
            "run_manifest.json records is_idempotent=true with self-consistent sha256",
            "daily_rollup_v2.csv adds a rolling-average column (evolution evidence)",
            "test_results.json records concrete integer pass/fail counts",
        ],
        checks=[
            CheckConfig(
                check_type="golden",
                name="q06_deliverables_exist",
                params={
                    "assertions": [
                        {"kind": "exists", "deliverable": "results/q06/daily_rollup.csv"},
                        {"kind": "exists", "deliverable": "results/q06/run_manifest.json"},
                        {"kind": "exists", "deliverable": "results/q06/daily_rollup_v2.csv"},
                        {"kind": "exists", "deliverable": "results/q06/test_results.json"},
                        {"kind": "exists", "deliverable": "results/q06/capstone_summary.md"},
                    ]
                },
            ),
            CheckConfig(
                check_type="structural",
                name="q06_rollup_csv_schema",
                params={"deliverable": "results/q06/daily_rollup.csv", "format": "csv", "min_rows": 1},
            ),
            CheckConfig(
                check_type="structural",
                name="q06_rollup_v2_csv_schema",
                params={"deliverable": "results/q06/daily_rollup_v2.csv", "format": "csv", "min_rows": 1},
            ),
            CheckConfig(
                check_type="execution",
                name="q06_manifest_idempotent",
                params={
                    # The manifest's is_idempotent flag must be true AND its
                    # output_sha256 must equal the actual sha256 of daily_rollup.csv
                    # on disk (catches a fabricated flag), and its input_sha256 must
                    # match the reconciled input found under _RESULTS_ROOT.
                    "code": (
                        "import json as _j, hashlib, os, sys\n"
                        "manifest = next((p for p in _DELIVERABLES if p.endswith('run_manifest.json')), '')\n"
                        "rollup = next((p for p in _DELIVERABLES if p.endswith('daily_rollup.csv')), '')\n"
                        "if not manifest or not rollup:\n"
                        "    print('missing run_manifest.json or daily_rollup.csv'); sys.exit(1)\n"
                        "m = _j.load(open(manifest))\n"
                        "if m.get('is_idempotent') is not True:\n"
                        "    print('manifest.is_idempotent is not True'); sys.exit(1)\n"
                        "disk_sha = hashlib.sha256(open(rollup, 'rb').read()).hexdigest()\n"
                        "if m.get('output_sha256') and m.get('output_sha256') != disk_sha:\n"
                        "    print('output_sha256 mismatch: manifest ' + str(m.get('output_sha256'))[:12] + ' != disk ' + disk_sha[:12]); sys.exit(1)\n"
                        "recon_hashes = []\n"
                        "if _RESULTS_ROOT:\n"
                        "    for _d, _, _fs in os.walk(_RESULTS_ROOT):\n"
                        "        if 'reconciled.csv' in _fs:\n"
                        "            _rp = os.path.join(_d, 'reconciled.csv')\n"
                        "            try:\n"
                        "                recon_hashes.append(hashlib.sha256(open(_rp, 'rb').read()).hexdigest())\n"
                        "            except OSError:\n"
                        "                pass\n"
                        "# The agent may re-reconcile into its own subdir (a stray\n"
                        "# results/qNN/q05/reconciled.csv); the manifest's input_sha256 is\n"
                        "# honest if it matches ANY reconciled.csv on disk. Match-by-\n"
                        "# membership (not first-walk-hit, whose os.walk order is\n"
                        "# non-deterministic) preserves the anti-fabrication intent.\n"
                        "if recon_hashes and m.get('input_sha256') and m.get('input_sha256') not in recon_hashes:\n"
                        "    print('input_sha256 mismatch: manifest not over any reconciled.csv on disk'); sys.exit(1)\n"
                        "print('ok: manifest idempotent + sha256 self-consistent')\n"
                    ),
                    "timeout": 20,
                },
            ),
            CheckConfig(
                check_type="idempotency",
                name="q06_rollup_deterministic",
                params={
                    # The reusable determinism family: run a reference daily-rollup
                    # transform over the q05 reconciled input TWICE and require
                    # byte-identical output. Proves the rollup transform class is
                    # reproducible (sorted aggregation, no wall-clock/RNG drift).
                    "input_deliverable": "results/q05/reconciled.csv",
                    "transform_code": (
                        "import csv\n"
                        "from collections import defaultdict\n"
                        "agg = defaultdict(lambda: [0, 0.0, set()])\n"
                        "with open(_INPUT, newline='') as _f:\n"
                        "    reader = csv.DictReader(_f)\n"
                        "    cols = reader.fieldnames or []\n"
                        "    dcol = next((c for c in cols if c.lower() in ('date', 'order_date', 'day')), None)\n"
                        "    if dcol is None:\n"
                        "        dcol = next((c for c in cols if 'timestamp' in c.lower() "
                        "or c.lower().endswith('_at') or c.lower().endswith('time')), None)\n"
                        "    rcol = next((c for c in cols if c.lower() in ('revenue', 'amount', "
                        "'order_total', 'total')), None)\n"
                        "    for _r in reader:\n"
                        "        d = ((_r.get(dcol) or '') if dcol else '').strip()[:10]\n"
                        "        if not d:\n"
                        "            continue\n"
                        "        agg[d][0] += 1\n"
                        "        if rcol:\n"
                        "            try:\n"
                        "                agg[d][1] += float(_r.get(rcol) or 0)\n"
                        "            except ValueError:\n"
                        "                pass\n"
                        "        cid = (_r.get('customer_id') or '').strip().lower()\n"
                        "        if cid:\n"
                        "            agg[d][2].add(cid)\n"
                        "for _d in sorted(agg):\n"
                        "    print(_d, agg[_d][0], round(agg[_d][1], 2), len(agg[_d][2]))\n"
                    ),
                    "timeout": 30,
                },
            ),
            CheckConfig(
                check_type="execution",
                name="q06_v2_has_rolling_avg",
                params={
                    "code": (
                        "import csv, re, sys\n"
                        "v1 = next((p for p in _DELIVERABLES if p.endswith('daily_rollup.csv')), '')\n"
                        "v2 = next((p for p in _DELIVERABLES if p.endswith('daily_rollup_v2.csv')), '')\n"
                        "if not v1 or not v2:\n"
                        "    print('missing a rollup'); sys.exit(1)\n"
                        "h1 = next(csv.reader(open(v1, newline='')))\n"
                        "h2 = next(csv.reader(open(v2, newline='')))\n"
                        "rolling = [c for c in h2 if re.search(r'rolling|7.?day|avg', c, re.I)]\n"
                        "if not rolling:\n"
                        "    print('v2 has no rolling/7-day/avg column: ' + str(h2)); sys.exit(1)\n"
                        "if len(h2) <= len(h1):\n"
                        "    print('v2 not wider than v1 (' + str(len(h2)) + ' <= ' + str(len(h1)) + ')'); sys.exit(1)\n"
                        "print('ok: v2 adds rolling-avg column(s) ' + str(rolling))\n"
                    ),
                    "timeout": 20,
                },
            ),
            CheckConfig(
                check_type="execution",
                name="q06_test_results_real_counts",
                params={
                    # Clone of the q04 stub-rejection probe: reject {status: failed}
                    # stubs and require concrete INTEGER pass/fail verdicts (>= 1
                    # pass or fail, not just errors/zero).
                    "code": (
                        "import json as _j, sys\n"
                        "path = next((p for p in _DELIVERABLES if p.endswith('test_results.json')), '')\n"
                        "if not path:\n"
                        "    print('no test_results.json'); sys.exit(1)\n"
                        "try:\n"
                        "    data = _j.load(open(path))\n"
                        "except Exception as _e:\n"
                        "    print('unparseable test_results.json:', _e); sys.exit(1)\n"
                        "_STUB = {'failed', 'error', 'fail', 'errored', 'crashed', 'aborted'}\n"
                        "_status = ''\n"
                        "if isinstance(data, dict):\n"
                        "    for _k in ('status', 'overall_status', 'result', 'outcome'):\n"
                        "        _v = str(data.get(_k, '')).lower().strip()\n"
                        "        if _v in _STUB:\n"
                        "            _status = _v; break\n"
                        "if _status:\n"
                        "    print('rejected: test_results.json is a failure stub (' + _status + ')'); sys.exit(1)\n"
                        "_COUNT_KEYS = ('passed', 'failed', 'errors', 'skipped', 'total', 'collected', 'pass', 'fail')\n"
                        "def _ints(d):\n"
                        "    out = {}\n"
                        "    if isinstance(d, dict):\n"
                        "        for k, v in d.items():\n"
                        "            if k in _COUNT_KEYS:\n"
                        "                try:\n"
                        "                    f = float(v)\n"
                        "                except (TypeError, ValueError):\n"
                        "                    continue\n"
                        "                if f == int(f):\n"
                        "                    out[k] = int(f)\n"
                        "    return out\n"
                        "counts = _ints(data)\n"
                        "if not counts and isinstance(data, dict):\n"
                        "    for _sub in (data.get('summary'), data.get('totals'), data.get('stats')):\n"
                        "        counts = _ints(_sub)\n"
                        "        if counts:\n"
                        "            break\n"
                        "if not counts:\n"
                        "    print('no integer pass/fail counts recorded'); sys.exit(1)\n"
                        "_verdicts = counts.get('passed', 0) + counts.get('pass', 0) + counts.get('failed', 0) + counts.get('fail', 0)\n"
                        "if _verdicts <= 0:\n"
                        "    print('no tests reached a pass/fail verdict (suite did not execute)'); sys.exit(1)\n"
                        "print('ok: test results record counts ' + _j.dumps(counts))\n"
                    ),
                    "timeout": 20,
                },
            ),
            CheckConfig(
                check_type="execution",
                name="q06_utc_dates",
                params={
                    "code": (
                        "import csv, sys\n"
                        "from datetime import datetime\n"
                        "path = next((p for p in _DELIVERABLES if p.endswith('daily_rollup.csv')), '')\n"
                        "if not path:\n"
                        "    print('no daily_rollup.csv'); sys.exit(1)\n"
                        "bad = []; count = 0\n"
                        "with open(path, newline='') as _f:\n"
                        "    reader = csv.DictReader(_f)\n"
                        "    dcol = next((c for c in (reader.fieldnames or []) if 'date' in c.lower() or 'day' in c.lower()), None)\n"
                        "    if dcol is None:\n"
                        "        print('no date column'); sys.exit(1)\n"
                        "    for _r in reader:\n"
                        "        count += 1\n"
                        "        val = (_r.get(dcol) or '').strip()[:10]\n"
                        "        try:\n"
                        "            datetime.fromisoformat(val)\n"
                        "        except ValueError:\n"
                        "            bad.append(repr(val))\n"
                        "if count == 0:\n"
                        "    print('no rows'); sys.exit(1)\n"
                        "if bad:\n"
                        "    print('unparseable dates: ' + '; '.join(bad[:5])); sys.exit(1)\n"
                        "print('ok: ' + str(count) + ' dates parse as ISO')\n"
                    ),
                    "timeout": 20,
                },
            ),
            CheckConfig(
                check_type="execution",
                name="q06_no_placeholder_summary",
                params={
                    # Clone of the q02 placeholder-leak probe: capstone_summary.md
                    # prose must not carry >= 2 unsubstituted {var}/{{var}} residues.
                    "code": (
                        "import re, sys\n"
                        "path = next((p for p in _DELIVERABLES if p.endswith('capstone_summary.md')), '')\n"
                        "if not path:\n"
                        "    print('no capstone_summary.md'); sys.exit(1)\n"
                        "t = open(path, encoding='utf-8', errors='replace').read()\n"
                        "t = re.sub(r'```.*?```', '', t, flags=re.DOTALL)\n"
                        "t = re.sub(r'`[^`\\n]*`', '', t)\n"
                        "fmt = set(re.findall(r'(?<!\\{)\\{([a-z_][a-z0-9_]{2,})\\}(?!\\})', t))\n"
                        "jv = set(re.findall(r'\\{\\{\\s*([a-zA-Z_][a-zA-Z0-9_.-]*)\\s*\\}\\}', t))\n"
                        "leaks = fmt | jv\n"
                        "if len(leaks) >= 2:\n"
                        "    print('placeholder leak: ' + ', '.join(sorted(leaks)[:6])); sys.exit(1)\n"
                        "print('ok: no template-placeholder residue in summary')\n"
                    ),
                    "timeout": 20,
                },
            ),
        ],
    )


def _battery04_q07() -> GoalSpec:
    """Constraint-satisfaction optimization + adversarial self-test.

    New dimension vs q01–q06: a hard+soft CONSTRAINT-SATISFACTION problem whose
    verifier RECOMPUTES every hard constraint and the objective from the RAW
    instance (the O3 anti-fabrication backbone), then an adversarial probe proves
    the recompute is SOUND by injecting a deliberate violation and requiring it
    be flagged. A fabricated ``constraint_report.json`` (plausible numbers that
    disagree with the actual ``solution.json``) fails the recompute; a no-op
    checker that always returns 0 violations fails the soundness probe. The
    objective-sane check adds a non-tie-flaky optimization property: when
    capacity binds, a dropped skill-feasible task must not outrank a kept one.
    """

    # Anti-fabrication backbone — recomputes every hard constraint + objective
    # from instance.json + solution.json. Shared verbatim across the no-hard-
    # violations, constraints-recomputed, objective-sane, and adversarial
    # soundness probes so they cannot diverge.
    _RECOMPUTE = (
        "import json as _j, sys\n"
        "inst = next((p for p in _DELIVERABLES if p.endswith('instance.json')), '')\n"
        "sol = next((p for p in _DELIVERABLES if p.endswith('solution.json')), '')\n"
        "if not inst or not sol:\n"
        "    print('missing instance.json or solution.json'); sys.exit(1)\n"
        "I = _j.load(open(inst)); S = _j.load(open(sol))\n"
        "eng = {}\n"
        "for _e in I.get('engineers', []):\n"
        "    _eid = str(_e.get('id') or '').strip()\n"
        "    if _eid:\n"
        "        eng[_eid] = {'cap': float(_e.get('capacity_hours') or 0), "
        "'skills': set(str(s).strip() for s in (_e.get('skills') or []))}\n"
        "tasks = {}\n"
        "for _t in I.get('tasks', []):\n"
        "    _tid = str(_t.get('id') or '').strip()\n"
        "    if _tid:\n"
        "        tasks[_tid] = {'hours': float(_t.get('required_hours') or 0), "
        "'skill': str(_t.get('required_skill') or '').strip(), "
        "'pri': float(_t.get('priority') or 0)}\n"
        "asg = S.get('assignments') or []\n"
        "unassigned = set(str(x) for x in (S.get('unassigned') or []))\n"
        "seen = set(); skill_mismatch = 0; cap_used = {}; violations = []\n"
        "for _a in asg:\n"
        "    _tid = str((_a.get('task_id') if isinstance(_a, dict) else '') or '').strip()\n"
        "    _eid = str((_a.get('engineer_id') if isinstance(_a, dict) else '') or '').strip()\n"
        "    if _tid in seen:\n"
        "        violations.append('duplicate task ' + _tid); continue\n"
        "    seen.add(_tid)\n"
        "    if _tid not in tasks:\n"
        "        violations.append('unknown task ' + _tid); continue\n"
        "    if _eid not in eng:\n"
        "        violations.append('unknown engineer ' + _eid); continue\n"
        "    if tasks[_tid]['skill'] not in eng[_eid]['skills']:\n"
        "        skill_mismatch += 1; violations.append('skill mismatch ' + _tid)\n"
        "    cap_used[_eid] = cap_used.get(_eid, 0.0) + tasks[_tid]['hours']\n"
        "for _eid, _u in cap_used.items():\n"
        "    if _u > eng[_eid]['cap'] + 1e-6:\n"
        "        violations.append('capacity exceeded ' + _eid)\n"
        "for _tid in unassigned:\n"
        "    if _tid in seen:\n"
        "        violations.append('task in both assigned and unassigned ' + _tid)\n"
        "hard_violations = len(violations)\n"
        "priority_sum = sum(tasks[_t]['pri'] for _t in seen if _t in tasks)\n"
        "total_priority = sum(_t['pri'] for _t in tasks.values())\n"
    )

    return GoalSpec(
        spec_id="battery04_q07",
        name="battery04_q07",
        description=(
            "Constraint-satisfaction optimization (task→engineer assignment with "
            "skill + capacity hard constraints, maximize priority) verified by raw "
            "recomputation, then hardened against an adversarial self-test"
        ),
        goal_text=(
            "Generate a constraint-satisfaction instance and write it to "
            "results/q07/instance.json with EXACTLY this shape: {\"engineers\": "
            "[{\"id\": str, \"capacity_hours\": number, \"skills\": [str], "
            "\"cost_per_hour\": number}], \"tasks\": [{\"id\": str, "
            "\"required_hours\": number, \"required_skill\": str, \"priority\": int}]}. "
            "Use >= 6 engineers and >= 15 tasks across >= 3 distinct skills, with "
            "TIGHT total capacity (sum of capacity_hours noticeably less than sum "
            "of required_hours) so the optimization is binding — some tasks CANNOT "
            "all be assigned. Task priorities are integers (higher = more important).\n"
            "STRATEGY (critical — do NOT hand-compute): Write ONE Python solver "
            "script and EXECUTE it via the code_executor tool. The script must "
            "(a) load instance.json it just wrote; (b) compute the priority-"
            "maximizing assignment that respects all three HARD constraints — the "
            "instance is small, so sort tasks by priority descending and greedily "
            "assign each to the eligible engineer (has the required_skill) with "
            "the most spare capacity; (c) write BOTH results/q07/solution.json AND "
            "results/q07/constraint_report.json from the SAME in-memory assignment "
            "so every recomputed value provably agrees with the solution. Never "
            "edit either file by hand after the script runs.\n"
            "HARD constraints the assignment must satisfy: (1) each task assigned "
            "to at most one engineer, (2) the engineer possesses the task's "
            "required_skill, (3) the sum of required_hours assigned to an engineer "
            "never exceeds their capacity_hours. Write results/q07/solution.json "
            "with EXACTLY: "
            "{\"assignments\": [{\"task_id\": str, \"engineer_id\": str}], "
            "\"unassigned\": [str], \"objective_value\": number} where objective_value "
            "is the achieved total assigned priority.\n"
            "Write results/q07/constraint_report.json with EXACTLY: {\"total_tasks\": "
            "int, \"assigned_tasks\": int, \"unassigned_tasks\": int, \"hard_violations\": "
            "int, \"skill_mismatches\": int, \"capacity_used\": {engineer_id: number}, "
            "\"objective_recomputed\": number} — every value RECOMPUTED from instance.json "
            "+ solution.json inside the solver script. objective_value (solution) and "
            "objective_recomputed (report) MUST both equal the sum of priorities of "
            "assigned tasks.\n"
            "Then SPAWN an adversarial sub-agent that tries to BREAK the solution. The "
            "adversarial sub-agent MUST itself load instance.json + solution.json and "
            "test the REAL assignments present in those files (never invent task/"
            "engineer ids that do not appear): it performs >= 2 attacks (e.g. move a "
            "task to an engineer lacking the required_skill, push an engineer past "
            "capacity_hours, double-assign one task) and for each confirms the "
            "constraint checker WOULD flag it, then confirms the ORIGINAL solution has "
            "zero such violations. Write results/q07/adversarial_report.json "
            "with EXACTLY: {\"attacks_tried\": int, \"violations_found\": int, "
            "\"hardened\": bool, \"attacks\": [str]}. violations_found is the count "
            "of hard-constraint violations present in the ORIGINAL (unmutated) "
            "solution — it MUST be 0; the violations your attacks INJECT into "
            "mutated copies are NOT counted here (they only prove the checker "
            "works). The original solution must have violations_found == 0."
        ),
        category="optimization",
        max_iterations=50,
        timeout_seconds=1100,
        expected_deliverables=[
            "results/q07/instance.json",
            "results/q07/solution.json",
            "results/q07/constraint_report.json",
            "results/q07/adversarial_report.json",
        ],
        success_criteria=[
            "results/q07/solution.json assigns tasks with zero hard-constraint violations",
            "constraint_report.json values match an independent recomputation from the raw instance",
            "objective_value equals the recomputed assigned-priority sum",
            "adversarial_report.json records attacks_tried >= 2 with violations_found == 0",
        ],
        checks=[
            CheckConfig(
                check_type="golden",
                name="q07_deliverables_exist",
                params={
                    "assertions": [
                        {"kind": "exists", "deliverable": "results/q07/instance.json"},
                        {"kind": "exists", "deliverable": "results/q07/solution.json"},
                        {"kind": "exists", "deliverable": "results/q07/constraint_report.json"},
                        {"kind": "exists", "deliverable": "results/q07/adversarial_report.json"},
                    ]
                },
            ),
            CheckConfig(
                check_type="structural",
                name="q07_instance_json_schema",
                params={"deliverable": "results/q07/instance.json", "format": "json"},
            ),
            CheckConfig(
                check_type="structural",
                name="q07_solution_json_schema",
                params={"deliverable": "results/q07/solution.json", "format": "json"},
            ),
            CheckConfig(
                check_type="structural",
                name="q07_constraint_report_schema",
                params={"deliverable": "results/q07/constraint_report.json", "format": "json"},
            ),
            CheckConfig(
                check_type="execution",
                name="q07_constraints_recomputed",
                params={
                    # O3 anti-fabrication crux. Recompute every constraint-report
                    # value from the RAW instance + solution and require it EQUAL
                    # the report's claim. An agent that writes a plausible-but-
                    # fabricated constraint_report (e.g. hard_violations=0 while the
                    # solution actually violates) is caught here.
                    "code": _RECOMPUTE
                    + (
                        "rep = _j.load(open(next((p for p in _DELIVERABLES if "
                        "p.endswith('constraint_report.json')), '')))\n"
                        "def _near(a, b):\n"
                        "    try:\n"
                        "        return abs(float(a) - float(b)) <= 1e-6\n"
                        "    except (TypeError, ValueError):\n"
                        "        return False\n"
                        "mism = []\n"
                        "if int(rep.get('assigned_tasks', -999)) != len(seen):\n"
                        "    mism.append('assigned_tasks report=' + str(rep.get('assigned_tasks')) + ' recompute=' + str(len(seen)))\n"
                        "if int(rep.get('unassigned_tasks', -999)) != len(unassigned):\n"
                        "    mism.append('unassigned_tasks report=' + str(rep.get('unassigned_tasks')) + ' recompute=' + str(len(unassigned)))\n"
                        "if int(rep.get('hard_violations', -999)) != hard_violations:\n"
                        "    mism.append('hard_violations report=' + str(rep.get('hard_violations')) + ' recompute=' + str(hard_violations))\n"
                        "if int(rep.get('skill_mismatches', -999)) != skill_mismatch:\n"
                        "    mism.append('skill_mismatches report=' + str(rep.get('skill_mismatches')) + ' recompute=' + str(skill_mismatch))\n"
                        "if not _near(rep.get('objective_recomputed'), priority_sum):\n"
                        "    mism.append('objective_recomputed report=' + str(rep.get('objective_recomputed')) + ' recompute=' + str(priority_sum))\n"
                        "rep_cap = rep.get('capacity_used') or {}\n"
                        "if set(str(k) for k in rep_cap) != set(cap_used):\n"
                        "    mism.append('capacity_used engineers differ')\n"
                        "else:\n"
                        "    for _k, _v in cap_used.items():\n"
                        "        if not _near(rep_cap.get(str(_k)), _v):\n"
                        "            mism.append('capacity_used ' + str(_k) + ' report=' + str(rep_cap.get(str(_k))) + ' recompute=' + str(_v))\n"
                        "if mism:\n"
                        "    print('constraint_report disagrees with recomputation: ' + '; '.join(mism[:6])); sys.exit(1)\n"
                        "print('ok: constraint_report matches independent recomputation')\n"
                    ),
                    "timeout": 20,
                },
            ),
            CheckConfig(
                check_type="execution",
                name="q07_no_hard_violations",
                params={
                    "code": _RECOMPUTE
                    + (
                        "if hard_violations != 0:\n"
                        "    print('solution has ' + str(hard_violations) + ' hard violations: ' + '; '.join(violations[:5])); sys.exit(1)\n"
                        "print('ok: zero hard violations across ' + str(len(seen)) + ' assignments')\n"
                    ),
                    "timeout": 20,
                },
            ),
            CheckConfig(
                check_type="execution",
                name="q07_objective_sane",
                params={
                    # Anti-fabrication (objective_value == recomputed) + bounds +
                    # a non-tie-flaky optimization property: when capacity binds,
                    # a dropped skill-feasible task must not strictly outrank a
                    # kept one. A greedy-by-priority solution satisfies this; a
                    # random/non-optimizing assignment that drops high-priority
                    # work while keeping low-priority work violates it.
                    "code": _RECOMPUTE
                    + (
                        "def _near(a, b):\n"
                        "    try:\n"
                        "        return abs(float(a) - float(b)) <= 1e-6\n"
                        "    except (TypeError, ValueError):\n"
                        "        return False\n"
                        "if not _near(S.get('objective_value'), priority_sum):\n"
                        "    print('objective_value ' + str(S.get('objective_value')) + ' != recomputed ' + str(priority_sum)); sys.exit(1)\n"
                        "if priority_sum < 0 or priority_sum > total_priority + 1e-6:\n"
                        "    print('objective out of bounds [0, ' + str(total_priority) + ']'); sys.exit(1)\n"
                        "if len(seen) == 0:\n"
                        "    print('nothing assigned'); sys.exit(1)\n"
                        "# Optimization property (only when capacity binds → some task unassigned).\n"
                        "if unassigned:\n"
                        "    assigned_pri = [tasks[_t]['pri'] for _t in seen if _t in tasks]\n"
                        "    min_assigned = min(assigned_pri) if assigned_pri else 0\n"
                        "    # addable = an eligible engineer (has the skill) with enough SPARE\n"
                        "    # capacity to take the task right now. A skill-feasible task whose\n"
                        "    # eligible engineers are all capacity-saturated is genuinely\n"
                        "    # unassignable, so dropping it is optimal — flagging it is a false\n"
                        "    # positive (battery-04 q07: a pri-6 task needed 16h of a skill whose\n"
                        "    # every engineer had <9h spare). This still catches the real failure\n"
                        "    # mode: a greedy solution that leaves an addable high-priority task\n"
                        "    # unassigned while keeping lower-priority work.\n"
                        "    bad = []\n"
                        "    for _u in unassigned:\n"
                        "        if _u not in tasks:\n"
                        "            continue\n"
                        "        _sk = tasks[_u]['skill']; _hrs = tasks[_u]['hours']\n"
                        "        addable = any((_sk in _e['skills']) and (_e['cap'] - cap_used.get(_eid, 0.0) >= _hrs - 1e-6) for _eid, _e in eng.items())\n"
                        "        if addable and tasks[_u]['pri'] > min_assigned + 1e-6:\n"
                        "            bad.append(_u + '(pri ' + str(tasks[_u]['pri']) + ')')\n"
                        "    if bad:\n"
                        "        print('non-optimal: dropped addable higher-priority task(s) ' + ', '.join(bad[:5]) + ' while keeping pri<=' + str(min_assigned)); sys.exit(1)\n"
                        "print('ok: objective ' + str(priority_sum) + ' sane and near-optimal')\n"
                    ),
                    "timeout": 20,
                },
            ),
            CheckConfig(
                check_type="execution",
                name="q07_adversarial_sanity",
                params={
                    # The adversarial self-test with teeth. (1) The adversarial
                    # report must record >= 2 attacks with violations_found == 0
                    # on the original. (2) SOUNDNESS: inject a deliberate
                    # duplicate assignment into an in-memory copy and require the
                    # SAME recomputation flags hard_violations > 0 — proving the
                    # 0-violation verdict above is meaningful, not a no-op checker.
                    "code": _RECOMPUTE
                    + (
                        "adv_path = next((p for p in _DELIVERABLES if p.endswith('adversarial_report.json')), '')\n"
                        "if not adv_path:\n"
                        "    print('no adversarial_report.json'); sys.exit(1)\n"
                        "adv = _j.load(open(adv_path))\n"
                        "if int(adv.get('attacks_tried', 0)) < 2:\n"
                        "    print('adversarial attacks_tried < 2'); sys.exit(1)\n"
                        "if int(adv.get('violations_found', -1)) != 0:\n"
                        "    print('adversarial_report.violations_found must be 0 — it is the count of hard violations in the ORIGINAL (unmutated) solution, NOT the violations the attacks injected; got ' + str(adv.get('violations_found'))); sys.exit(1)\n"
                        "if adv.get('hardened') is not True:\n"
                        "    print('adversarial_report.hardened is not True'); sys.exit(1)\n"
                        "# Soundness: duplicate the first assignment → must be flagged.\n"
                        "if not asg:\n"
                        "    print('no assignments to mutate for soundness check'); sys.exit(1)\n"
                        "mutated = list(asg) + [asg[0]]\n"
                        "seen2 = set(); viols2 = 0\n"
                        "for _a in mutated:\n"
                        "    _tid = str((_a.get('task_id') if isinstance(_a, dict) else '') or '').strip()\n"
                        "    if _tid in seen2:\n"
                        "        viols2 += 1\n"
                        "    else:\n"
                        "        seen2.add(_tid)\n"
                        "if viols2 < 1:\n"
                        "    print('soundness FAILED: injected duplicate was NOT detected'); sys.exit(1)\n"
                        "print('ok: adversarial attacks=' + str(adv.get('attacks_tried')) + ' original clean; checker sound')\n"
                    ),
                    "timeout": 20,
                },
            ),
            CheckConfig(
                check_type="idempotency",
                name="q07_recompute_deterministic",
                params={
                    # The recomputation itself must be reproducible. Run a
                    # canonical constraint recompute over instance.json (with
                    # solution.json located under _RESULTS_ROOT) TWICE and require
                    # byte-identical output — proves the verifier is deterministic.
                    "input_deliverable": "results/q07/instance.json",
                    "transform_code": (
                        "import json as _j, os, sys\n"
                        "I = _j.load(open(_INPUT))\n"
                        "sol = ''\n"
                        "for _d, _, _fs in os.walk(_RESULTS_ROOT):\n"
                        "    if 'solution.json' in _fs and 'q07' in _d:\n"
                        "        sol = os.path.join(_d, 'solution.json'); break\n"
                        "if not sol:\n"
                        "    print('no solution.json'); sys.exit(1)\n"
                        "S = _j.load(open(sol))\n"
                        "eng = {_e['id']: _e for _e in I.get('engineers', []) if _e.get('id')}\n"
                        "tasks = {_t['id']: _t for _t in I.get('tasks', []) if _t.get('id')}\n"
                        "seen = set(); skill_mismatch = 0; cap_used = {}; viols = 0\n"
                        "for _a in (S.get('assignments') or []):\n"
                        "    _tid = str(_a.get('task_id') or '').strip()\n"
                        "    _eid = str(_a.get('engineer_id') or '').strip()\n"
                        "    if _tid in seen:\n"
                        "        viols += 1; continue\n"
                        "    seen.add(_tid)\n"
                        "    if _tid in tasks and _eid in eng:\n"
                        "        if str(tasks[_tid].get('required_skill')) not in "
                        "(str(s) for s in (eng[_eid].get('skills') or [])):\n"
                        "            skill_mismatch += 1\n"
                        "        cap_used[_eid] = cap_used.get(_eid, 0.0) + "
                        "float(tasks[_tid].get('required_hours') or 0)\n"
                        "for _eid, _u in cap_used.items():\n"
                        "    if _u > float(eng[_eid].get('capacity_hours') or 0) + 1e-6:\n"
                        "        viols += 1\n"
                        "obj = sum(float(tasks[_t].get('priority') or 0) for _t in seen if _t in tasks)\n"
                        "print('assigned=' + str(len(seen)) + ' violations=' + str(viols) + "
                        "' skill_mismatch=' + str(skill_mismatch) + ' objective=' + str(round(obj, 6)))\n"
                    ),
                    "timeout": 30,
                },
            ),
        ],
    )


def _battery04_q08() -> GoalSpec:
    """Long-horizon multi-agent orchestration with content-verified handoffs.

    New dimension vs q01–q07: cross-agent HANDOFF INTEGRITY. A 3-stage pipeline
    (research → analyze → synthesize) plus an orchestrator where each downstream
    stage consumes an upstream artifact BY CONTENT. The handoff_manifest records,
    per stage, the sha256 of the upstream artifact it read plus a derived_value
    that is a function of that artifact's CONTENT; the probe recomputes both the
    hash and the derived value — so an agent that hand-waves a manifest (hashes a
    filename, or copies a derived_value without parsing) is caught. A second
    probe recomputes the full analysis→synthesis chain from raw_findings.jsonl,
    rejecting a fabricated mid-stage artifact. The placeholder-leak clone guards
    the final markdown.
    """

    # Full recompute chain: raw_findings.jsonl → per-category aggregates (analysis)
    # → overall synthesis (total, worst_category). Shared across the aggregation
    # probe and the handoff-integrity derived-value checks.
    _AGG_FROM_RAW = (
        "import json as _j, sys\n"
        "from collections import defaultdict\n"
        "raw = next((p for p in _DELIVERABLES if p.endswith('raw_findings.jsonl')), '')\n"
        "if not raw:\n"
        "    print('no raw_findings.jsonl'); sys.exit(1)\n"
        "cat = defaultdict(list)\n"
        "for _line in open(raw):\n"
        "    _line = _line.strip()\n"
        "    if not _line:\n"
        "        continue\n"
        "    try:\n"
        "        _o = _j.loads(_line)\n"
        "    except ValueError:\n"
        "        continue\n"
        "    if isinstance(_o, dict) and _o.get('category') is not None:\n"
        "        cat[str(_o['category'])].append(_o)\n"
        "if len(cat) < 3:\n"
        "    print('fewer than 3 categories in raw_findings'); sys.exit(1)\n"
        "total = sum(len(v) for v in cat.values())\n"
        "if total < 15:\n"
        "    print('fewer than 15 raw findings (' + str(total) + ')'); sys.exit(1)\n"
        "agg = {}\n"
        "for _c, items in cat.items():\n"
        "    sevs = [float(it.get('severity') or 0) for it in items]\n"
        "    mets = [float(it.get('metric_value') or 0) for it in items]\n"
        "    ids = sorted((str(it.get('id') or '') for it in items), key=lambda _x: _x)\n"
        "    agg[_c] = {'count': len(items), 'avg_severity': (sum(sevs)/len(sevs) if sevs else 0.0), "
        "'total_metric': sum(mets), 'ids': ids}\n"
        "worst = max(agg, key=lambda _c: (agg[_c]['avg_severity'], agg[_c]['total_metric']))\n"
    )

    return GoalSpec(
        spec_id="battery04_q08",
        name="battery04_q08",
        description=(
            "Long-horizon 3-stage multi-agent pipeline (research → analyze → "
            "synthesize) + orchestrator with content-verified handoffs and "
            "recomputed aggregation chain"
        ),
        goal_text=(
            "Run a 3-STAGE multi-agent pipeline with REAL handoff dependencies, "
            "then an orchestrator that aggregates. Use 2–3 specialized sub-agents.\n"
            "Stage 1 (researcher sub-agent): GENERATE a set of findings and write "
            "results/q08/raw_findings.jsonl (one JSON object per line), each with "
            "EXACTLY keys id (str), category (str), severity (int 1–5), "
            "metric_value (number), description (str). Produce >= 15 findings across "
            ">= 3 distinct categories.\n"
            "Stage 2 (analyst sub-agent): READ results/q08/raw_findings.jsonl and "
            "compute per-category aggregates FROM ITS CONTENT (do not fabricate). "
            "Write results/q08/analysis.json with EXACTLY: {\"categories\": "
            "[{\"category\": str, \"count\": int, \"avg_severity\": number, "
            "\"total_metric\": number, \"top_finding_ids\": [str]}], "
            "\"total_findings\": int}.\n"
            "Stage 3 (synthesizer sub-agent): READ results/q08/analysis.json and "
            "derive the overall picture FROM ITS CONTENT. Write results/q08/synthesis.json "
            "with EXACTLY: {\"total_findings\": int, \"worst_category\": str, "
            "\"worst_avg_severity\": number, \"category_count\": int, "
            "\"recommended_actions\": [str]}. worst_category is the category with the "
            "highest avg_severity (ties broken by highest total_metric).\n"
            "Orchestrator: aggregate everything into results/q08/final_report.md "
            "(real prose summarizing the totals, the worst category, and recommended "
            "actions — cite the recomputed numbers) AND results/q08/handoff_manifest.json "
            "with EXACTLY: {\"stages\": [{\"stage\": str, \"sub_agent\": str, "
            "\"input_artifact\": str (relative path, '' for stage 1), \"input_sha256\": "
            "str (sha256 of the upstream artifact bytes, '' for stage 1), "
            "\"output_artifact\": str (relative path), \"derived_value\": str}]}. "
            "For each downstream stage, derived_value MUST be a value computed by "
            "parsing the upstream artifact's CONTENT: stage 'analysis' → "
            "derived_value = str(total findings in raw_findings.jsonl); stage "
            "'synthesis' → derived_value = the worst_category recomputed from "
            "analysis.json. Record one stage entry for analysis, synthesis, and the "
            "orchestrator (whose input_artifact is synthesis.json)."
        ),
        category="multi_agent",
        max_iterations=60,
        timeout_seconds=1300,
        expected_deliverables=[
            "results/q08/raw_findings.jsonl",
            "results/q08/analysis.json",
            "results/q08/synthesis.json",
            "results/q08/final_report.md",
            "results/q08/handoff_manifest.json",
        ],
        success_criteria=[
            "Each downstream stage's handoff_manifest entry matches a recomputation of its upstream artifact",
            "analysis.json aggregates and synthesis.json totals match an independent recomputation from raw_findings.jsonl",
            "final_report.md cites the recomputed totals with no template-placeholder residue",
        ],
        checks=[
            CheckConfig(
                check_type="golden",
                name="q08_deliverables_exist",
                params={
                    "assertions": [
                        {"kind": "exists", "deliverable": "results/q08/raw_findings.jsonl"},
                        {"kind": "exists", "deliverable": "results/q08/analysis.json"},
                        {"kind": "exists", "deliverable": "results/q08/synthesis.json"},
                        {"kind": "exists", "deliverable": "results/q08/final_report.md"},
                        {"kind": "exists", "deliverable": "results/q08/handoff_manifest.json"},
                    ]
                },
            ),
            CheckConfig(
                check_type="structural",
                name="q08_raw_jsonl_rows",
                params={"deliverable": "results/q08/raw_findings.jsonl", "format": "jsonl", "min_rows": 15},
            ),
            CheckConfig(
                check_type="structural",
                name="q08_analysis_json_schema",
                params={"deliverable": "results/q08/analysis.json", "format": "json"},
            ),
            CheckConfig(
                check_type="structural",
                name="q08_synthesis_json_schema",
                params={"deliverable": "results/q08/synthesis.json", "format": "json"},
            ),
            CheckConfig(
                check_type="structural",
                name="q08_handoff_manifest_schema",
                params={"deliverable": "results/q08/handoff_manifest.json", "format": "json"},
            ),
            CheckConfig(
                check_type="execution",
                name="q08_handoff_integrity",
                params={
                    # The new dimension. For each downstream manifest stage, the
                    # input_sha256 must equal the actual sha256 of the named
                    # upstream artifact on disk (it read THAT file), AND the
                    # derived_value must match an independent recomputation from
                    # that artifact's CONTENT. A fabricated manifest (hash of a
                    # filename, or a derived_value not actually parsed) fails.
                    "code": (
                        "import json as _j, hashlib, os, sys\n"
                        "man = next((p for p in _DELIVERABLES if p.endswith('handoff_manifest.json')), '')\n"
                        "if not man:\n"
                        "    print('no handoff_manifest.json'); sys.exit(1)\n"
                        "M = _j.load(open(man))\n"
                        "stages = M.get('stages') or []\n"
                        "if len(stages) < 3:\n"
                        "    print('manifest has fewer than 3 stages'); sys.exit(1)\n"
                        "root = _RESULTS_ROOT or os.getcwd()\n"
                        "def _resolve(rel):\n"
                        "    rel = str(rel or '').strip()\n"
                        "    if not rel:\n"
                        "        return ''\n"
                        "    # hermetic: every candidate MUST live under _RESULTS_ROOT. A bare\n"
                        "    # `rel` resolves against the subprocess CWD and silently picks up an\n"
                        "    # unrelated real artifact there (breaks test isolation + reads outside\n"
                        "    # the results root). The os.walk loop below still finds a basename\n"
                        "    # match anywhere under root, so honest upstreams still resolve.\n"
                        "    cands = [os.path.join(root, rel)]\n"
                        "    for _d, _, _fs in os.walk(root):\n"
                        "        if os.path.basename(rel) in _fs:\n"
                        "            cands.append(os.path.join(_d, os.path.basename(rel)))\n"
                        "    for c in cands:\n"
                        "        if c and os.path.isfile(c):\n"
                        "            return c\n"
                        "    return ''\n"
                        "def _sha(p):\n"
                        "    return hashlib.sha256(open(p, 'rb').read()).hexdigest()\n"
                        "def _near(a, b):\n"
                        "    try:\n"
                        "        return abs(float(a) - float(b)) <= 1e-6\n"
                        "    except (TypeError, ValueError):\n"
                        "        return False\n"
                        "for st in stages:\n"
                        "    inp = str(st.get('input_artifact') or '').strip()\n"
                        "    if not inp:\n"
                        "        continue  # stage 1 has no upstream\n"
                        "    ip = _resolve(inp)\n"
                        "    if not ip:\n"
                        "        print('stage ' + str(st.get('stage')) + ': upstream ' + inp + ' not found'); sys.exit(1)\n"
                        "    if str(st.get('input_sha256') or '') != _sha(ip):\n"
                        "        print('stage ' + str(st.get('stage')) + ': input_sha256 mismatch (did not read ' + inp + ')'); sys.exit(1)\n"
                        "    dv = str(st.get('derived_value') or '').strip()\n"
                        "    # Recompute derived_value from the upstream CONTENT.\n"
                        "    if inp.endswith('raw_findings.jsonl'):\n"
                        "        n = 0\n"
                        "        for _line in open(ip):\n"
                        "            if _line.strip():\n"
                        "                                n += 1\n"
                        "        if dv != str(n):\n"
                        "            print('stage ' + str(st.get('stage')) + ': derived_value ' + dv + ' != recomputed findings count ' + str(n)); sys.exit(1)\n"
                        "    elif inp.endswith('analysis.json'):\n"
                        "        A = _j.load(open(ip))\n"
                        "        cats = A.get('categories') or []\n"
                        "        best = None\n"
                        "        for _c in cats:\n"
                        "            sev = float(_c.get('avg_severity') or 0); met = float(_c.get('total_metric') or 0)\n"
                        "            if best is None or (sev, met) > (best[1], best[2]):\n"
                        "                best = (str(_c.get('category')), sev, met)\n"
                        "        if not best or dv != best[0]:\n"
                        "            print('stage ' + str(st.get('stage')) + ': derived_value ' + dv + ' != recomputed worst_category ' + str(best[0] if best else None)); sys.exit(1)\n"
                        "    elif inp.endswith('synthesis.json'):\n"
                        "        Ydoc = _j.load(open(ip))\n"
                        "        if dv != str(Ydoc.get('worst_category') or ''):\n"
                        "            print('stage ' + str(st.get('stage')) + ': derived_value ' + dv + ' != synthesis.worst_category ' + str(Ydoc.get('worst_category'))); sys.exit(1)\n"
                        "print('ok: all downstream handoffs content-verified (sha256 + derived_value recomputed)')\n"
                    ),
                    "timeout": 20,
                },
            ),
            CheckConfig(
                check_type="execution",
                name="q08_aggregation_complete",
                params={
                    # Full recompute chain. Recompute per-category aggregates from
                    # raw_findings.jsonl and require analysis.json to match; recompute
                    # total + worst_category and require synthesis.json to match; and
                    # require final_report.md to cite the recomputed total + worst
                    # category. A fabricated analysis/synthesis is caught at the raw
                    # recomputation.
                    "code": _AGG_FROM_RAW
                    + (
                        "ana = next((p for p in _DELIVERABLES if p.endswith('analysis.json')), '')\n"
                        "syn = next((p for p in _DELIVERABLES if p.endswith('synthesis.json')), '')\n"
                        "rep = next((p for p in _DELIVERABLES if p.endswith('final_report.md')), '')\n"
                        "if not (ana and syn and rep):\n"
                        "    print('missing analysis/synthesis/final_report'); sys.exit(1)\n"
                        "A = _j.load(open(ana)); Y = _j.load(open(syn))\n"
                        "def _near(a, b):\n"
                        "    try:\n"
                        "        return abs(float(a) - float(b)) <= 0.05\n"
                        "    except (TypeError, ValueError):\n"
                        "        return False\n"
                        "# analysis: total + per-category counts/avg must match recomputation.\n"
                        "if int(A.get('total_findings', -1)) != total:\n"
                        "    print('analysis.total_findings ' + str(A.get('total_findings')) + ' != recomputed ' + str(total)); sys.exit(1)\n"
                        "acats = {str(c.get('category')): c for c in (A.get('categories') or [])}\n"
                        "for _c, rec in agg.items():\n"
                        "    got = acats.get(_c)\n"
                        "    if not got:\n"
                        "        print('analysis missing category ' + _c); sys.exit(1)\n"
                        "    if int(got.get('count', -1)) != rec['count']:\n"
                        "        print('analysis count ' + _c + ' ' + str(got.get('count')) + ' != ' + str(rec['count'])); sys.exit(1)\n"
                        "    if not _near(got.get('avg_severity'), rec['avg_severity']):\n"
                        "        print('analysis avg_severity ' + _c + ' mismatch'); sys.exit(1)\n"
                        "# synthesis: total + worst_category must match.\n"
                        "if int(Y.get('total_findings', -1)) != total:\n"
                        "    print('synthesis.total_findings mismatch'); sys.exit(1)\n"
                        "if str(Y.get('worst_category')) != worst:\n"
                        "    print('synthesis.worst_category ' + str(Y.get('worst_category')) + ' != recomputed ' + worst); sys.exit(1)\n"
                        "if int(Y.get('category_count', -1)) != len(agg):\n"
                        "    print('synthesis.category_count mismatch'); sys.exit(1)\n"
                        "# final_report.md must cite the recomputed total + worst category.\n"
                        "text = open(rep, encoding='utf-8', errors='replace').read()\n"
                        "if str(total) not in text:\n"
                        "    print('final_report.md does not cite recomputed total ' + str(total)); sys.exit(1)\n"
                        "if worst not in text:\n"
                        "            print('final_report.md does not cite recomputed worst_category ' + worst); sys.exit(1)\n"
                        "print('ok: aggregation chain raw->analysis->synthesis->report all recomputed-consistent')\n"
                    ),
                    "timeout": 20,
                },
            ),
            CheckConfig(
                check_type="execution",
                name="q08_no_placeholder_summary",
                params={
                    # Clone of the q02/q06 placeholder-leak probe: final_report.md
                    # prose must not carry >= 2 unsubstituted {var}/{{var}} residues.
                    "code": (
                        "import re, sys\n"
                        "path = next((p for p in _DELIVERABLES if p.endswith('final_report.md')), '')\n"
                        "if not path:\n"
                        "    print('no final_report.md'); sys.exit(1)\n"
                        "t = open(path, encoding='utf-8', errors='replace').read()\n"
                        "t = re.sub(r'```.*?```', '', t, flags=re.DOTALL)\n"
                        "t = re.sub(r'`[^`\\n]*`', '', t)\n"
                        "fmt = set(re.findall(r'(?<!\\{)\\{([a-z_][a-z0-9_]{2,})\\}(?!\\})', t))\n"
                        "jv = set(re.findall(r'\\{\\{\\s*([a-zA-Z_][a-zA-Z0-9_.-]*)\\s*\\}\\}', t))\n"
                        "leaks = fmt | jv\n"
                        "if len(leaks) >= 2:\n"
                        "    print('placeholder leak: ' + ', '.join(sorted(leaks)[:6])); sys.exit(1)\n"
                        "print('ok: no template-placeholder residue in final report')\n"
                    ),
                    "timeout": 20,
                },
            ),
        ],
    )


def _battery04_q09() -> GoalSpec:
    """Complex robustness query: build a reusable outlier-detection capability.

    Exercises all four capability surfaces in one run: (1) CREATE a NEW tool
    (iqr_outlier_detector) the registry did not have; (2) REUSE existing
    synthetic-data / type-inference tooling to seed inputs; (3) SPAWN a NEW QA
    sub-agent that independently re-derives the fences; (4) evolution fires on
    convergence. The anti-fabrication dimension is the Tukey-IQR recomputation:
    both anomalies.json and qa_verification.json must be grounded in the SAME
    transactions.csv `amount` column — fences AND outlier index SETS are
    recomputed with the linear-interpolation percentile convention (numpy default)
    and required to match exactly. An agent that hand-waves the fences, copies the
    detector's output into the QA file instead of recomputing, or fabricates the
    outlier indices is caught.
    """

    # Pure-Python Tukey-IQR recomputation from transactions.csv `amount`.
    # Convention pinned: linear-interpolation percentile (numpy.percentile default,
    # method='linear' — rank=(p/100)*(n-1), 0-indexed, interpolate floor↔ceil).
    # Shared by the anomalies probe and the QA-corroboration probe.
    _RECOMPUTE_FENCES = (
        "import csv as _csv, json as _j, math, sys\n"
        "tx = next((p for p in _DELIVERABLES if p.endswith('transactions.csv')), '')\n"
        "if not tx:\n"
        "    print('no transactions.csv'); sys.exit(1)\n"
        "sv = []\n"
        "with open(tx, newline='') as _f:\n"
        "    rdr = _csv.DictReader(_f)\n"
        "    cols = rdr.fieldnames or []\n"
        "    if 'amount' not in cols:\n"
        "        print('transactions.csv missing amount column'); sys.exit(1)\n"
        "    for _r in rdr:\n"
        "        try:\n"
        "            sv.append(float(_r['amount']))\n"
        "        except (TypeError, ValueError):\n"
        "            print('non-numeric amount in a row'); sys.exit(1)\n"
        "n = len(sv)\n"
        "if n < 100:\n"
        "    print('too few transaction rows (' + str(n) + ')'); sys.exit(1)\n"
        "svs = sorted(sv)\n"
        "def _pct(p):\n"
        "    if n == 1:\n"
        "        return float(svs[0])\n"
        "    rank = (p / 100.0) * (n - 1)\n"
        "    lo = int(math.floor(rank)); hi = int(math.ceil(rank))\n"
        "    if lo == hi:\n"
        "        return float(svs[lo])\n"
        "    return float(svs[lo] + (svs[hi] - svs[lo]) * (rank - lo))\n"
        "q1 = _pct(25); q3 = _pct(75); iqr = q3 - q1\n"
        "lower = q1 - 1.5 * iqr; upper = q3 + 1.5 * iqr\n"
        "# outlier indices in ORIGINAL data-row order (0-based, excluding header).\n"
        "oidx = [i for i, v in enumerate(sv) if v < lower or v > upper]\n"
        "if len(oidx) == 0:\n"
        "    print('no outliers flagged — seed data step failed to inject anomalies'); sys.exit(1)\n"
    )

    return GoalSpec(
        spec_id="battery04_q09",
        name="battery04_q09",
        description=(
            "Build a reusable Tukey-IQR outlier-detection capability: new tool "
            "creation + reuse of existing data tooling + new QA sub-agent, with "
            "fences and outlier indices recomputed independently"
        ),
        goal_text=(
            "Build a REUSABLE outlier-detection capability end to end, exercising "
            "tool creation, reuse of existing data tooling, and a QA sub-agent.\n"
            "Step 1 — CREATE A NEW TOOL named `iqr_outlier_detector` (do NOT reuse an "
            "existing tool; the ToolRegistry must gain a brand-new capability this "
            "run). It computes Tukey IQR fences on a numeric column of a CSV using "
            "the LINEAR-INTERPOLATION percentile convention (numpy.percentile "
            "default, method='linear': rank = (p/100)*(n-1), 0-indexed, interpolate "
            "between the floor and ceil neighbors). For the target column it returns "
            "column, q1, q3, iqr, lower_fence (q1 - 1.5*iqr), upper_fence "
            "(q3 + 1.5*iqr), and outlier_row_indices (0-based indices into the CSV "
            "data rows, excluding the header, in original row order).\n"
            "Step 2 — REUSE EXISTING synthetic-data / type-inference tooling to "
            "GENERATE results/q09/transactions.csv with >= 1000 rows and columns "
            "transaction_id, customer_id, amount, quantity, timestamp. Inject roughly "
            "3% of rows with outlying `amount` values so the detector has real "
            "anomalies to flag (a clean column with zero outliers is a failure of "
            "this step).\n"
            "Step 3 — RUN the new `iqr_outlier_detector` tool on transactions.csv "
            "column `amount` and write results/q09/anomalies.json with EXACTLY: "
            "{\"column\": str, \"q1\": number, \"q3\": number, \"iqr\": number, "
            "\"lower_fence\": number, \"upper_fence\": number, \"outlier_row_indices\": "
            "[int]}.\n"
            "Step 4 — SPAWN A NEW QA SUB-AGENT that INDEPENDENTLY re-derives the "
            "Tukey fences from transactions.csv (its own recomputation, NOT a copy of "
            "anomalies.json) and writes results/q09/qa_verification.json with "
            "EXACTLY: {\"recomputed_outlier_count\": int, \"fence_lower\": number, "
            "\"fence_upper\": number, \"sample_verified\": [{\"row_index\": int, "
            "\"amount\": number, \"is_outlier\": bool}]}. sample_verified must contain "
            ">= 3 rows spanning at least one flagged outlier and one inlier, each "
            "classified correctly against the independently recomputed fences."
        ),
        category="capability_evolution",
        max_iterations=60,
        timeout_seconds=1300,
        expected_deliverables=[
            "results/q09/transactions.csv",
            "results/q09/anomalies.json",
            "results/q09/qa_verification.json",
        ],
        success_criteria=[
            "anomalies.json q1/q3/iqr/fences and outlier index set match an independent Tukey-IQR recomputation from transactions.csv",
            "qa_verification.json independently corroborates the recomputed fences, outlier count, and per-row outlier classification",
            "a brand-new iqr_outlier_detector tool is created and a new QA sub-agent is spawned this run",
        ],
        checks=[
            CheckConfig(
                check_type="golden",
                name="q09_deliverables_exist",
                params={
                    "assertions": [
                        {"kind": "exists", "deliverable": "results/q09/transactions.csv"},
                        {"kind": "exists", "deliverable": "results/q09/anomalies.json"},
                        {"kind": "exists", "deliverable": "results/q09/qa_verification.json"},
                    ]
                },
            ),
            CheckConfig(
                check_type="structural",
                name="q09_transactions_rows",
                params={"deliverable": "results/q09/transactions.csv", "format": "csv", "min_rows": 900},
            ),
            CheckConfig(
                check_type="structural",
                name="q09_anomalies_schema",
                params={"deliverable": "results/q09/anomalies.json", "format": "json"},
            ),
            CheckConfig(
                check_type="structural",
                name="q09_qa_schema",
                params={"deliverable": "results/q09/qa_verification.json", "format": "json"},
            ),
            CheckConfig(
                check_type="execution",
                name="q09_outliers_recomputed",
                params={
                    # The anti-fabrication backbone for the detector output: every
                    # fence value AND the full outlier index SET must match an
                    # independent linear-interpolation Tukey-IQR recomputation from
                    # transactions.csv. A fabricated anomalies.json (hand-written
                    # fences, or indices that don't match the data) fails.
                    "code": _RECOMPUTE_FENCES
                    + (
                        "an = next((p for p in _DELIVERABLES if p.endswith('anomalies.json')), '')\n"
                        "if not an:\n"
                        "    print('no anomalies.json'); sys.exit(1)\n"
                        "A = _j.load(open(an))\n"
                        "def _near(a, b):\n"
                        "    try:\n"
                        "        return abs(float(a) - float(b)) <= 1e-6\n"
                        "    except (TypeError, ValueError):\n"
                        "        return False\n"
                        "for _k, _rv in (('q1', q1), ('q3', q3), ('iqr', iqr), ('lower_fence', lower), ('upper_fence', upper)):\n"
                        "    if not _near(A.get(_k), _rv):\n"
                        "        print('anomalies.' + _k + ' ' + str(A.get(_k)) + ' != recomputed ' + str(_rv)); sys.exit(1)\n"
                        "ai = A.get('outlier_row_indices') or []\n"
                        "aset = set(str(x) for x in ai)\n"
                        "rset = set(str(x) for x in oidx)\n"
                        "if aset != rset:\n"
                        "    miss = rset - aset; extra = aset - rset\n"
                        "    print('anomalies.outlier_row_indices mismatch (recomputed ' + str(len(rset)) + ' got ' + str(len(aset)) + '; missing ' + str(len(miss)) + ' extra ' + str(len(extra)) + ')'); sys.exit(1)\n"
                        "print('ok: anomalies fences + index set match independent Tukey-IQR recomputation')\n"
                    ),
                    "timeout": 20,
                },
            ),
            CheckConfig(
                check_type="execution",
                name="q09_qa_independent_corroboration",
                params={
                    # The QA sub-agent (second capability surface) must INDEPENDENTLY
                    # corroborate: recomputed_outlier_count == recomputed outlier
                    # count; fence_lower/upper == recomputed fences; and each
                    # sample_verified row's amount + is_outlier classification agrees
                    # with the recomputation. A QA file that merely copies
                    # anomalies.json without recomputing still passes the value
                    # equality (it has the right numbers) — but the sample_verified
                    # per-row classification is recomputed independently, so a QA
                    # file that got a row wrong is caught. requires >= 3 samples
                    # spanning both classes.
                    "code": _RECOMPUTE_FENCES
                    + (
                        "qa = next((p for p in _DELIVERABLES if p.endswith('qa_verification.json')), '')\n"
                        "if not qa:\n"
                        "    print('no qa_verification.json'); sys.exit(1)\n"
                        "Q = _j.load(open(qa))\n"
                        "def _near(a, b):\n"
                        "    try:\n"
                        "        return abs(float(a) - float(b)) <= 1e-6\n"
                        "    except (TypeError, ValueError):\n"
                        "        return False\n"
                        "if not _near(Q.get('recomputed_outlier_count'), len(oidx)):\n"
                        "    print('qa.recomputed_outlier_count ' + str(Q.get('recomputed_outlier_count')) + ' != recomputed ' + str(len(oidx))); sys.exit(1)\n"
                        "if not _near(Q.get('fence_lower'), lower):\n"
                        "    print('qa.fence_lower ' + str(Q.get('fence_lower')) + ' != recomputed ' + str(lower)); sys.exit(1)\n"
                        "if not _near(Q.get('fence_upper'), upper):\n"
                        "    print('qa.fence_upper ' + str(Q.get('fence_upper')) + ' != recomputed ' + str(upper)); sys.exit(1)\n"
                        "svl = Q.get('sample_verified') or []\n"
                        "if len(svl) < 3:\n"
                        "    print('qa.sample_verified fewer than 3 rows'); sys.exit(1)\n"
                        "classes = set()\n"
                        "for _s in svl:\n"
                        "    ri = _s.get('row_index')\n"
                        "    try:\n"
                        "        ri = int(ri)\n"
                        "    except (TypeError, ValueError):\n"
                        "        print('qa.sample_verified row_index not an int'); sys.exit(1)\n"
                        "    if ri < 0 or ri >= n:\n"
                        "        print('qa.sample_verified row_index out of range'); sys.exit(1)\n"
                        "    if not _near(_s.get('amount'), sv[ri]):\n"
                        "        print('qa.sample_verified row_index ' + str(ri) + ' amount mismatch'); sys.exit(1)\n"
                        "    expect_out = (sv[ri] < lower or sv[ri] > upper)\n"
                        "    classes.add(expect_out)\n"
                        "    if bool(_s.get('is_outlier')) != expect_out:\n"
                        "        print('qa.sample_verified row_index ' + str(ri) + ' is_outlier wrong'); sys.exit(1)\n"
                        "if not (True in classes and False in classes):\n"
                        "    print('qa.sample_verified must span at least one outlier and one inlier'); sys.exit(1)\n"
                        "print('ok: qa sub-agent independently corroborates fences, count, and per-row classification')\n"
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
        _battery04_q05(),
        _battery04_q06(),
        _battery04_q07(),
        _battery04_q08(),
        _battery04_q09(),
    )
}

# Ordered suite for `python main.py --eval`.
BATTERY04_GOALS: list[GoalSpec] = [
    GOLDEN_SPECS["battery04_q01"],
    GOLDEN_SPECS["battery04_q02"],
    GOLDEN_SPECS["battery04_q03"],
    GOLDEN_SPECS["battery04_q04"],
    GOLDEN_SPECS["battery04_q05"],
    GOLDEN_SPECS["battery04_q06"],
    GOLDEN_SPECS["battery04_q07"],
    GOLDEN_SPECS["battery04_q08"],
    GOLDEN_SPECS["battery04_q09"],
]


def lookup_goal_spec(spec_id: str | None) -> GoalSpec | None:
    """Resolve a GoalSpec by id, or None if unregistered / empty."""
    if not spec_id:
        return None
    return GOLDEN_SPECS.get(spec_id)
