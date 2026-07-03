"""Learning probes — GoalSpecs that exercise the *cross-run* self-improvement
signal the Battery-04 goals cannot show on their own.

The 9-goal battery is **intra-run** (each goal is independent; q01's created tool
does not seed q02). The thesis — *does a prior run's crystallized capability
improve a later run* — needs goals that repeat across generations so recall of an
earlier run's tool/skill/prompt is both **possible** and **measurable**.

This module supplies three such probes, registered into ``GOLDEN_SPECS`` (so the
verify node + ``--score-spec`` resolve them) but **deliberately excluded from
``BATTERY04_GOALS``** (mirroring the classify canaries) so the nightly
capability-curve battery stays unperturbed:

- ``probe_create_tool`` — cold run: build a price-normalization transform from a
  deterministic input, write ``normalized.csv``. Exercises tool-**creation**.
- ``probe_reuse_tool`` — the **paired recall run**: same transform logic, new
  input; the goal nudges "if a ``normalize_prices`` capability already exists,
  reuse it". Exercises tool-**reuse**. Run cold→create then warm→reuse: a recall
  link that fires should finish *cheaper / fewer LLM calls / fewer verify cycles*
  (the efficiency delta is the self-improvement signal; the golden check only
  verifies correctness via recomputation).
- ``probe_analytics_recall`` — a repeatable analytical goal (deterministic stats)
  for the same-goal-recall test: run it twice across generations and compare.

**Anti-fabrication backbone.** Every probe's ``execution`` check RECOMPUTES the
expected output from the deterministic input embedded in the goal text and
compares it to the on-disk deliverable (``sys.exit(1)`` on mismatch). A
hallucinated or stale deliverable cannot pass — it must contain the exact
recomputed values. (Same technique as q07/q08 handoff integrity.)

The probes are **data-correctness only** (no ``state`` check); ``target_node`` is
``None`` (universal) like q01…q09.
"""

from __future__ import annotations

from src.eval.models import CheckConfig, GoalSpec

# Fixed exchange rates (USD target). Embedded in both create + reuse goal texts so
# the transform is identical and recall is a valid signal.
_EUR_USD = 1.10
_GBP_USD = 1.27
_JPY_USD = 0.0067


def _recompute_prices_code(n_rows: int, expected: dict[str, float]) -> str:
    """Build the sandbox recompute probe for a price-normalization deliverable.

    Recomputes the EXPECTED ``{sku: price_usd}`` map from the known input + rates
    (handed in via ``expected``) and asserts the deliverable matches it exactly:
    right row count (dedup happened), distinct skus, each value within 0.01.
    A fabricated or non-deduped deliverable fails. Anti-fabrication.
    """
    return (
        "import csv, sys\n"
        "path = next((p for p in _DELIVERABLES if p.endswith('normalized.csv')), '')\n"
        "if not path:\n"
        "    print('no normalized.csv'); sys.exit(1)\n"
        "rows = list(csv.DictReader(open(path, newline='')))\n"
        f"if len(rows) != {n_rows}:\n"
        f"    print('expected {n_rows} deduped rows, got %d' % len(rows)); sys.exit(1)\n"
        "skus = [(r.get('sku') or '').strip() for r in rows]\n"
        f"if len(set(skus)) != {n_rows}:\n"
        "    print('duplicate skus remain: %s' % skus); sys.exit(1)\n"
        f"expected = {expected!r}\n"
        "got = {}\n"
        "for r in rows:\n"
        "    sku = (r.get('sku') or '').strip()\n"
        "    try:\n"
        "        got[sku] = round(float(r.get('price_usd')), 2)\n"
        "    except (TypeError, ValueError):\n"
        "        print('bad price_usd for %s' % sku); sys.exit(1)\n"
        "if set(got) != set(expected):\n"
        "    print('sku mismatch: got %s expected %s' % (sorted(got), sorted(expected))); sys.exit(1)\n"
        "for sku, v in expected.items():\n"
        "    if abs(got[sku] - v) > 0.01:\n"
        "        print('%s: got %.2f expected %.2f' % (sku, got[sku], v)); sys.exit(1)\n"
        f"print('ok: {n_rows} rows normalized correctly')\n"
    )


# Deterministic expected outputs, recomputed from the inputs in the goal texts.
# (EUR x1.10, GBP x1.27, JPY x0.0067, round 2dp; duplicate sku → last row wins.)
_CREATE_EXPECTED = {"A001": 13.75, "A002": 0.67, "A003": 6.35}  # 4 input rows → 3 deduped
_REUSE_EXPECTED = {"B001": 31.75, "B002": 50.00, "B003": 6.70, "B004": 8.89}  # 5 → 4 deduped


def _probe_create_tool() -> GoalSpec:
    """Cold run: build a price-normalization transform and emit normalized.csv."""
    return GoalSpec(
        spec_id="probe_create_tool",
        name="probe_create_tool",
        description=(
            "Learning probe (create-step): build a deterministic price-normalization "
            "capability cold. Paired with probe_reuse_tool for the cross-run recall test."
        ),
        goal_text=(
            "Create a REUSABLE tool named `normalize_prices` that takes a list of "
            "product price rows (each row: sku, price, currency) and normalizes every "
            f"price to USD using fixed rates EUR={_EUR_USD}, GBP={_GBP_USD}, "
            f"JPY={_JPY_USD}; deduplicates by sku keeping the LAST occurrence; and "
            "rounds each USD price to 2 decimals (USD passes through unchanged). "
            "The tool must return the normalized rows. Register the tool so it is "
            "available for reuse, then USE the `normalize_prices` tool to process "
            "this input:\n"
            "  A001, 10.00, EUR\n"
            "  A001, 12.50, EUR   # duplicate sku — keep the LAST occurrence\n"
            "  A002, 100,   JPY\n"
            "  A003, 5.00,  GBP\n"
            "and write its output as a CSV with columns sku,price_usd to "
            "results/probe_create/normalized.csv."
        ),
        category="tool_creation",
        max_iterations=40,
        timeout_seconds=600,
        expected_deliverables=["results/probe_create/normalized.csv"],
        success_criteria=[
            "results/probe_create/normalized.csv exists and is valid CSV",
            "Prices are correctly converted to USD and rounded to 2 decimals",
            "Duplicate sku rows are collapsed (last wins)",
        ],
        checks=[
            CheckConfig(
                check_type="golden",
                name="probe_create_deliverable_exists",
                params={
                    "assertions": [
                        {"kind": "exists", "deliverable": "results/probe_create/normalized.csv"},
                    ]
                },
            ),
            CheckConfig(
                check_type="structural",
                name="probe_create_csv_schema",
                params={
                    "deliverable": "results/probe_create/normalized.csv",
                    "format": "csv",
                    "min_rows": 1,
                },
            ),
            CheckConfig(
                check_type="execution",
                name="probe_create_recompute_probe",
                params={
                    "code": _recompute_prices_code(len(_CREATE_EXPECTED), _CREATE_EXPECTED),
                    "timeout": 20,
                },
            ),
        ],
    )


def _probe_reuse_tool() -> GoalSpec:
    """Paired recall run: apply the SAME normalization logic to new data, reusing
    the capability created in probe_create_tool if it persists across runs."""
    return GoalSpec(
        spec_id="probe_reuse_tool",
        name="probe_reuse_tool",
        description=(
            "Learning probe (reuse-step): apply the price-normalization capability to "
            "new data. If a normalize_prices tool/skill already exists in the registry "
            "or memory, REUSE it rather than rebuilding — the recall efficiency delta "
            "vs probe_create_tool is the self-improvement signal."
        ),
        goal_text=(
            "You are given the following product price table (sku, price, currency):\n"
            "  B001, 20.00, EUR\n"
            "  B001, 25.00, GBP   # duplicate sku — keep the LAST occurrence\n"
            "  B002, 50.00, USD\n"
            "  B003, 1000,  JPY\n"
            "  B004, 7.00,  GBP\n"
            f"\nNormalize every price to USD using these fixed rates: "
            f"EUR={_EUR_USD}, GBP={_GBP_USD}, JPY={_JPY_USD} "
            "(USD is already USD; pass through unchanged). "
            "Deduplicate by sku (last row wins), round each USD price to 2 decimals, "
            "and write a CSV with columns sku,price_usd to "
            "results/probe_reuse/normalized.csv.\n\n"
            "If a normalize_prices capability (tool or skill) already exists from a "
            "prior run, REUSE it instead of rebuilding from scratch."
        ),
        category="tool_creation",
        max_iterations=40,
        timeout_seconds=600,
        expected_deliverables=["results/probe_reuse/normalized.csv"],
        success_criteria=[
            "results/probe_reuse/normalized.csv exists and is valid CSV",
            "Prices are correctly converted to USD and rounded to 2 decimals",
            "Duplicate sku rows are collapsed (last wins)",
        ],
        checks=[
            CheckConfig(
                check_type="golden",
                name="probe_reuse_deliverable_exists",
                params={
                    "assertions": [
                        {"kind": "exists", "deliverable": "results/probe_reuse/normalized.csv"},
                    ]
                },
            ),
            CheckConfig(
                check_type="structural",
                name="probe_reuse_csv_schema",
                params={
                    "deliverable": "results/probe_reuse/normalized.csv",
                    "format": "csv",
                    "min_rows": 1,
                },
            ),
            CheckConfig(
                check_type="execution",
                name="probe_reuse_recompute_probe",
                params={
                    "code": _recompute_prices_code(len(_REUSE_EXPECTED), _REUSE_EXPECTED),
                    "timeout": 20,
                },
            ),
        ],
    )


# Deterministic expected stats for the analytics probe, recomputed from the series
# [50,100,150,200,250,300,400]: count=7, sum=1450, mean≈207.1429, median=200.
_ANALYTICS_EXPECTED = {
    "count": 7,
    "sum": 1450,
    "mean": 207.142857,
    "median": 200,
    "min": 50,
    "max": 400,
}


def _probe_analytics_recall() -> GoalSpec:
    """Repeatable analytical goal: compute deterministic summary stats → stats.json.

    Run across generations; a prior run's crystallized stats/skill recall should
    make later generations converge cheaper/faster (efficiency signal).
    """
    return GoalSpec(
        spec_id="probe_analytics_recall",
        name="probe_analytics_recall",
        description=(
            "Learning probe (same-goal recall): deterministic summary statistics. "
            "Repeatable across generations to measure recall-driven efficiency gains."
        ),
        goal_text=(
            "Compute summary statistics for this series of numbers:\n"
            "  50, 100, 150, 200, 250, 300, 400\n\n"
            "Write a JSON object with the keys count, sum, mean, median, min, max to "
            "results/probe_analytics/stats.json. Round mean to at least 4 decimals."
        ),
        category="simple",
        max_iterations=25,
        timeout_seconds=300,
        expected_deliverables=["results/probe_analytics/stats.json"],
        success_criteria=[
            "results/probe_analytics/stats.json exists and is valid JSON",
            "count=7, sum=1450, mean≈207.1429, median=200, min=50, max=400",
        ],
        checks=[
            CheckConfig(
                check_type="golden",
                name="probe_analytics_deliverable_exists",
                params={
                    "assertions": [
                        {"kind": "exists", "deliverable": "results/probe_analytics/stats.json"},
                    ]
                },
            ),
            CheckConfig(
                check_type="structural",
                name="probe_analytics_json_schema",
                params={
                    "deliverable": "results/probe_analytics/stats.json",
                    "format": "json",
                },
            ),
            CheckConfig(
                check_type="execution",
                name="probe_analytics_recompute_probe",
                params={
                    "code": (
                        "import json, sys\n"
                        "path = next((p for p in _DELIVERABLES if p.endswith('stats.json')), '')\n"
                        "if not path:\n"
                        "    print('no stats.json'); sys.exit(1)\n"
                        "with open(path) as _f:\n"
                        "    d = json.load(_f)\n"
                        f"expected = {_ANALYTICS_EXPECTED!r}\n"
                        "def _find(key):\n"
                        "    for k, v in d.items():\n"
                        "        if str(k).lower() == key:\n"
                        "            return v\n"
                        "    return None\n"
                        "for key, want in expected.items():\n"
                        "    got = _find(key)\n"
                        "    if got is None:\n"
                        "        print('missing field %s' % key); sys.exit(1)\n"
                        "    try:\n"
                        "        gotv = float(got)\n"
                        "    except (TypeError, ValueError):\n"
                        "        print('field %s not numeric: %r' % (key, got)); sys.exit(1)\n"
                        "    if abs(gotv - want) > 0.01:\n"
                        "        print('%s: got %.4f expected %.4f' % (key, gotv, want)); sys.exit(1)\n"
                        "print('ok: all stats verified')\n"
                    ),
                    "timeout": 20,
                },
            ),
        ],
    )


# Ordered list — the experiment enqueuer iterates this; wiring into GOLDEN_SPECS
# makes each resolvable via lookup_goal_spec(spec_id).
LEARNING_PROBES: list[GoalSpec] = [
    _probe_create_tool(),
    _probe_reuse_tool(),
    _probe_analytics_recall(),
]
