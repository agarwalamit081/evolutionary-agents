#!/usr/bin/env python3
"""Battery-02 driver: 10 PAIRED recall queries (create → reuse).

Distinct from ``run_battery.py`` (battery-01). Q(n) creates a capability, Q(n+1)
is written to NEED and INVOKE it, so cross-run persistence + recall is observed
by the capability actually being called — not just a startup log line.

Enhancements over the original harness:
  - per-query UTC ``t_start``/``t_end`` (epoch + ISO) for clean cost attribution,
  - DB snapshots BEFORE and AFTER each query → per-table deltas (not just totals),
  - recall-evidence grep of the per-query structured log (``logs/<run_id>.log``)
    for the ``Loaded N dynamic tools/sub-agents from DB`` startup lines + the
    target capability name,
  - persisted ``monitor/battery_02_summary.json`` (the console table is ephemeral).

Each query runs as ``python main.py --goal "..." --run-id nN`` against the
canonical docker stack (PG 5433 / redis 6380) on the resolved default model
(no ``--model``), so the app's own key resolution + fallback + timeout path is
exercised exactly as production. Robust by design: every query is isolated in
try/except so one failure never aborts the batch. NOT run in CI.

Usage::

    python scripts/run_battery_02.py            # all 10 (N1..N10)
    python scripts/run_battery_02.py n1 n2      # pilot subset only
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import text

# Reuse the exact SQL the counts probe uses (same scripts/ dir) — single source
# of truth for which tables the rubric measures.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _battery_counts import _QUERIES as COUNT_QUERIES  # noqa: E402

from src.db.session import close_db, get_session  # noqa: E402

# (run_id, kind, goal, deliverable, recall_target). Recall target = the tool or
# sub-agent name this query creates (odd N) or must invoke (even N).
QUERIES: list[tuple[str, str, str, str, str]] = [
    (
        "n1", "tool create",
        "Create a short, simple tool called char_counter that returns character, "
        "word, and line counts for a text string. Use it on the README's first "
        "paragraph and save the metrics to results/n1_char_counter.md.",
        "results/n1_char_counter.md", "char_counter",
    ),
    (
        "n2", "tool reuse",
        "Using the char_counter tool created earlier, compare src/graph/state.py "
        "and src/graph/enums.py; save the side-by-side counts to results/n2_char_compare.md.",
        "results/n2_char_compare.md", "char_counter",
    ),
    (
        "n3", "sub-agent create",
        "Spawn a dedicated sub-agent called python_file_inventory that walks a "
        "directory and reports Python file count and total lines by subdirectory. "
        "Delegate the repo root to it and save the report to results/n3_inventory.md.",
        "results/n3_inventory.md", "python_file_inventory",
    ),
    (
        "n4", "sub-agent reuse",
        "Delegate to the existing python_file_inventory sub-agent to inventory "
        "tests/ versus src/ and save the comparison to results/n4_inventory_tests.md.",
        "results/n4_inventory_tests.md", "python_file_inventory",
    ),
    (
        "n5", "tool create",
        "Create a short tool called duplicate_finder that takes a glob and returns "
        "duplicate non-empty lines across matched files. Run it on results/*.md and "
        "save the output to results/n5_dups.md.",
        "results/n5_dups.md", "duplicate_finder",
    ),
    (
        "n6", "tool reuse",
        "Using the duplicate_finder and char_counter tools created earlier, scan "
        "logs/*.log for duplicate error lines and summarize the counts in results/n6_log_dups.md.",
        "results/n6_log_dups.md", "duplicate_finder",
    ),
    (
        "n7", "sub-agent create",
        "Spawn a sub-agent called doc_outline that reads a markdown doc and emits "
        "its section outline (headings plus per-section word counts). Run it on "
        "docs/design-docs/00-project-overview.md and save the outline to results/n7_outline.md.",
        "results/n7_outline.md", "doc_outline",
    ),
    (
        "n8", "sub-agent reuse",
        "Using the doc_outline and python_file_inventory sub-agents created earlier, "
        "build a combined repo map (file inventory plus doc outline) and save it to "
        "results/n8_repomap.md.",
        "results/n8_repomap.md", "python_file_inventory",
    ),
    (
        "n9", "evolution",
        "Read docs/design-docs/00-project-overview.md and write a concise onboarding "
        "summary (purpose, architecture layers, how to run) to results/n9_onboarding.md "
        "with at least 3 sections.",
        "results/n9_onboarding.md", "evolve",
    ),
    (
        "n10", "evolution + reuse",
        "Read pyproject.toml and the tests/ directory and write a testing guide "
        "(how to run unit/integration/e2e tests, the three test layers, key fixtures) "
        "to results/n10_testing.md with at least 3 sections.",
        "results/n10_testing.md", "evolve",
    ),
]

PER_QUERY_TIMEOUT = 900  # 15-min hard cap; a hang is killed, not the whole batch
REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "logs"
RESULTS_DIR = REPO_ROOT / "results"
MONITOR_DIR = REPO_ROOT / "monitor"

# Recall evidence we grep for in the per-query structured log.
_RECALL_TOOL_RE = re.compile(r"Loaded (\d+) dynamic tools from DB:(.*)")
_RECALL_AGENT_RE = re.compile(r"Loaded (\d+) sub-agents from DB:(.*)")


async def _snapshot() -> dict[str, Any]:
    """Current row counts/spend across every rubric table (read-only).

    Closes the DB after reading: this driver calls ``_snapshot`` across many
    separate ``asyncio.run()`` loops (baseline, per-query before/after, post).
    ``get_engine()``/``_get_session_factory()`` cache singletons bound to the
    FIRST loop, so without disposal every later snapshot silently fails on a
    dead loop and the per-query deltas come back empty (Bug B — observed: all
    Δtools/Δagents/Δ$ were None). ``close_db()`` resets both the session factory
    and the engine so the next snapshot re-binds a fresh engine to its own loop.
    """
    out: dict[str, object] = {}
    try:
        async with get_session() as session:
            for name, sql in COUNT_QUERIES.items():
                val = (await session.execute(text(sql))).scalar_one()
                if isinstance(val, float):
                    val = round(float(val), 4)
                out[name] = val
    finally:
        await close_db()
    return out


def _delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return {k: after[k] - before.get(k, 0) for k in after}


def _grep_recall(run_id: str, target: str) -> dict[str, Any]:
    """Extract recall lines + target-name mentions from the per-query log."""
    log_path = LOG_DIR / f"{run_id}.log"
    evidence: dict[str, object] = {
        "tool_recall_line": None,
        "agent_recall_line": None,
        "target_mentions": 0,
        "log_exists": log_path.exists(),
    }
    if not log_path.exists():
        return evidence
    text_all = log_path.read_text(encoding="utf-8", errors="replace")
    m = _RECALL_TOOL_RE.search(text_all)
    if m:
        evidence["tool_recall_line"] = f"Loaded {m.group(1)}:{m.group(2).strip()}"
    m = _RECALL_AGENT_RE.search(text_all)
    if m:
        evidence["agent_recall_line"] = f"Loaded {m.group(1)}:{m.group(2).strip()}"
    # Count target capability mentions anywhere in the log (creation OR invocation).
    if target and target not in {"evolve"}:
        evidence["target_mentions"] = len(re.findall(re.escape(target), text_all))
    return evidence


def _clear_stale(deliverable: str) -> None:
    """Remove a prior deliverable so the run is a clean test."""
    path = REPO_ROOT / deliverable
    if path.exists():
        path.unlink()


def run_one(qid: str, kind: str, goal: str, deliverable: str, target: str) -> dict[str, Any]:
    """Run a single query; return a result record. Never raises."""
    _clear_stale(deliverable)
    stdout_path = LOG_DIR / f"{qid}.stdout.log"

    t_start = time.time()
    iso_start = datetime.fromtimestamp(t_start, tz=timezone.utc).isoformat()

    try:
        before: dict[str, Any] = asyncio.run(_snapshot())
    except Exception as exc:  # noqa: BLE001 — must not abort the batch
        before = {"error": str(exc)[:120]}

    status = "ok"
    exit_code: int | str = 0
    try:
        with stdout_path.open("w") as log:
            proc = subprocess.run(  # noqa: S603 — trusted local entrypoint
                [sys.executable, "main.py", "--goal", goal, "--run-id", qid],
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=PER_QUERY_TIMEOUT,
                check=False,
            )
        exit_code = proc.returncode
        if exit_code != 0:
            status = f"exit={exit_code}"
    except subprocess.TimeoutExpired:
        status = "timeout"
        exit_code = "timeout"
    except Exception as exc:  # noqa: BLE001 — driver must never abort the batch
        status = f"error: {exc!s:.80}"
        exit_code = "error"

    t_end = time.time()
    iso_end = datetime.fromtimestamp(t_end, tz=timezone.utc).isoformat()

    try:
        after: dict[str, Any] = asyncio.run(_snapshot())
        delta: dict[str, Any] = (
            _delta(before, after) if isinstance(before, dict) and "error" not in before else {}
        )
    except Exception as exc:  # noqa: BLE001
        after = {"error": str(exc)[:120]}
        delta = {}

    delivered = (REPO_ROOT / deliverable).exists()
    size = (REPO_ROOT / deliverable).stat().st_size if delivered else -1

    return {
        "qid": qid,
        "kind": kind,
        "recall_target": target,
        "status": status,
        "exit_code": exit_code,
        "t_start_epoch": round(t_start, 3),
        "t_end_epoch": round(t_end, 3),
        "t_start_iso": iso_start,
        "t_end_iso": iso_end,
        "elapsed_s": round(t_end - t_start, 1),
        "deliverable_present": delivered,
        "deliverable_bytes": size,
        "deliverable": deliverable,
        "stdout_log": str(stdout_path.relative_to(REPO_ROOT)),
        "structured_log": str((LOG_DIR / f"{qid}.log").relative_to(REPO_ROOT)),
        "db_delta": delta,
        "recall_evidence": _grep_recall(qid, target),
    }


def main() -> int:
    LOG_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    MONITOR_DIR.mkdir(exist_ok=True)

    # Subset selector: argv run-ids (e.g. "n1 n2") filter the queue; default all.
    wanted = {a.lower() for a in sys.argv[1:]}
    queue = [q for q in QUERIES if (not wanted or q[0] in wanted)]
    if not queue:
        print(f"No queries matched {sorted(wanted)}; valid: {[q[0] for q in QUERIES]}")
        return 2

    print(f"=== battery-02 start: {len(queue)} queries, default model, "
          f"per-query timeout {PER_QUERY_TIMEOUT}s ===", flush=True)
    try:
        baseline: dict[str, Any] = asyncio.run(_snapshot())
        print(f"baseline: cost_usd={baseline.get('cost_total_usd')} "
              f"tools={baseline.get('tool_registrations')} "
              f"agents={baseline.get('sub_agent_definitions')} "
              f"chains={baseline.get('mutation_chains')}", flush=True)
    except Exception as exc:  # noqa: BLE001
        baseline = {"error": str(exc)[:120]}
        print(f"baseline snapshot failed (continuing): {exc}", flush=True)

    results: list[dict[str, Any]] = []
    for qid, kind, goal, deliverable, target in queue:
        print(f"\n--- {qid} [{kind}] ({deliverable}) target={target} ---", flush=True)
        rec = run_one(qid, kind, goal, deliverable, target)
        results.append(rec)
        ev = rec["recall_evidence"]
        d = rec["db_delta"]
        print(
            f"  => {rec['status']} | {rec['elapsed_s']}s | "
            f"deliverable={'yes' if rec['deliverable_present'] else 'NO'} "
            f"({rec['deliverable_bytes']} B) | "
            f"Δtools={d.get('tool_registrations')} Δagents={d.get('sub_agent_definitions')} "
            f"Δchains={d.get('mutation_chains')} Δ$={d.get('cost_total_usd')} | "
            f"tool_recall={ev['tool_recall_line'] is not None} "
            f"agent_recall={ev['agent_recall_line'] is not None}", flush=True)

    try:
        post: dict[str, Any] = asyncio.run(_snapshot())
    except Exception as exc:  # noqa: BLE001
        post = {"error": str(exc)[:120]}

    summary = {
        "generated_iso": datetime.now(tz=timezone.utc).isoformat(),
        "per_query_timeout_s": PER_QUERY_TIMEOUT,
        "baseline": baseline,
        "post_run": post,
        "results": results,
    }
    out_path = MONITOR_DIR / "battery_02_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n=== SUMMARY ===", flush=True)
    n_ok = sum(1 for r in results if r["status"] == "ok" and r["deliverable_present"])
    n_exit = sum(1 for r in results if isinstance(r["exit_code"], int) and r["exit_code"] != 0)
    n_timeout = sum(1 for r in results if r["exit_code"] == "timeout")
    print(f"ok+deliverable: {n_ok}/{len(results)} | non-zero exit: {n_exit} | "
          f"timeout: {n_timeout}", flush=True)
    for r in results:
        d = r["db_delta"]
        print(f"  {r['qid']} [{r['kind']}]: {r['status']} | {r['elapsed_s']}s | "
              f"deliverable={'yes' if r['deliverable_present'] else 'NO'} | "
              f"Δtools={d.get('tool_registrations')} Δagents={d.get('sub_agent_definitions')} "
              f"Δchains={d.get('mutation_chains')}", flush=True)
    print(f"\nsummary written → {out_path.relative_to(REPO_ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
