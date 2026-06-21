#!/usr/bin/env python3
"""Sequential driver for the 10-query validation battery (queries 1-8 + 10).

Q9 (onboarding) is already validated as the pilot. This runs the remaining nine
goals one at a time against the canonical docker stack (PG 5433 / redis 6380)
on the resolved default model (no --model flag), so each query gets:

  - its own log file (logs/qN_deepseek.log) for per-query triage,
  - a per-query wall-clock + exit-code capture,
  - a hard per-query timeout (a hang is killed, not the whole batch),
  - clean DB attribution (sequential => no interleaved cost_ledger rows).

Robust by design: every query is isolated in try/except so one failure never
aborts the batch. A summary table is printed at the end. NOT run in CI; this is
an operator validation harness, not a unit test.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

# (id, goal, expected deliverable under results/). Q9 already validated.
QUERIES: list[tuple[str, str, str]] = [
    (
        "q1",
        "Create a tool called text_stats that computes word count, sentence count, "
        "and Flesch reading-ease for input text. Use it to analyze the README's first "
        "paragraph and save the metrics to results/q1_text_stats.md.",
        "results/q1_text_stats.md",
    ),
    (
        "q2",
        "Build a utility named palette_gen that produces a 5-color analogous palette "
        "from a base hex color. Generate 3 palettes and save them to results/q2_palettes.json.",
        "results/q2_palettes.json",
    ),
    (
        "q3",
        "Use specialized sub-agents in parallel to: (1) summarize the architecture from "
        "CLAUDE.md, (2) enumerate the built-in tools from src/tools/builtin, (3) count the "
        "design docs in docs/design-docs. Merge the results into results/q3_overview.md.",
        "results/q3_overview.md",
    ),
    (
        "q4",
        "Deploy dedicated sub-agents — one that audits the pinned versions in pyproject.toml, "
        "another that drafts a supply-chain risk note. Combine both into results/q4_deps.md.",
        "results/q4_deps.md",
    ),
    (
        "q5",
        "Analyze the 3-tier memory system in src/memory/manager.py, cold.py, and warm.py. "
        "Write a report with at least 3 sections covering the tiers, their interaction, and "
        "one improvement, saved to results/q5_memory.md.",
        "results/q5_memory.md",
    ),
    (
        "q6",
        "Document the LLM gateway: model routing, cost tracking, and fallback chains in "
        "src/llm/ and src/config/model_registry.py. Write a 3-section report saved to "
        "results/q6_llm.md.",
        "results/q6_llm.md",
    ),
    (
        "q7",
        "Implement a script-tool git_log_summary that summarizes recent commits (count, top "
        "authors, last 5 messages) for a repo path. Run it on this repo and save the output "
        "to results/q7_git.md.",
        "results/q7_git.md",
    ),
    (
        "q8",
        "Use sub-agents in parallel to: (1) extract the API endpoints from src/api, "
        "(2) list the graph nodes in src/graph/nodes, (3) summarize the safety pipeline in "
        "src/safety. Combine into a report saved to results/q8_system.md.",
        "results/q8_system.md",
    ),
    (
        "q10",
        "Read pyproject.toml and the tests/ directory and write a testing guide covering how "
        "to run unit/integration/e2e tests, the three test layers, and key fixtures. Save it "
        "to results/q10_testing.md with at least 3 sections.",
        "results/q10_testing.md",
    ),
]

PER_QUERY_TIMEOUT = 900  # 15 min hard cap per query (sub-agent delegations run nested subgraphs)
REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "logs"
RESULTS_DIR = REPO_ROOT / "results"


def _clear_stale(deliverable: str) -> None:
    """Remove a prior deliverable so the run is a clean test of the hardening."""
    path = REPO_ROOT / deliverable
    if path.exists():
        path.unlink()
        print(f"  cleared stale {deliverable}", flush=True)


def run_one(qid: str, goal: str, deliverable: str) -> dict[str, object]:
    """Run a single query; return a result record. Never raises."""
    _clear_stale(deliverable)
    log_path = LOG_DIR / f"{qid}_deepseek.log"
    start = time.monotonic()
    status = "ok"
    exit_code: int | str = 0
    try:
        with log_path.open("w") as log:
            proc = subprocess.run(  # noqa: S603 — trusted local entrypoint
                [sys.executable, "main.py", "--goal", goal],
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

    elapsed = time.monotonic() - start
    delivered = (REPO_ROOT / deliverable).exists()
    record: dict[str, object] = {
        "qid": qid,
        "status": status,
        "exit_code": exit_code,
        "elapsed_s": round(elapsed, 1),
        "deliverable_present": delivered,
        "deliverable": deliverable,
        "log": str(log_path.relative_to(REPO_ROOT)),
    }
    size = -1
    if delivered:
        size = (REPO_ROOT / deliverable).stat().st_size
    record["deliverable_bytes"] = size
    return record


def main() -> int:
    LOG_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"=== battery start: {len(QUERIES)} queries, default model, "
          f"per-query timeout {PER_QUERY_TIMEOUT}s ===", flush=True)
    results: list[dict[str, object]] = []
    for qid, goal, deliverable in QUERIES:
        print(f"\n--- {qid} ({deliverable}) ---", flush=True)
        rec = run_one(qid, goal, deliverable)
        results.append(rec)
        print(f"  => {rec['status']} | {rec['elapsed_s']}s | "
              f"deliverable={'yes' if rec['deliverable_present'] else 'NO'} "
              f"({rec['deliverable_bytes']} B) | log={rec['log']}", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    n_ok = sum(1 for r in results if r["status"] == "ok" and r["deliverable_present"])
    n_exit = sum(1 for r in results if isinstance(r["exit_code"], int) and r["exit_code"] != 0)
    n_timeout = sum(1 for r in results if r["exit_code"] == "timeout")
    print(f"ok+deliverable: {n_ok}/{len(results)} | non-zero exit: {n_exit} | "
          f"timeout: {n_timeout}", flush=True)
    for r in results:
        print(f"  {r['qid']}: {r['status']} | {r['elapsed_s']}s | "
              f"deliverable={'yes' if r['deliverable_present'] else 'NO'} "
              f"({r['deliverable_bytes']} B)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
