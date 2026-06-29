"""EXPLAIN ANALYZE harness for the pgvector HNSW queries.

Runs an ``EXPLAIN (ANALYZE)`` on a representative cosine-ANN query for each of
the four HNSW-indexed tables and reports whether the planner used the index (vs
a Seq Scan) plus the measured execution time. Per the pgvector/performance rule
("ALWAYS run EXPLAIN ANALYZE on vector queries to verify index usage").

Read-only — issues only ``SELECT``/``EXPLAIN``; it never mutates schema or data,
and never edits an applied migration. Any HNSW tuning that follows would be a
NEW migration, never an in-place edit of ``idx_*`` definitions.

Connects through the app's own async engine (``src.db.session.get_session``),
which reads ``DATABASE_URL`` from ``.env`` — no connection string or password is
ever printed or hardcoded here. Confirm the resolved DB is the self-evolving-agent
postgres container @ localhost:5433.

Usage::

    python scripts/analyze_vector_queries.py
    python scripts/analyze_vector_queries.py --out logs/vector_query_analysis.json
    python scripts/analyze_vector_queries.py --k 20          # larger LIMIT (more candidates)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Make ``src`` importable when run as ``python scripts/analyze_vector_queries.py``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger  # noqa: E402
from sqlalchemy import text  # noqa: E402

from src.db.session import get_session  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# (table, vector column, HNSW index name). All four are Vector(768) with
# vector_cosine_ops, so the cosine-distance operator ``<=>`` is the one the
# index accelerates.
TARGETS: list[tuple[str, str, str]] = [
    ("cold_memories", "embedding", "idx_cold_memories_embedding"),
    ("memory_embeddings", "embedding", "idx_memory_embeddings_vector"),
    ("tool_registrations", "capability_embedding", "idx_tool_registrations_capability_emb"),
    ("sub_agent_definitions", "capability_embedding", "idx_sub_agent_capability_emb"),
]

_EXEC_RE = re.compile(r"Execution Time: ([\d.]+) ms")
_PLAN_RE = re.compile(r"Planning Time: ([\d.]+) ms")


async def _analyze_one(session: Any, table: str, column: str, index: str, k: int) -> dict[str, Any]:
    """EXPLAIN ANALYZE the ANN query for one table and parse the plan."""
    # Read-only counts. Table/column come from the hardcoded TARGETS constant,
    # never from caller input, so f-string interpolation here carries no
    # injection surface (annotated for the linter).
    count_row = (
        await session.execute(
            text(f"SELECT count(*) AS n, count({column}) AS nn FROM {table}")  # noqa: S608
        )
    ).one()
    total_rows = int(count_row.n or 0)
    non_null = int(count_row.nn or 0)

    # EXPLAIN ANALYZE a cosine-ANN query. The query vector is sampled from the
    # table itself via a scalar subquery (a runtime constant), so the HNSW
    # index is usable AND nothing is interpolated into SQL — only LIMIT is a
    # bound parameter (the asyncpg dialect leaves casts on bound params
    # un-rewritten, so we keep the vector out of the param list entirely).
    explain_sql = (
        f"EXPLAIN (ANALYZE) "  # noqa: S608
        f"SELECT 1 FROM {table} "
        f"WHERE {column} IS NOT NULL "
        f"ORDER BY {column} <=> "
        f"(SELECT {column} FROM {table} WHERE {column} IS NOT NULL LIMIT 1) "
        f"LIMIT :k"
    )
    plan_lines = (
        await session.execute(text(explain_sql), {"k": k})
    ).scalars().all()
    plan = "\n".join(str(line) for line in plan_lines)

    exec_match = _EXEC_RE.search(plan)
    plan_match = _PLAN_RE.search(plan)
    return {
        "table": table,
        "column": column,
        "index": index,
        "total_rows": total_rows,
        "non_null_vectors": non_null,
        "query_vector": "sampled-from-table" if non_null else "none-empty",
        "limit": k,
        "uses_index": ("Index Scan" in plan or "Index Only Scan" in plan),
        "uses_seq_scan": "Seq Scan" in plan,
        "index_named_in_plan": index in plan,
        "execution_time_ms": float(exec_match.group(1)) if exec_match else None,
        "planning_time_ms": float(plan_match.group(1)) if plan_match else None,
        "plan": plan,
    }


async def _run(out_path: Path, k: int) -> dict[str, Any]:
    report: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "limit": k,
        "targets": [],
    }
    async with get_session() as session:
        for table, column, index in TARGETS:
            logger.info("EXPLAIN ANALYZE on {}.{} ...", table, column)
            try:
                report["targets"].append(await _analyze_one(session, table, column, index, k))
            except Exception as exc:  # noqa: BLE001 — non-fatal per table
                # An aborted statement poisons the implicit transaction; roll
                # back so subsequent tables can still be analyzed.
                await session.rollback()
                report["targets"].append(
                    {"table": table, "column": column, "index": index, "error": f"{type(exc).__name__}: {exc}"}
                )
                logger.error("Failed on {}: {}", table, exc)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    return report


def _print_summary(report: dict[str, Any]) -> None:
    print("\nVector query EXPLAIN ANALYZE summary (read-only):")
    print(f"{'table':<24} {'rows':>7} {'nn':>7} {'index?':>7} {'seq?':>5} {'exec_ms':>9}")
    print("-" * 64)
    for t in report.get("targets", []):
        if "error" in t:
            print(f"{t['table']:<24} {'-':>7} {'-':>7} {'ERR':>7} {'-':>5} {'-':>9}  {t['error']}")
            continue
        exec_ms = f"{t['execution_time_ms']:.2f}" if t["execution_time_ms"] is not None else "-"
        print(
            f"{t['table']:<24} {t['total_rows']:>7} {t['non_null_vectors']:>7} "
            f"{'yes' if t['uses_index'] else 'NO':>7} "
            f"{'yes' if t['uses_seq_scan'] else '-':>5} {exec_ms:>9}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--out", default=None, help="JSON report path (default logs/vector_query_analysis.json).")
    parser.add_argument("--k", type=int, default=10, help="ANN LIMIT / number of candidates (default 10).")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else REPO_ROOT / "logs" / "vector_query_analysis.json"
    report = asyncio.run(_run(out_path, args.k))
    _print_summary(report)
    logger.info("Report written to {}", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
