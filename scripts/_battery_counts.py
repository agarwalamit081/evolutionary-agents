"""Battery DB baseline/delta probe. THROWAWAY — used only while running the
10-query validation battery, then deleted.

Prints row counts for every table the battery rubric measures, plus
cold_memories.embedding non-null coverage and cost_ledger spend totals.
Connects through the app's own async engine (reads DATABASE_URL from .env) so
NO connection string or password is ever printed or hardcoded.

Usage::

    python scripts/_battery_counts.py            # human-readable
    python scripts/_battery_counts.py --label Q9 # tag a line
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Make ``src`` importable when run directly as ``python scripts/_battery_counts.py``
# (Python puts the script's own dir on the path, not the repo root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

from src.db.session import get_session  # noqa: E402


_QUERIES: dict[str, str] = {
    "warm_memories": "SELECT count(*) FROM warm_memories",
    "cold_memories": "SELECT count(*) FROM cold_memories",
    "cold_embeddings_nonnull": (
        "SELECT count(*) FROM cold_memories WHERE embedding IS NOT NULL"
    ),
    "memory_embeddings": "SELECT count(*) FROM memory_embeddings",
    "cost_ledger_rows": "SELECT count(*) FROM cost_ledger",
    "cost_total_tokens": "SELECT coalesce(sum(total_tokens),0) FROM cost_ledger",
    "cost_total_usd": "SELECT coalesce(sum(cost_usd),0) FROM cost_ledger",
    "tool_registrations": "SELECT count(*) FROM tool_registrations",
    "tool_versions": "SELECT count(*) FROM tool_versions",
    "sub_agent_definitions": "SELECT count(*) FROM sub_agent_definitions",
    "sub_agent_runs": "SELECT count(*) FROM sub_agent_runs",
    "mutations": "SELECT count(*) FROM mutations",
    "mutation_chains": "SELECT count(*) FROM mutation_chains",
    "evolution_telemetry": "SELECT count(*) FROM evolution_telemetry",
}


async def _probe(label: str) -> int:
    async with get_session() as session:
        parts: list[str] = []
        for name, sql in _QUERIES.items():
            val = (await session.execute(text(sql))).scalar_one()
            if isinstance(val, float):
                val = round(float(val), 4)
            parts.append(f"{name}={val}")
        line = f"[{label}] " if label else ""
        print(line + " | ".join(parts))
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(_probe(args.label)))


if __name__ == "__main__":
    main()
