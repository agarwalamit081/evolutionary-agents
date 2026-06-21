"""Query the LLM cost ledger and print a spend breakdown.

Reports per-run / per-model aggregates from the ``cost_ledger`` table populated
by ``CostTracker``. Connects through the app's own async engine
(``src.db.session.get_session``), which reads ``DATABASE_URL`` from ``.env`` —
NO connection string or password is ever printed, hardcoded, or read from
``os.environ`` here.

Usage::

    python scripts/cost_query.py                       # all-time, per-run + per-model
    python scripts/cost_query.py --run-id cli-q07       # one run's breakdown
    python scripts/cost_query.py --today                # today only
    python scripts/cost_query.py --since 2026-06-20     # since a date (inclusive)
    python scripts/cost_query.py --by-model             # collapse across runs
    python scripts/cost_query.py --run-id cli-q07 --today   # filters combine (AND)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Make ``src`` importable when run directly as ``python scripts/cost_query.py``
# (Python puts the script's own dir on the path, not the repo root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

from src.db.session import get_session  # noqa: E402


def _build_where(args: argparse.Namespace) -> tuple[str, dict[str, object]]:
    """Build a parameterized WHERE clause (AND of the active filters).

    Never interpolates values into SQL — every filter is a bound parameter, so
    there is no injection surface.
    """
    clauses: list[str] = []
    params: dict[str, object] = {}
    if args.run_id:
        clauses.append("run_id = :run_id")
        params["run_id"] = args.run_id
    if args.model:
        clauses.append("model = :model")
        params["model"] = args.model
    if args.today:
        clauses.append("created_at::date = CURRENT_DATE")
    if args.since:
        clauses.append("created_at >= :since::date")
        params["since"] = args.since
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


async def _report(args: argparse.Namespace) -> int:
    where, params = _build_where(args)

    # Grand total for the filtered set.
    total_sql = text(
        "SELECT count(*) AS calls, coalesce(sum(cost_usd), 0) AS spend, "
        "coalesce(sum(total_tokens), 0) AS tok FROM cost_ledger " + where
    )

    if args.by_model:
        detail_sql = text(
            "SELECT model, count(*) AS calls, coalesce(sum(cost_usd), 0) AS spend, "
            "coalesce(sum(total_tokens), 0) AS tok "
            "FROM cost_ledger " + where +
            " GROUP BY model ORDER BY spend DESC"
        )
    else:
        detail_sql = text(
            "SELECT coalesce(nullif(run_id, ''), '(none)') AS run_id, model, "
            "count(*) AS calls, coalesce(sum(cost_usd), 0) AS spend, "
            "coalesce(sum(total_tokens), 0) AS tok "
            "FROM cost_ledger " + where +
            " GROUP BY run_id, model ORDER BY run_id, spend DESC"
        )

    async with get_session() as session:
        tot = (await session.execute(total_sql, params)).one()
        detail = (await session.execute(detail_sql, params)).all()

    filt: list[str] = []
    if args.run_id:
        filt.append(f"run_id={args.run_id}")
    if args.model:
        filt.append(f"model={args.model}")
    if args.today:
        filt.append("today")
    if args.since:
        filt.append(f"since={args.since}")
    scope = (" [" + ", ".join(filt) + "]") if filt else " [all-time]"

    if tot.calls == 0:
        print(f"cost_ledger{scope}: no rows matched.")
        return 0

    print(f"cost_ledger{scope}")
    print(
        f"  TOTAL: ${float(tot.spend):.4f} | {tot.calls} calls | "
        f"{int(tot.tok):,} tokens\n"
    )

    if args.by_model:
        print("  by model:")
        for row in detail:
            print(
                f"    {row.model:28s} ${float(row.spend):8.4f} | "
                f"{row.calls:5d} calls | {int(row.tok):>12,} tokens"
            )
        return 0

    # Per-run subtotals.
    subtotals: dict[str, tuple[int, float, int]] = {}
    for row in detail:
        c, s, t = subtotals.get(row.run_id, (0, 0.0, 0))
        subtotals[row.run_id] = (c + row.calls, s + float(row.spend), t + int(row.tok))

    print("  by run:")
    for g in sorted(subtotals, key=lambda k: subtotals[k][1], reverse=True):
        c, s, t = subtotals[g]
        print(f"    {g:24s} ${s:8.4f} | {c:5d} calls | {t:>12,} tokens")

    print("\n  by run × model:")
    for row in detail:
        print(
            f"    {row.run_id:18s} {row.model:22s} ${float(row.spend):8.4f} | "
            f"{row.calls:4d} calls | {int(row.tok):>12,} tokens"
        )
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Query the cost_ledger spend breakdown."
    )
    ap.add_argument("--run-id", default="", help="filter to a single run_id")
    ap.add_argument("--model", default="", help="filter to a single model id")
    ap.add_argument("--today", action="store_true", help="only rows from today")
    ap.add_argument(
        "--since", default="", help="only rows on/after DATE (YYYY-MM-DD)"
    )
    ap.add_argument(
        "--by-model", action="store_true", help="collapse breakdown across runs"
    )
    args = ap.parse_args()
    if args.today and args.since:
        ap.error("--today and --since are mutually exclusive")
    raise SystemExit(asyncio.run(_report(args)))


if __name__ == "__main__":
    main()
