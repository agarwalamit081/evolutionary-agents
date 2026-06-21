"""Inspect or retire tools in the ``tool_registrations`` table.

Reads through the app's own async engine (``src.db.session.get_session``), which
pulls ``DATABASE_URL`` from ``.env`` — NO connection string or password is ever
printed, hardcoded, or read from ``os.environ`` here.

Default (no flags) lists every ACTIVE tool. ``--names`` filters to a comma list,
``--include-retired`` shows inactive rows too, and ``--retire --names a,b`` marks
those tools ``is_active=False`` via the governed ``ToolPersister.retire`` path
(reversible — it flips the active flag, never deletes).

Usage::

    python scripts/tool_query.py                       # all active tools
    python scripts/tool_query.py --names engineers,tasks   # specific tools only
    python scripts/tool_query.py --include-retired     # active + retired
    python scripts/tool_query.py --retire --names a,b  # retire named tools
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Make ``src`` importable when run directly as ``python scripts/tool_query.py``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

from src.db.session import get_session  # noqa: E402
from src.tools.dynamic.persister import ToolPersister  # noqa: E402


def _parse_names(raw: str) -> list[str]:
    """Split a comma list, trim, drop empties."""
    return [n.strip() for n in raw.split(",") if n.strip()]


def _build_where(names: list[str], include_retired: bool) -> tuple[str, dict[str, object]]:
    """Parameterized WHERE clause for the active filter + optional name list.

    Never interpolates values into SQL — names go through ``= ANY(:names)``.
    """
    clauses: list[str] = []
    params: dict[str, object] = {}
    if not include_retired:
        clauses.append("is_active = true")
    if names:
        clauses.append("tool_name = ANY(:names)")
        params["names"] = tuple(names)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


async def _list(names: list[str], include_retired: bool) -> int:
    where, params = _build_where(names, include_retired)
    sql = text(
        "SELECT tool_name, tool_type, is_active, calls, success_rate, "
        "empty_output_rate, last_run_at, created_at "
        "FROM tool_registrations " + where +
        " ORDER BY is_active DESC, calls DESC, tool_name"
    )
    async with get_session() as session:
        rows = (await session.execute(sql, params)).all()

    scope = " [all active]" if not names else f" [{', '.join(names)}]"
    scope += "" if include_retired else " (active only)"
    if not rows:
        print(f"tool_registrations{scope}: no rows matched.")
        return 0

    print(f"tool_registrations{scope}: {len(rows)} row(s)")
    print(
        f"  {'tool_name':24s} {'type':12s} {'active':6s} "
        f"{'calls':>6s} {'succ%':>5s} {'empty%':>6s} created"
    )
    for r in rows:
        succ = f"{r.success_rate * 100:.0f}" if r.success_rate is not None else "-"
        empty = f"{r.empty_output_rate * 100:.0f}" if r.empty_output_rate is not None else "-"
        created = r.created_at.strftime("%Y-%m-%d") if r.created_at else "-"
        print(
            f"  {r.tool_name:24s} {r.tool_type:12s} {str(r.is_active):6s} "
            f"{r.calls:6d} {succ:>5s} {empty:>6s} {created}"
        )
    return 0


async def _retire(names: list[str]) -> int:
    if not names:
        print("--retire requires --names <a,b,c> (will not bulk-retire).")
        return 2
    retired = await ToolPersister().retire(names)
    print(f"retired {retired}/{len(names)} tool(s): {', '.join(names)}")
    return 0


async def _main(args: argparse.Namespace) -> int:
    names = _parse_names(args.names)
    if args.retire:
        return await _retire(names)
    return await _list(names, args.include_retired)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Inspect or retire tools in the tool_registrations table."
    )
    ap.add_argument(
        "--names", default="",
        help="comma-separated tool_name list to filter on (or retire)",
    )
    ap.add_argument(
        "--include-retired", action="store_true",
        help="also list is_active=False rows (default: active only)",
    )
    ap.add_argument(
        "--retire", action="store_true",
        help="mark --names tools is_active=False (governed; reversible)",
    )
    args = ap.parse_args()
    raise SystemExit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()
