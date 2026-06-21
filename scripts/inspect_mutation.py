"""Inspect an evolution mutation row from the ``mutations`` table.

Reusable diagnostic for the evolution→live promotion path: the promotion gate
(``src/evolution/promote.py:parse_prompt_payload``) only promotes a PROMPT
mutation whose ``mutated_content`` is JSON ``{"target_node":..., "suffixes":[...]}``.
If the live engine emits a free-text file rewrite instead, the gate silently
no-ops. This script reads the stored row so we can see the actual payload shape
without guessing.

Usage::

    python scripts/inspect_mutation.py                  # latest mutation
    python scripts/inspect_mutation.py 4c5c11d4-...     # by id (full or prefix)
    python scripts/inspect_mutation.py --full            # print full mutated_content

No secrets are read or printed; connects via the app's SQLAlchemy engine.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from typing import Any

from sqlalchemy import Text, cast, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Mutation
from src.db.session import get_session


def _summarize(content: str, full: bool) -> str:
    if full:
        return content
    snippet = content[:800]
    suffix = " …[truncated]" if len(content) > 800 else ""
    return f"{snippet}{suffix}"


def _classify(content: str) -> str:
    """Heuristic shape label for mutated_content."""
    stripped = content.lstrip()
    if stripped.startswith("{"):
        return "json-object (may be {target_node,suffixes})"
    if stripped.startswith("["):
        return "json-array"
    if stripped.startswith("<") or "#!/usr/bin" in content[:80]:
        return "markup/script"
    return "free-text (NOT a {target_node,suffixes} payload → gate cannot promote)"


async def _fetch(session: AsyncSession, ident: str | None) -> Any:
    stmt = select(Mutation).order_by(Mutation.created_at.desc())
    if ident:
        # Allow a prefix match (the run logs a short prefix like 4c5c11d4).
        like = ident if "-" in ident else f"{ident}%"
        if "-" not in ident:
            stmt = (
                select(Mutation)
                .where(cast(Mutation.id, Text).like(like))
                .order_by(Mutation.created_at.desc())
            )
        else:
            stmt = select(Mutation).where(Mutation.id == uuid.UUID(ident))
    result = await session.execute(stmt)
    return result.scalars().first()


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("id", nargs="?", help="mutation id (full UUID or short prefix)")
    parser.add_argument("--full", action="store_true", help="print full mutated_content")
    args = parser.parse_args()

    async with get_session() as session:
        row = await _fetch(session, args.id)

    if row is None:
        print(f"No mutation found for id={args.id!r}", file=sys.stderr)
        return 1

    content = row.mutated_content or ""
    print(f"id            : {row.id}")
    print(f"mutation_type : {row.mutation_type}")
    print(f"target_path   : {row.target_path}")
    print(f"status        : {row.status}")
    print(f"model_used    : {row.model_used}")
    print(f"description   : {row.description}")
    print(f"content_len   : {len(content)} chars")
    print(f"shape         : {_classify(content)}")
    print("--- mutated_content ---")
    print(_summarize(content, args.full))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
