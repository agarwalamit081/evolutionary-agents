"""Backfill NULL ``capability_embedding`` vectors on tools + sub-agents (F1 gap).

Every existing tool (``tool_registrations``) and sub-agent
(``sub_agent_definitions``) was created **before** the B3 capability-embedding
fix, so all active rows have a NULL ``capability_embedding`` — invisible to
``find_similar``, which filters on ``capability_embedding IS NOT NULL``. This
makes semantic dedup inert for the existing population (no reuse, no
consolidation). This one-shot script embeds each such row's capability text and
writes the vector back in place, populating the dedup index.

Only ``source == "api"`` (real provider) vectors are stored. Hash-fallback
vectors are deterministic but not semantically meaningful, so storing them would
pollute the cosine index without enabling real dedup — those rows are skipped
(left NULL) and retried on the next run. See
``EmbeddingGenerator.last_source`` / ``embed_capability``.

Properties:
- **Idempotent** — the fetch only returns rows ``WHERE capability_embedding IS
  NULL``, so a re-run after a successful backfill selects nothing and is a
  no-op. Skipped (hash / no-text) rows stay NULL and are retried next run.
- **Bounded concurrency** — an ``asyncio.Semaphore`` caps concurrent embedding
  API calls (the expensive part); writes apply within a single transaction.
- **Two tables** — tools first, then sub-agents, each with its own summary.
- **Logged** — every store/skip carries the row id + reason.

Run once against the canonical (docker) DB **after** the stack migration::

    source /home/amiagarw/aiml01/bin/activate
    python scripts/backfill_capability_embeddings.py
    python scripts/backfill_capability_embeddings.py --concurrency 8 --dry-run

``--dry-run`` reports what would be backfilled without writing. Does NOT touch
``cold_memories.embedding`` (already fully populated — a separate concern).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

import sqlalchemy as sa
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings import get_settings
from src.db.models import SubAgentModel, ToolRegistration
from src.db.session import close_db, get_session
from src.memory.embeddings import EmbeddingGenerator


def select_embedding_text(
    capability_text: str | None,
    name: str | None,
    description: str | None,
) -> str | None:
    """Return the text to embed for one capability row.

    Preference order:
    1. ``capability_text`` (the canonical embedded text, kept for re-embedding).
    2. ``"{name}: {description}"`` synthesized from the row's identity fields.
    3. ``None`` when nothing usable is available → skip the row (leave NULL).

    Whitespace-only values are treated as empty so a blank ``capability_text``
    falls through to the name/description synthesis rather than embedding "".
    """
    if capability_text and capability_text.strip():
        return capability_text
    if name and name.strip() and description and description.strip():
        return f"{name}: {description}"
    return None


def should_store(source: str | None) -> bool:
    """True only when ``source == "api"``.

    Hash-fallback vectors are not semantically meaningful, so storing them would
    pollute the cosine index without enabling real dedup. Any other source
    (``"hash"``, ``None``, unknown) is skipped.
    """
    return source == "api"


async def _fetch_null_rows(session: AsyncSession, model_class: Any) -> list[Any]:
    """Return rows of ``model_class`` whose capability_embedding is NULL."""
    stmt = sa.select(model_class).where(
        model_class.capability_embedding.is_(None)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def _apply_capability(
    session: AsyncSession,
    model_class: Any,
    row_id: object,
    vector: list[float],
    capability_text: str,
) -> None:
    """Write the capability vector (+ embedded text) back to the row (one UPDATE).

    ``capability_text`` is set so a later re-embedding (model change, etc.) can
    reproduce the vector. A plain ``list[float]`` binds directly to the pgvector
    ``Vector(768)`` column — the asyncpg/pgvector adapter handles the cast, the
    same binding path the persisters use (see ``_spec_to_model``).
    """
    await session.execute(
        sa.update(model_class)
        .where(model_class.id == row_id)
        .values(
            capability_embedding=vector,
            capability_text=capability_text,
        )
    )


async def backfill_table(
    session: AsyncSession,
    model_class: Any,
    generator: EmbeddingGenerator,
    *,
    dry_run: bool = False,
    concurrency: int = 5,
) -> tuple[int, int, int, int]:
    """Embed every NULL-capability row of ``model_class`` and persist vectors.

    Args:
        session: async DB session. Writes commit once at the end; rolled back on
            error by ``get_session``. With a dry run, no writes/commit happen.
        model_class: ``ToolRegistration`` or ``SubAgentModel`` (generic over
            both — they share the ``id`` / ``capability_embedding`` /
            ``capability_text`` / name+description shape).
        generator: embedding generator. ``generator.last_source`` is read after
            each ``generate()`` to decide store-vs-skip (api-only).
        dry_run: when True, embed (to exercise the path) but do not persist.
        concurrency: max concurrent embedding API calls.

    Returns:
        ``(scanned, stored, skipped_hash, skipped_no_text)`` counts. Skipped
        rows are left NULL and retried on the next run.
    """
    table_name = getattr(model_class, "__tablename__", str(model_class))
    rows = await _fetch_null_rows(session, model_class)
    if not rows:
        logger.info(f"[{table_name}] no rows need backfill (all have vectors).")
        return 0, 0, 0, 0

    logger.info(
        f"[{table_name}] backfilling {len(rows)} NULL-capability rows "
        f"(concurrency={concurrency}, dry_run={dry_run})."
    )

    sem = asyncio.Semaphore(max(1, concurrency))

    async def _embed(row: Any) -> tuple[object, str | None, list[float] | None, str | None]:
        name = getattr(row, "tool_name", None) or getattr(row, "name", None)
        description = getattr(row, "description", None)
        text = select_embedding_text(row.capability_text, name, description)
        if text is None:
            logger.info(f"[{table_name}] skip row {row.id}: no embedding text.")
            return row.id, None, None, "no_text"
        async with sem:
            try:
                vec = await generator.generate(text)
            except Exception as exc:  # a bad row must not abort the whole run
                logger.warning(f"[{table_name}] embed failed for row {row.id}: {exc}")
                return row.id, text, None, "error"
        source = generator.last_source
        if not should_store(source):
            logger.info(
                f"[{table_name}] skip row {row.id}: source={source!r} "
                f"(not a semantically-meaningful vector)."
            )
            return row.id, text, None, "hash"
        if not vec:
            logger.warning(f"[{table_name}] empty embedding for row {row.id}.")
            return row.id, text, None, "error"
        return row.id, text, vec, "store"

    outcomes = await asyncio.gather(*(_embed(r) for r in rows))

    stored = 0
    skipped_hash = 0
    skipped_no_text = 0
    for _row_id, text, vector, disposition in outcomes:
        if disposition == "no_text":
            skipped_no_text += 1
        elif disposition == "hash":
            skipped_hash += 1
        elif disposition == "store" and vector is not None and text is not None:
            if dry_run:
                stored += 1
                continue
            await _apply_capability(session, model_class, _row_id, vector, text)
            logger.info(f"[{table_name}] stored capability vector for row {_row_id}.")
            stored += 1
        # disposition == "error" → counted as neither stored nor skipped; left
        # NULL and retried on the next run (not a hash/no-text skip).

    if not dry_run and stored:
        await session.commit()

    logger.info(
        f"[{table_name}] done: stored={stored} skipped_hash={skipped_hash} "
        f"skipped_no_text={skipped_no_text}."
    )
    return len(rows), stored, skipped_hash, skipped_no_text


async def _run(concurrency: int, dry_run: bool) -> int:
    """Wire settings → session/generator → backfill both tables. Returns exit code."""
    settings = get_settings()
    generator = EmbeddingGenerator(settings)
    logger.info(f"Embedding model={generator.model} dim={generator.dimension}.")
    scanned = 0
    stored = 0
    async with get_session() as session:
        for model_class in (ToolRegistration, SubAgentModel):
            s, st, _h, _t = await backfill_table(
                session, model_class, generator,
                dry_run=dry_run, concurrency=concurrency,
            )
            scanned += s
            stored += st
    logger.info(f"Backfill summary: scanned={scanned} stored={stored} (dry_run={dry_run}).")
    # Non-zero exit only if rows were eligible but none stored (e.g. no API key
    # and nothing skipped). A run that stores or skips anything exits 0 so
    # automation can re-run idempotently to retry the remaining NULLs.
    return 1 if scanned and stored == 0 else 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill NULL capability_embedding vectors (tools + sub-agents)."
    )
    parser.add_argument(
        "--concurrency", type=int, default=5,
        help="Max concurrent embedding API calls (default: 5).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would backfill without writing.",
    )
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return await _run(concurrency=args.concurrency, dry_run=args.dry_run)
    finally:
        await close_db()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
