"""Backfill NULL ``cold_memories`` embeddings (§10.2 gap).

Every cold memory stored **before** the Phase-7 embedding fix has a NULL
``embedding``, so it is invisible to semantic recall —
``cold.search_by_embedding`` / ``cold.search_by_query`` filter on
``embedding IS NOT NULL``. This one-shot script embeds each such row's
``content`` and writes the vector back in place.

Properties:
- **Idempotent** — the fetch only returns rows ``WHERE embedding IS NULL``, so a
  re-run after a successful backfill selects nothing and is a no-op. Rows whose
  embedding generation failed are left NULL and retried on the next run.
- **Bounded concurrency** — an ``asyncio.Semaphore`` caps concurrent embedding
  API calls (the expensive part); the writes apply within a single transaction.
- **Logged** — progress and a final stats summary.

Run once against the canonical (docker) DB **after** the stack migration::

    source /home/amiagarw/aiml01/bin/activate
    python scripts/backfill_cold_embeddings.py
    python scripts/backfill_cold_embeddings.py --concurrency 8 --dry-run

``--dry-run`` reports how many rows would be backfilled without writing.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass

import sqlalchemy as sa
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings import get_settings
from src.db.models import ColdMemory as ColdMemoryModel
from src.db.session import close_db, get_session
from src.memory.embeddings import EmbeddingGenerator


@dataclass
class BackfillStats:
    """Outcome counters for one backfill run."""

    scanned: int = 0  # NULL-embedding rows found
    embedded: int = 0  # rows that received a vector this run
    failed: int = 0  # rows whose embedding generation failed (left NULL)
    skipped: int = 0  # non-NULL rows encountered (always 0 via the NULL filter)


async def _fetch_null_rows(session: AsyncSession) -> list[ColdMemoryModel]:
    """Return cold_memories rows whose embedding is NULL (the backfill set)."""
    stmt = sa.select(ColdMemoryModel).where(ColdMemoryModel.embedding.is_(None))
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def _apply_embedding(
    session: AsyncSession, memory_id: object, vector: list[float]
) -> None:
    """Write the embedding vector back to the row (single UPDATE)."""
    await session.execute(
        sa.update(ColdMemoryModel)
        .where(ColdMemoryModel.id == memory_id)
        .values(embedding=vector)
    )


async def backfill_missing_embeddings(
    session: AsyncSession,
    generator: EmbeddingGenerator,
    *,
    concurrency: int = 5,
    dry_run: bool = False,
) -> BackfillStats:
    """Embed every NULL-embedding cold memory and persist the vectors.

    Args:
        session: async DB session. Writes commit once at the end; rolled back on
            error by ``get_session``. With a dry run, no writes/commit happen.
        generator: embedding generator (litellm real vectors, or hash fallback).
        concurrency: max concurrent embedding API calls.
        dry_run: when True, embed (to exercise the path) but do not persist.

    Returns:
        ``BackfillStats`` counting scanned / embedded / failed rows.
    """
    rows = await _fetch_null_rows(session)
    if not rows:
        logger.info("No cold_memories rows need backfill (all have embeddings).")
        return BackfillStats()

    stats = BackfillStats(scanned=len(rows))
    logger.info(f"Backfilling {stats.scanned} NULL-embedding cold memories "
                f"(concurrency={concurrency}, dry_run={dry_run}).")

    sem = asyncio.Semaphore(max(1, concurrency))

    async def _embed(row: ColdMemoryModel) -> tuple[object, list[float] | None]:
        async with sem:
            try:
                vec = await generator.generate(row.content)
            except Exception as exc:  # a bad row must not abort the whole run
                logger.warning(f"Embedding failed for cold memory {row.id}: {exc}")
                return row.id, None
            if not vec:
                logger.warning(f"Empty embedding returned for cold memory {row.id}.")
                return row.id, None
            return row.id, vec

    outcomes = await asyncio.gather(*(_embed(r) for r in rows))

    for memory_id, vector in outcomes:
        if vector is None:
            stats.failed += 1
            continue
        if dry_run:
            stats.embedded += 1
            continue
        await _apply_embedding(session, memory_id, vector)
        stats.embedded += 1

    if not dry_run:
        await session.commit()

    logger.info(
        f"Backfill complete: scanned={stats.scanned} embedded={stats.embedded} "
        f"failed={stats.failed} skipped={stats.skipped}."
    )
    return stats


async def _run(concurrency: int, dry_run: bool) -> int:
    """Wire settings → session/generator → backfill. Returns a process exit code."""
    settings = get_settings()
    generator = EmbeddingGenerator(settings)
    logger.info(f"Embedding model={generator.model} dim={generator.dimension}.")
    async with get_session() as session:
        stats = await backfill_missing_embeddings(
            session, generator, concurrency=concurrency, dry_run=dry_run
        )
    # Non-zero exit only if every row failed — partial success still exits 0 so a
    # re-run can retry the failures idempotently without alarming automation.
    return 1 if stats.scanned and stats.failed == stats.scanned else 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill NULL cold_memories embeddings.")
    parser.add_argument(
        "--concurrency", type=int, default=5,
        help="Max concurrent embedding API calls (default: 5).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report how many rows would backfill without writing.",
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
