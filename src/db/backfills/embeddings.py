"""Idempotent embedding backfills for the capability + cold-memory tables.

Promoted from ``scripts/backfill_capability_embeddings.py`` +
``scripts/backfill_cold_embeddings.py`` (SI-1, scripts→src integration). These
are the **data** half of already-shipped schema migrations that added nullable
embedding columns:

- capability: ``c4e5a6b7c8d9_add_capability_embeddings`` added
  ``capability_embedding`` / ``capability_text`` to ``tool_registrations`` +
  ``sub_agent_definitions``.
- cold: the ``cold_memories.embedding`` column (Phase-7 embedding fix).

Rows created before those fixes have NULL embeddings, so they are invisible to
semantic recall / dedup (both filter on ``<col> IS NOT NULL``). These backfills
embed each NULL row's text and write the vector back in place. Invoked via
``main.py --backfill-embeddings``; both are idempotent (re-runs are no-ops).

The two backfills differ in one important way, preserved here exactly:

- **capability** stores ONLY ``source == "api"`` vectors (hash-fallback vectors
  are deterministic but not semantically meaningful — storing them would pollute
  the cosine dedup index), and synthesizes the embed text from
  ``capability_text`` falling back to ``"{name}: {description}"``.
- **cold** stores EVERY non-empty vector (including the hash fallback) from the
  row's ``content`` — a cold memory is better recalled on a deterministic vector
  than not recalled at all.

Properties (both):
- **Idempotent** — the fetch returns only ``WHERE <col> IS NULL`` rows, so a
  re-run after success selects nothing. Failed / skipped rows stay NULL and are
  retried on the next run.
- **Bounded concurrency** — an ``asyncio.Semaphore`` caps concurrent embedding
  API calls (the expensive part); writes apply within a single transaction.
- **Robust** — a bad row never aborts the whole run; it is counted ``failed``
  and left NULL.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import sqlalchemy as sa
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import ColdMemory as ColdMemoryModel
from src.db.models import SubAgentModel, ToolRegistration
from src.memory.embeddings import EmbeddingGenerator


@dataclass
class BackfillStats:
    """Outcome counters for one backfill run (capability and/or cold)."""

    scanned: int = 0  # NULL-embedding rows found
    stored: int = 0  # rows that received a persisted vector this run
    skipped_hash: int = 0  # capability-only: hash-fallback vector, not stored
    skipped_no_text: int = 0  # capability-only: no embeddable text, not stored
    failed: int = 0  # embedding raised / returned empty; left NULL, retried


# ─── shared pure helpers (capability path) ──────────────────────────


def select_capability_text(
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


def should_store_capability(source: str | None) -> bool:
    """True only when ``source == "api"``.

    Hash-fallback vectors are not semantically meaningful, so storing them would
    pollute the cosine dedup index without enabling real dedup. Any other source
    (``"hash"``, ``None``, unknown) is skipped.
    """
    return source == "api"


# ─── capability backfill (ToolRegistration + SubAgentModel) ─────────


async def _fetch_null_capability_rows(
    session: AsyncSession, model_class: Any
) -> list[Any]:
    """Return rows of ``model_class`` whose ``capability_embedding`` is NULL."""
    stmt = sa.select(model_class).where(model_class.capability_embedding.is_(None))
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
    same binding path the persisters use.
    """
    await session.execute(
        sa.update(model_class)
        .where(model_class.id == row_id)
        .values(capability_embedding=vector, capability_text=capability_text)
    )


async def backfill_capability_table(
    session: AsyncSession,
    model_class: Any,
    generator: EmbeddingGenerator,
    *,
    dry_run: bool = False,
    concurrency: int = 5,
) -> BackfillStats:
    """Embed every NULL-capability row of ``model_class`` and persist vectors.

    Args:
        session: async DB session. Writes commit once at the end; rolled back on
            error by the caller's session context. With a dry run, no
            writes/commit happen.
        model_class: ``ToolRegistration`` or ``SubAgentModel`` (generic over
            both — they share the ``id`` / ``capability_embedding`` /
            ``capability_text`` / name+description shape).
        generator: embedding generator. ``generator.last_source`` is read after
            each ``generate()`` to decide store-vs-skip (api-only).
        dry_run: when True, embed (to exercise the path) but do not persist.
        concurrency: max concurrent embedding API calls.

    Returns:
        ``BackfillStats`` — skipped/failed rows are left NULL and retried next run.
    """
    table_name = getattr(model_class, "__tablename__", str(model_class))
    rows = await _fetch_null_capability_rows(session, model_class)
    if not rows:
        logger.info(f"[{table_name}] no rows need backfill (all have vectors).")
        return BackfillStats()

    logger.info(
        f"[{table_name}] backfilling {len(rows)} NULL-capability rows "
        f"(concurrency={concurrency}, dry_run={dry_run})."
    )

    sem = asyncio.Semaphore(max(1, concurrency))

    async def _embed(
        row: Any,
    ) -> tuple[object, str | None, list[float] | None, str]:
        name = getattr(row, "tool_name", None) or getattr(row, "name", None)
        description = getattr(row, "description", None)
        text = select_capability_text(row.capability_text, name, description)
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
        if not should_store_capability(source):
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

    stats = BackfillStats(scanned=len(rows))
    for _row_id, text, vector, disposition in outcomes:
        if disposition == "no_text":
            stats.skipped_no_text += 1
        elif disposition == "hash":
            stats.skipped_hash += 1
        elif disposition == "error":
            stats.failed += 1
        elif disposition == "store" and vector is not None and text is not None:
            if dry_run:
                stats.stored += 1
                continue
            await _apply_capability(session, model_class, _row_id, vector, text)
            logger.info(f"[{table_name}] stored capability vector for row {_row_id}.")
            stats.stored += 1

    if not dry_run and stats.stored:
        await session.commit()

    logger.info(
        f"[{table_name}] done: stored={stats.stored} skipped_hash="
        f"{stats.skipped_hash} skipped_no_text={stats.skipped_no_text} "
        f"failed={stats.failed}."
    )
    return stats


# ─── cold-memory backfill (ColdMemory) ──────────────────────────────


async def _fetch_null_cold_rows(session: AsyncSession) -> list[ColdMemoryModel]:
    """Return ``cold_memories`` rows whose ``embedding`` is NULL (the backfill set)."""
    stmt = sa.select(ColdMemoryModel).where(ColdMemoryModel.embedding.is_(None))
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def _apply_cold_embedding(
    session: AsyncSession, memory_id: object, vector: list[float]
) -> None:
    """Write the embedding vector back to the cold-memory row (single UPDATE)."""
    await session.execute(
        sa.update(ColdMemoryModel)
        .where(ColdMemoryModel.id == memory_id)
        .values(embedding=vector)
    )


async def backfill_cold_memories(
    session: AsyncSession,
    generator: EmbeddingGenerator,
    *,
    dry_run: bool = False,
    concurrency: int = 5,
) -> BackfillStats:
    """Embed every NULL-embedding cold memory and persist the vectors.

    Unlike the capability path, EVERY non-empty vector is stored (including the
    hash fallback) — a cold memory is better recalled on a deterministic vector
    than not recalled at all.

    Args:
        session: async DB session. Writes commit once at the end. With a dry
            run, no writes/commit happen.
        generator: embedding generator (litellm real vectors, or hash fallback).
        dry_run: when True, embed (to exercise the path) but do not persist.
        concurrency: max concurrent embedding API calls.

    Returns:
        ``BackfillStats`` counting scanned / stored / failed rows.
    """
    rows = await _fetch_null_cold_rows(session)
    if not rows:
        logger.info("No cold_memories rows need backfill (all have embeddings).")
        return BackfillStats()

    logger.info(
        f"Backfilling {len(rows)} NULL-embedding cold memories "
        f"(concurrency={concurrency}, dry_run={dry_run})."
    )

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

    stats = BackfillStats(scanned=len(rows))
    for memory_id, vector in outcomes:
        if vector is None:
            stats.failed += 1
            continue
        if dry_run:
            stats.stored += 1
            continue
        await _apply_cold_embedding(session, memory_id, vector)
        stats.stored += 1

    if not dry_run and stats.stored:
        await session.commit()

    logger.info(
        f"Backfill complete: scanned={stats.scanned} stored={stats.stored} "
        f"failed={stats.failed}."
    )
    return stats


# ─── top-level entry (DI seam — session + generator injected) ───────


_CAPABILITY_TABLES: tuple[Any, ...] = (ToolRegistration, SubAgentModel)


async def run_backfill(
    *,
    table: str = "all",
    concurrency: int = 5,
    dry_run: bool = False,
    session: AsyncSession,
    generator: EmbeddingGenerator,
) -> BackfillStats:
    """Run the requested backfill(s), combining stats across tables.

    Thin orchestrator over :func:`backfill_capability_table` /
    :func:`backfill_cold_memories`. ``session`` and ``generator`` are injected
    (no internal I/O) so the dispatch is unit-testable; the ``main.py`` handler
    wires the real settings/session/generator.

    Args:
        table: ``"capability"`` (tools + sub-agents), ``"cold"``, or ``"all"``.
        concurrency: max concurrent embedding API calls (passed through).
        dry_run: embed but do not persist / commit.
        session: an open async DB session (caller owns commit/rollback lifecycle
            via its context manager; the per-table funcs commit internally when
            they store rows).
        generator: the :class:`EmbeddingGenerator` to use.

    Returns:
        Combined :class:`BackfillStats` across every table touched.
    """
    if table not in ("capability", "cold", "all"):
        raise ValueError(f"unknown backfill table {table!r}; use capability|cold|all")

    combined = BackfillStats()
    if table in ("capability", "all"):
        for model_class in _CAPABILITY_TABLES:
            partial = await backfill_capability_table(
                session, model_class, generator,
                dry_run=dry_run, concurrency=concurrency,
            )
            combined.scanned += partial.scanned
            combined.stored += partial.stored
            combined.skipped_hash += partial.skipped_hash
            combined.skipped_no_text += partial.skipped_no_text
            combined.failed += partial.failed
    if table in ("cold", "all"):
        partial = await backfill_cold_memories(
            session, generator, dry_run=dry_run, concurrency=concurrency,
        )
        combined.scanned += partial.scanned
        combined.stored += partial.stored
        combined.failed += partial.failed
    return combined
