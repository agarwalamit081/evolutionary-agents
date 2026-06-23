"""add_warm_memory_fact_key

Revision ID: h1a2b3c4d5e6
Revises: g9d0e1f2a3b4
Create Date: 2026-06-23 12:00:00.000000

Fact dedup (findings A5). ``fact_key`` was metadata-only before: ``store_fact``
stuffed it into ``extra_data`` and used ``title`` as the lookup key, so re-running
a goal duplicated the same fact row each time (no constraint to stop it). This
promotes ``fact_key`` to a real nullable column + a partial unique index so
``WarmMemoryStore.store_fact`` can ``ON CONFLICT (fact_key) DO UPDATE`` — re-
extraction updates the existing row instead of appending a clone (the race-free
form; concurrent memory folds can both extract the same fact).

The index is partial (``memory_type='fact' AND expires_at IS NULL``): only facts
carry ``fact_key`` (NULL on every other type, and NULLs are distinct so they never
collide), and a retired (expires_at-set) fact must not shadow a freshly-extracted
one (``retrieve_facts`` already filters ``expires_at IS NULL``).

Backfill then **dedup** run before the index build: with no dedup before, the live
DB had accumulated many duplicate active facts per key (e.g. ``integrity_report_schema``
had 11 copies from 11 re-runs). For each key the newest row (``updated_at`` desc,
then ``created_at`` desc) survives and the older copies are **retired**
(``expires_at = now()``) rather than deleted — retired rows are preserved for audit
but invisible to recall (``retrieve_facts`` filters ``expires_at IS NULL``), so the
partial index sees exactly one active row per key with no data loss.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "h1a2b3c4d5e6"
down_revision: Union[str, None] = "g9d0e1f2a3b4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1) Nullable fact_key column. NULL for every non-fact row, so the partial
    #    unique index below never collides with skills/procedures.
    op.add_column("warm_memories", sa.Column("fact_key", sa.Text(), nullable=True))

    # 2) Backfill existing facts from their extra_data payload (the column was
    #    metadata-only before this migration). MUST precede the dedup + index
    #    creation: two legacy facts sharing a key would otherwise fail the build.
    op.execute(
        "UPDATE warm_memories "
        "SET fact_key = extra_data->>'fact_key' "
        "WHERE memory_type = 'fact' AND extra_data ? 'fact_key'"
    )

    # 3) Reconcile duplicate facts: with no dedup before, the live DB had
    #    accumulated many duplicate active rows per key (re-runs re-extracted the
    #    same fact each time). For each key keep the newest row (updated_at desc,
    #    then created_at desc) and retire the older copies (expires_at = now()).
    #    Retire — not delete — so retired rows are preserved but invisible to
    #    recall (retrieve_facts filters expires_at IS NULL), leaving exactly one
    #    active row per key for the unique index to cover, with no data loss.
    op.execute(
        "UPDATE warm_memories wm "
        "SET expires_at = now() "
        "WHERE memory_type = 'fact' "
        "  AND expires_at IS NULL "
        "  AND fact_key IS NOT NULL "
        "  AND id <> ( "
        "      SELECT wm2.id FROM warm_memories wm2 "
        "      WHERE wm2.memory_type = 'fact' "
        "        AND wm2.expires_at IS NULL "
        "        AND wm2.fact_key = wm.fact_key "
        "      ORDER BY wm2.updated_at DESC, wm2.created_at DESC "
        "      LIMIT 1 "
        "  )"
    )

    # 4) Partial unique index backing the ON CONFLICT (fact_key) upsert. Scoped
    #    to active facts so a retired fact never shadows re-extraction.
    op.create_index(
        "uq_warm_memories_fact_key",
        "warm_memories",
        ["fact_key"],
        unique=True,
        postgresql_where=sa.text("memory_type = 'fact' AND expires_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_warm_memories_fact_key", table_name="warm_memories")
    op.drop_column("warm_memories", "fact_key")
