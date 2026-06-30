"""Data backfills — the data half of already-shipped schema migrations.

These populate nullable columns added by DDL migrations. API-calling data
backfills never belong in a DDL migration (a migration must not depend on a
provider key), so they live here and are invoked via ``main.py`` flags, never
run automatically. See :mod:`src.db.backfills.embeddings` for the capability +
cold-memory embedding backfills (``main.py --backfill-embeddings``).
"""

from src.db.backfills.embeddings import (
    BackfillStats,
    backfill_capability_table,
    backfill_cold_memories,
    run_backfill,
    select_capability_text,
    should_store_capability,
)

__all__ = [
    "BackfillStats",
    "backfill_capability_table",
    "backfill_cold_memories",
    "run_backfill",
    "select_capability_text",
    "should_store_capability",
]
