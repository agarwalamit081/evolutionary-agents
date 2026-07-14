"""add_is_active_to_agent_config_version

Revision ID: n8f9a0b1c2d3
Revises: m7e8f9a0b1c2
Create Date: 2026-07-14 00:00:00.000000

Adds the atomic active-pointer column ``is_active`` to ``agent_config_versions``
(Phase 3b — atomic config versioning). At most ONE version is active at a time;
``MutationPersister.set_active_version`` performs the swap in a single
transaction (clear every other row, set the target). A CONFIG mutation's
rollback re-points ``is_active`` to the prior version instead of deleting data.

The column is added ``NOT NULL`` with ``server_default='false'`` so any existing
rows back-fill cleanly (the table has no application writers today — verified
0 readers/writers outside ``models.py`` — so this is a pure capability add).

A partial unique index ``idx_agent_config_versions_one_active`` gives PostgreSQL
a second invariant check (only one ``is_active = true`` row can exist) and makes
the "find the active version" lookup cheap.

Fully reversible: ``downgrade`` drops the partial index then the column, so the
schema round-trips to its prior shape.
"""
from __future__ import annotations

from typing import Union  # noqa: F401  — Alembic template convention

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "n8f9a0b1c2d3"
down_revision: Union[str, None] = "m7e8f9a0b1c2"
branch_labels: object = None  # type: ignore[assignment]
depends_on: object = None  # type: ignore[assignment]


def upgrade() -> None:
    # NOT NULL with a server_default so existing rows back-fill to False.
    op.add_column(
        "agent_config_versions",
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    # Partial unique index — at most one active version. PostgreSQL-specific.
    op.create_index(
        "idx_agent_config_versions_one_active",
        "agent_config_versions",
        ["is_active"],
        unique=True,
        postgresql_where=sa.text("is_active = true"),
    )


def downgrade() -> None:
    op.drop_index(
        "idx_agent_config_versions_one_active", table_name="agent_config_versions"
    )
    op.drop_column("agent_config_versions", "is_active")
