"""add_tool_version_status

Revision ID: i1b2c3d4e5f6
Revises: h1a2b3c4d5e6
Create Date: 2026-06-25 12:00:00.000000

D10 (Phase-3 P2). Adds a review lifecycle to ``ToolVersion`` so an operator can
edit a stored tool's handler/test, stage it for human approval, and only then
promote it into the loadable set. Without this column every persisted version
was implicitly approved, so ``PATCH /tools/{name}`` had no place to park an
edited-but-not-yet-blessed version other than the active row — which
``load_active_tools`` would immediately materialize (defeating the review gate).

Values: ``approved`` (loadable, the default), ``pending_review`` (operator-
edited, awaiting HITL approval), ``rejected`` (dismissed edit). Existing and
auto-persisted versions backfill to ``approved`` so recall never regresses:
``load_active_tools`` (D10) loads only ``status='approved' AND is_active``;
without the backfill every pre-existing tool would vanish from the registry on
the next worker start.

The ``NOT NULL DEFAULT 'approved'`` covers new rows from raw SQL + the ORM
``default``, and the explicit UPDATE below is a defensive idempotent backfill
for rows the ``NOT NULL DEFAULT`` already filled on ``ADD COLUMN`` (Postgres
backfills with the default during ``ALTER TABLE`` on existing rows, but the
``server_default`` keeps the column stable if a later migration ever re-added
it without the default). The partial index backs the "latest pending_review
version for a tool" lookup the approve/reject endpoints run — scoped to
``pending_review`` so it stays small and never covers the (many) approved/
rejected historical rows.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "i1b2c3d4e5f6"
down_revision: Union[str, None] = "h1a2b3c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1) Review-lifecycle column. NOT NULL DEFAULT 'approved' so new rows (ORM
    #    default or raw INSERT) land as loadable, and the ADD COLUMN backfills
    #    every existing row to 'approved' in the same statement.
    op.add_column(
        "tool_versions",
        sa.Column(
            "status",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'approved'"),
        ),
    )

    # 2) Defensive idempotent backfill. ADD COLUMN with a server_default already
    #    set every existing row to 'approved', but this guards a future
    #    re-migration that re-adds the column without the default and keeps the
    #    NOT NULL invariant honest (no NULL survives to violate it).
    op.execute("UPDATE tool_versions SET status = 'approved' WHERE status IS NULL")

    # 3) Partial index backing the approve/reject "latest pending_review version
    #    for a tool" lookup. Scoped to pending_review so it stays tiny relative
    #    to the approved/rejected history and never biases the active-version
    #    query (that one filters is_active + status='approved').
    op.create_index(
        "idx_tool_versions_pending",
        "tool_versions",
        ["tool_id", "version"],
        postgresql_where=sa.text("status = 'pending_review'"),
    )


def downgrade() -> None:
    op.drop_index("idx_tool_versions_pending", table_name="tool_versions")
    op.drop_column("tool_versions", "status")
