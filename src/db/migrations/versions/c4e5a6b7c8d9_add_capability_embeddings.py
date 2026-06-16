"""add_capability_embeddings

Revision ID: c4e5a6b7c8d9
Revises: 73a8b0323eb3
Create Date: 2026-06-16 12:00:00.000000

Adds a nullable ``capability_embedding Vector(768)`` (+ ``capability_text``)
column and an HNSW cosine index to ``tool_registrations`` and
``sub_agent_definitions``. This is the storage backing for semantic
dedup/consolidation (roadmap B3): before creating a tool/sub-agent the agent
embeds the capability gap and cosine-searches these indexes to reuse a
semantically identical capability instead of spawning a duplicate. Nullable so
pre-existing rows and rows created without an embedding API key (hash-fallback
vectors are intentionally NOT stored) remain valid; ``find_similar`` filters
NULLs.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = 'c4e5a6b7c8d9'
down_revision: Union[str, None] = '73a8b0323eb3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_HNSW_WITH = {"m": 16, "ef_construction": 64}
_COSINE_OPS = {"capability_embedding": "vector_cosine_ops"}


def upgrade() -> None:
    # ── tool_registrations ───────────────────────────────────────────────
    op.add_column(
        "tool_registrations",
        sa.Column("capability_embedding", Vector(768), nullable=True),
    )
    op.add_column(
        "tool_registrations",
        sa.Column("capability_text", sa.Text(), nullable=True),
    )
    op.create_index(
        "idx_tool_registrations_capability_emb",
        "tool_registrations",
        ["capability_embedding"],
        postgresql_using="hnsw",
        postgresql_with=_HNSW_WITH,
        postgresql_ops=_COSINE_OPS,
    )

    # ── sub_agent_definitions ────────────────────────────────────────────
    op.add_column(
        "sub_agent_definitions",
        sa.Column("capability_embedding", Vector(768), nullable=True),
    )
    op.add_column(
        "sub_agent_definitions",
        sa.Column("capability_text", sa.Text(), nullable=True),
    )
    op.create_index(
        "idx_sub_agent_capability_emb",
        "sub_agent_definitions",
        ["capability_embedding"],
        postgresql_using="hnsw",
        postgresql_with=_HNSW_WITH,
        postgresql_ops=_COSINE_OPS,
    )


def downgrade() -> None:
    op.drop_index("idx_sub_agent_capability_emb", table_name="sub_agent_definitions")
    op.drop_column("sub_agent_definitions", "capability_text")
    op.drop_column("sub_agent_definitions", "capability_embedding")

    op.drop_index("idx_tool_registrations_capability_emb", table_name="tool_registrations")
    op.drop_column("tool_registrations", "capability_text")
    op.drop_column("tool_registrations", "capability_embedding")
