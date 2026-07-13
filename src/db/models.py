from __future__ import annotations

import datetime as dt
import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


def _utcnow() -> dt.datetime:
    """UTC-aware datetime for use as column default."""
    return dt.datetime.now(dt.timezone.utc)


# =============================================================================
# Task Execution Domain
# =============================================================================


class TaskExecution(Base):
    """Main task records with goal, strategy, complexity, costs, and status."""

    __tablename__ = "task_executions"

    id: Mapped[uuid.UUID] = mapped_column(default=uuid.uuid4, primary_key=True)
    thread_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    goal_text: Mapped[str] = mapped_column(Text, nullable=False)
    goal_priority: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=5
    )
    strategy: Mapped[str] = mapped_column(Text, nullable=False, default="react")
    complexity: Mapped[str] = mapped_column(Text, nullable=False, default="simple")
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")
    model_used: Mapped[str | None] = mapped_column(Text, nullable=True)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_cost_usd: Mapped[float] = mapped_column(
        Numeric(10, 6), nullable=False, default=0
    )
    result_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    completed_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    extra_data: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)

    # Relationships
    execution_steps: Mapped[list[ExecutionStep]] = relationship(
        back_populates="task_execution", cascade="all, delete-orphan"
    )
    feedback_events: Mapped[list[FeedbackEvent]] = relationship(
        back_populates="task_execution", cascade="all, delete-orphan"
    )
    warm_memories: Mapped[list[WarmMemory]] = relationship(back_populates="source_task")
    cost_ledger_entries: Mapped[list[CostLedger]] = relationship(
        back_populates="task_execution"
    )

    # Indexes
    __table_args__ = (
        CheckConstraint("goal_priority BETWEEN 1 AND 10", name="check_goal_priority_range"),
        Index("idx_task_executions_thread", "thread_id"),
        Index("idx_task_executions_created", "created_at"),
        Index("idx_task_executions_status", "status"),
    )


class ExecutionStep(Base):
    """Individual execution phases with tool calls, results, and status."""

    __tablename__ = "execution_steps"

    id: Mapped[uuid.UUID] = mapped_column(default=uuid.uuid4, primary_key=True)
    task_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("task_executions.id", ondelete="CASCADE"), nullable=True
    )
    # Track-1 per-node timing attribution. NULL on legacy task-execution rows;
    # populated by the graph _wrap node-timer (timing-only rows carrying no
    # task_executions parent). run_metrics keys off this for per-node wall-clock.
    run_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    step_number: Mapped[int] = mapped_column(Integer, nullable=False)
    phase: Mapped[str] = mapped_column(Text, nullable=False)
    tool_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_input: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    tool_output: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    tokens_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    # Relationships
    task_execution: Mapped[TaskExecution] = relationship(back_populates="execution_steps")
    feedback_events: Mapped[list[FeedbackEvent]] = relationship(
        back_populates="execution_step"
    )

    # Indexes
    __table_args__ = (
        Index("idx_execution_steps_task_number", "task_id", "step_number"),
        Index("idx_execution_steps_phase", "phase"),
        Index("idx_execution_steps_failed", "task_id", postgresql_where="status = 'failed'"),
    )


class FeedbackEvent(Base):
    """Human feedback and quality ratings for tasks and steps."""

    __tablename__ = "feedback_events"

    id: Mapped[uuid.UUID] = mapped_column(default=uuid.uuid4, primary_key=True)
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("task_executions.id", ondelete="CASCADE"), nullable=False
    )
    step_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("execution_steps.id"), nullable=True
    )
    event_type: Mapped[str] = mapped_column(Text, nullable=False)
    rating: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    # Relationships
    task_execution: Mapped[TaskExecution] = relationship(back_populates="feedback_events")
    execution_step: Mapped[ExecutionStep | None] = relationship(
        back_populates="feedback_events"
    )

    # Indexes
    __table_args__ = (
        CheckConstraint("rating BETWEEN 1 AND 5", name="check_rating_range"),
        Index("idx_feedback_events_task", "task_id", "created_at"),
        Index("idx_feedback_events_type", "event_type"),
    )


# =============================================================================
# Memory Domain
# =============================================================================


class WarmMemory(Base):
    """Frequently accessed skills/procedures with fitness scores and tags."""

    __tablename__ = "warm_memories"

    id: Mapped[uuid.UUID] = mapped_column(default=uuid.uuid4, primary_key=True)
    memory_type: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source_task_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("task_executions.id"), nullable=True
    )
    fitness_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    access_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_accessed: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    expires_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    tags: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    extra_data: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    # A5: durable dedup key for facts (memory_type='fact'). NULL for every other
    # memory type, so the partial unique index below dedups facts without ever
    # colliding with skills/procedures (NULLs are distinct). Upserted by
    # WarmMemoryStore.store_fact via ON CONFLICT (fact_key).
    fact_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    # Relationships
    source_task: Mapped[TaskExecution | None] = relationship(back_populates="warm_memories")
    memory_embeddings: Mapped[list[MemoryEmbedding]] = relationship(
        back_populates="warm_memory", cascade="all, delete-orphan"
    )
    skill_versions: Mapped[list[SkillVersion]] = relationship(
        back_populates="warm_memory", cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        CheckConstraint("fitness_score BETWEEN 0 AND 1", name="check_fitness_score_range"),
        Index("idx_warm_memories_type", "memory_type"),
        Index("idx_warm_memories_tags", "tags", postgresql_using="gin"),
        Index(
            "idx_warm_memories_fitness",
            "fitness_score",
            postgresql_where="fitness_score > 0.7",
        ),
        Index(
            "idx_warm_memories_active",
            "memory_type",
            "fitness_score",
            postgresql_where="expires_at IS NULL",
        ),
        # A5: partial unique index backing fact dedup. Scoped to ACTIVE facts so a
        # retired (expires_at-set) fact never shadows a freshly-extracted one —
        # retrieve_facts filters expires_at IS NULL, so a full index would silently
        # redirect re-extraction onto a retired row (invisible to recall). NULL
        # fact_key on every non-fact row means they never collide here either.
        Index(
            "uq_warm_memories_fact_key",
            "fact_key",
            unique=True,
            postgresql_where="memory_type = 'fact' AND expires_at IS NULL",
        ),
    )


class ColdMemory(Base):
    """Archived experiences with pgvector embeddings and forgetting curve."""

    __tablename__ = "cold_memories"

    id: Mapped[uuid.UUID] = mapped_column(default=uuid.uuid4, primary_key=True)
    episode_type: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    context_tags: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    importance: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    embedding: Mapped[Vector] = mapped_column(Vector(768), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    expires_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    retention_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    stability_factor: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_reinforced_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    extra_data: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)

    # Indexes
    __table_args__ = (
        CheckConstraint("importance BETWEEN 0 AND 1", name="check_importance_range"),
        Index("idx_cold_memories_type", "episode_type"),
        Index("idx_cold_memories_importance", "importance"),
        Index("idx_cold_memories_tags", "context_tags", postgresql_using="gin"),
        Index(
            "idx_cold_memories_active",
            "episode_type",
            "importance",
            postgresql_where="expires_at IS NULL",
        ),
        Index(
            "idx_cold_memories_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class MemoryEmbedding(Base):
    """Vector storage for warm memories with pgvector."""

    __tablename__ = "memory_embeddings"

    id: Mapped[uuid.UUID] = mapped_column(default=uuid.uuid4, primary_key=True)
    memory_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("warm_memories.id", ondelete="CASCADE"), nullable=False
    )
    embedding: Mapped[Vector] = mapped_column(Vector(768), nullable=True)
    embedding_model: Mapped[str] = mapped_column(Text, nullable=False, default="text-embedding-3-small")
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    # Relationships
    warm_memory: Mapped[WarmMemory] = relationship(back_populates="memory_embeddings")

    # Indexes
    __table_args__ = (
        Index("idx_memory_embeddings_model", "embedding_model"),
        Index(
            "idx_memory_embeddings_vector",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class SkillVersion(Base):
    """Version history for evolved skills."""

    __tablename__ = "skill_versions"

    id: Mapped[uuid.UUID] = mapped_column(default=uuid.uuid4, primary_key=True)
    memory_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("warm_memories.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    code_content: Mapped[str] = mapped_column(Text, nullable=False)
    test_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    test_results: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    # Relationships
    warm_memory: Mapped[WarmMemory] = relationship(back_populates="skill_versions")

    # Indexes and constraints
    __table_args__ = (
        UniqueConstraint("memory_id", "version", name="uq_skill_version"),
        Index("idx_skill_versions_active", "memory_id", postgresql_where="is_active = true"),
        Index("idx_skill_versions_history", "memory_id", "version"),
    )


# =============================================================================
# Evolution Domain
# =============================================================================


class MutationChain(Base):
    """Sequences of related mutations with parent/child relationships."""

    __tablename__ = "mutation_chains"

    id: Mapped[uuid.UUID] = mapped_column(default=uuid.uuid4, primary_key=True)
    trigger_reason: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")
    parent_chain_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("mutation_chains.id"), nullable=True
    )
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    completed_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    extra_data: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)

    # Relationships
    parent_chain: Mapped[MutationChain | None] = relationship(
        back_populates="child_chains", remote_side=[id]
    )
    child_chains: Mapped[list[MutationChain]] = relationship(
        back_populates="parent_chain"
    )
    mutations: Mapped[list[Mutation]] = relationship(
        back_populates="mutation_chain", cascade="all, delete-orphan"
    )
    evolution_telemetry: Mapped[list[EvolutionTelemetry]] = relationship(
        back_populates="mutation_chain"
    )

    # Indexes
    __table_args__ = (
        Index(
            "idx_mutation_chains_status",
            "status",
            postgresql_where="status IN ('pending', 'in_progress')",
        ),
        Index("idx_mutation_chains_parent", "parent_chain_id"),
        Index("idx_mutation_chains_trigger", "trigger_reason", "created_at"),
    )


class Mutation(Base):
    """Individual code/prompt changes with approval status."""

    __tablename__ = "mutations"

    id: Mapped[uuid.UUID] = mapped_column(default=uuid.uuid4, primary_key=True)
    chain_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("mutation_chains.id", ondelete="CASCADE"), nullable=False
    )
    mutation_type: Mapped[str] = mapped_column(Text, nullable=False)
    target_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    original_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    mutated_content: Mapped[str] = mapped_column(Text, nullable=False)
    diff_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_used: Mapped[str | None] = mapped_column(Text, nullable=True)
    tokens_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="generated")
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    # Relationships
    mutation_chain: Mapped[MutationChain] = relationship(back_populates="mutations")
    ab_test_results: Mapped[list[ABTestResult]] = relationship(
        back_populates="mutation", cascade="all, delete-orphan"
    )
    code_versions: Mapped[list[CodeVersion]] = relationship(
        back_populates="mutation", cascade="all, delete-orphan"
    )
    tool_registrations: Mapped[list[ToolRegistration]] = relationship(
        back_populates="source_mutation"
    )

    # Indexes
    __table_args__ = (
        Index("idx_mutations_chain", "chain_id", "created_at"),
        Index("idx_mutations_type", "mutation_type"),
        Index(
            "idx_mutations_pending",
            "chain_id",
            postgresql_where="status IN ('generated', 'approved')",
        ),
    )


class ABTestResult(Base):
    """Statistical test results with p-values, significance, and metrics."""

    __tablename__ = "ab_test_results"

    id: Mapped[uuid.UUID] = mapped_column(default=uuid.uuid4, primary_key=True)
    mutation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("mutations.id", ondelete="CASCADE"), nullable=False
    )
    metric_name: Mapped[str] = mapped_column(Text, nullable=False)
    control_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    treatment_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    sample_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    p_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_significant: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    # Relationships
    mutation: Mapped[Mutation] = relationship(back_populates="ab_test_results")

    # Indexes
    __table_args__ = (
        Index("idx_ab_test_results_mutation", "mutation_id"),
        Index("idx_ab_test_results_metric", "metric_name", "created_at"),
        Index(
            "idx_ab_test_results_significant",
            "metric_name",
            postgresql_where="is_significant = true",
        ),
    )


class CodeVersion(Base):
    """Git-based version tracking with rollback capability."""

    __tablename__ = "code_versions"

    id: Mapped[uuid.UUID] = mapped_column(default=uuid.uuid4, primary_key=True)
    mutation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("mutations.id", ondelete="CASCADE"), nullable=False
    )
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    version_hash: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    deployed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    deployed_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    rolled_back: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    # Relationships
    mutation: Mapped[Mutation] = relationship(back_populates="code_versions")

    # Indexes
    __table_args__ = (
        Index("idx_code_versions_mutation", "mutation_id"),
        Index("idx_code_versions_path", "file_path", "created_at"),
        Index(
            "idx_code_versions_deployed",
            "file_path",
            "deployed_at",
            postgresql_where="deployed = true",
        ),
        Index(
            "idx_code_versions_rollback",
            "file_path",
            "created_at",
            postgresql_where="rolled_back = false",
        ),
    )


# =============================================================================
# Tools Domain
# =============================================================================


class ToolRegistration(Base):
    """Built-in, evolved, and MCP tools with JSONB schemas."""

    __tablename__ = "tool_registrations"

    id: Mapped[uuid.UUID] = mapped_column(default=uuid.uuid4, primary_key=True)
    tool_name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    tool_type: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    input_schema: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    source_mutation_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("mutations.id"), nullable=True
    )
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    # Run that registered/generated this tool (Track-1 attribution). NULL for
    # built-ins + pre-migration rows; populated for generated tools at persist
    # time via the active run_id contextvar (get_active_run_id).
    owner_run_id: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Capability embedding for semantic dedup/consolidation (B3). Nullable:
    # rows created before this migration, or while no embedding API key is
    # available (hash-fallback vectors are not stored — see
    # ``EmbeddingGenerator.last_source``), are NULL. ``find_similar`` filters
    # them. ``capability_text`` is the embedded text, kept for re-embedding/debug.
    capability_embedding: Mapped[Vector] = mapped_column(Vector(768), nullable=True)
    capability_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Per-tool success metrics (M4) — running aggregates maintained
    # incrementally by ``ToolMetricsRecorder`` on every invocation, so
    # performance-based retirement can score a tool without re-aggregating the
    # ``tool_call_metrics`` detail table. ``calls`` is the running mean count;
    # ``success_rate``/``empty_output_rate`` are incremental means in [0, 1]
    # (seeded 1.0/0.0 so an untried tool is never retired for performance).
    # ``last_run_at`` is NULL until the tool's first invocation.
    calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    success_rate: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    empty_output_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    last_run_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    source_mutation: Mapped[Mutation | None] = relationship(back_populates="tool_registrations")
    tool_versions: Mapped[list[ToolVersion]] = relationship(
        back_populates="tool_registration", cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        Index("idx_tool_registrations_active", "tool_name", postgresql_where="is_active = true"),
        Index("idx_tool_registrations_type", "tool_type"),
        Index(
            "idx_tool_registrations_capability_emb",
            "capability_embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"capability_embedding": "vector_cosine_ops"},
        ),
    )


class ToolCallMetric(Base):
    """Append-only per-invocation tool metric (M4).

    One row per executed tool call — the source-of-truth audit trail behind the
    running aggregates on :class:`ToolRegistration` (``calls``/``success_rate``/
    ``empty_output_rate``/``last_run_at``). Nullable ``run_id``: the execute
    chokepoint records with the run's thread id when available.
    """

    __tablename__ = "tool_call_metrics"

    id: Mapped[uuid.UUID] = mapped_column(default=uuid.uuid4, primary_key=True)
    tool_name: Mapped[str] = mapped_column(Text, nullable=False)
    run_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    empty_output: Mapped[bool] = mapped_column(Boolean, nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    __table_args__ = (Index("idx_tool_call_metrics_tool", "tool_name", "created_at"),)


class ToolVersion(Base):
    """Version history for evolving tools."""

    __tablename__ = "tool_versions"

    id: Mapped[uuid.UUID] = mapped_column(default=uuid.uuid4, primary_key=True)
    tool_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tool_registrations.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    code_content: Mapped[str] = mapped_column(Text, nullable=False)
    test_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    test_pass_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # D10 review lifecycle: ``approved`` (loadable), ``pending_review`` (operator-
    # edited, awaiting HITL approval), ``rejected``. Existing/auto-persisted
    # versions default to ``approved`` so the migration backfill + ``persist()``
    # never regress recall (``load_active_tools`` requires status='approved' AND
    # is_active). ``server_default`` keeps raw SQL inserts + alembic consistent.
    status: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="approved",
        server_default=text("'approved'"),
    )
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    # Relationships
    tool_registration: Mapped[ToolRegistration] = relationship(back_populates="tool_versions")

    # Indexes and constraints
    __table_args__ = (
        UniqueConstraint("tool_id", "version", name="uq_tool_version"),
        Index("idx_tool_versions_active", "tool_id", postgresql_where="is_active = true"),
        Index("idx_tool_versions_history", "tool_id", "version"),
    )


# =============================================================================
# Sub-Agent Domain
# =============================================================================


class SubAgentModel(Base):
    """Sub-agent definitions with configuration, versioning, and rolling performance metrics."""

    __tablename__ = "sub_agent_definitions"

    id: Mapped[uuid.UUID] = mapped_column(default=uuid.uuid4, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)

    # ── Configuration ────────────────────────────────────────────────────
    template_type: Mapped[str] = mapped_column(
        Text, nullable=False, default="fixed"
    )  # "fixed" or "custom"
    tool_scope: Mapped[str] = mapped_column(
        Text, nullable=False, default="inherit_all"
    )  # "inherit_all", "inherit_subset", "self_create"
    tool_subset: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    budget_mode: Mapped[str] = mapped_column(
        Text, nullable=False, default="shared"
    )  # "shared" or "separate"
    budget_limit: Mapped[float] = mapped_column(
        Numeric(10, 6), nullable=False, default=0
    )
    model_tier: Mapped[str] = mapped_column(
        Text, nullable=False, default="simple"
    )  # TaskComplexity value
    max_iterations: Mapped[int] = mapped_column(Integer, nullable=False, default=10)
    depth_limit: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # ── Custom subgraph config (for template_type="custom") ──────────────
    node_config: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    system_prompt_override: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── Rolling performance metrics ──────────────────────────────────────
    total_runs: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    success_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    success_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    avg_cost: Mapped[float] = mapped_column(Numeric(10, 6), nullable=False, default=0)
    avg_latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    quality_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)

    # ── Lifecycle ────────────────────────────────────────────────────────
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    source_mutation_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("mutations.id"), nullable=True
    )
    # Run that spawned this sub-agent (Track-1 attribution). NULL for
    # pre-migration rows; populated at persist time via the active run_id
    # contextvar (get_active_run_id).
    owner_run_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Capability embedding for semantic dedup/consolidation (B3). See
    # ``ToolRegistration.capability_embedding`` for the nullable rationale.
    capability_embedding: Mapped[Vector] = mapped_column(Vector(768), nullable=True)
    capability_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    # Relationships
    runs: Mapped[list[SubAgentRunModel]] = relationship(
        back_populates="sub_agent", cascade="all, delete-orphan"
    )
    source_mutation: Mapped[Mutation | None] = relationship()

    # Indexes and constraints
    __table_args__ = (
        CheckConstraint(
            "success_rate BETWEEN 0 AND 1", name="check_sa_success_rate"
        ),
        CheckConstraint(
            "quality_score BETWEEN 0 AND 1", name="check_sa_quality_score"
        ),
        Index(
            "idx_sub_agents_active",
            "name",
            postgresql_where="is_active = true",
        ),
        Index("idx_sub_agents_template", "template_type"),
        Index(
            "idx_sub_agents_performance",
            "success_rate",
            postgresql_where="is_active = true",
        ),
        Index(
            "idx_sub_agent_capability_emb",
            "capability_embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"capability_embedding": "vector_cosine_ops"},
        ),
    )


class SubAgentRunModel(Base):
    """Individual sub-agent execution records for performance tracking."""

    __tablename__ = "sub_agent_runs"

    id: Mapped[uuid.UUID] = mapped_column(default=uuid.uuid4, primary_key=True)
    sub_agent_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sub_agent_definitions.id", ondelete="CASCADE"),
        nullable=False,
    )
    parent_task_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("task_executions.id"), nullable=True
    )
    parent_thread_id: Mapped[str] = mapped_column(Text, nullable=False)

    # ── Input/Output ─────────────────────────────────────────────────────
    goal_text: Mapped[str] = mapped_column(Text, nullable=False)
    result_summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── Execution metrics ────────────────────────────────────────────────
    status: Mapped[str] = mapped_column(
        Text, nullable=False, default="pending"
    )  # pending, running, completed, failed, timeout
    iterations_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tokens_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost_usd: Mapped[float] = mapped_column(Numeric(10, 6), nullable=False, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # ── Quality assessment ───────────────────────────────────────────────
    quality_rating: Mapped[float | None] = mapped_column(Float, nullable=True)

    extra_data: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    completed_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    sub_agent: Mapped[SubAgentModel] = relationship(back_populates="runs")

    # Indexes
    __table_args__ = (
        Index("idx_sub_agent_runs_agent", "sub_agent_id", "created_at"),
        Index("idx_sub_agent_runs_parent", "parent_thread_id"),
        Index("idx_sub_agent_runs_status", "status"),
    )


# =============================================================================
# Knowledge Graph Domain
# =============================================================================


class KnowledgeEntity(Base):
    """Concepts, tools, APIs, patterns, errors with embeddings."""

    __tablename__ = "knowledge_entities"

    id: Mapped[uuid.UUID] = mapped_column(default=uuid.uuid4, primary_key=True)
    entity_type: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    properties: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    embedding: Mapped[Vector] = mapped_column(Vector(768), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    # Relationships
    source_relations: Mapped[list[EntityRelation]] = relationship(
        back_populates="source_entity",
        foreign_keys="[EntityRelation.source_id]",
        cascade="all, delete-orphan",
    )
    target_relations: Mapped[list[EntityRelation]] = relationship(
        back_populates="target_entity",
        foreign_keys="[EntityRelation.target_id]",
        cascade="all, delete-orphan",
    )

    # Indexes and constraints
    __table_args__ = (
        UniqueConstraint("entity_type", "name", name="uq_entity_type_name"),
        Index("idx_knowledge_entities_type", "entity_type"),
        Index("idx_knowledge_entities_properties", "properties", postgresql_using="gin"),
        Index(
            "idx_knowledge_entities_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class EntityRelation(Base):
    """Typed relations with strength score."""

    __tablename__ = "entity_relations"

    id: Mapped[uuid.UUID] = mapped_column(default=uuid.uuid4, primary_key=True)
    source_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("knowledge_entities.id", ondelete="CASCADE"), nullable=False
    )
    target_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("knowledge_entities.id", ondelete="CASCADE"), nullable=False
    )
    relation_type: Mapped[str] = mapped_column(Text, nullable=False)
    strength: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    extra_data: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    # Relationships
    source_entity: Mapped[KnowledgeEntity] = relationship(
        back_populates="source_relations",
        foreign_keys=[source_id],
    )
    target_entity: Mapped[KnowledgeEntity] = relationship(
        back_populates="target_relations",
        foreign_keys=[target_id],
    )

    # Indexes and constraints
    __table_args__ = (
        UniqueConstraint("source_id", "target_id", "relation_type", name="uq_entity_relation"),
        CheckConstraint("strength BETWEEN 0 AND 1", name="check_strength_range"),
        Index("idx_entity_relations_source", "source_id", "relation_type"),
        Index("idx_entity_relations_target", "target_id", "relation_type"),
        Index(
            "idx_entity_relations_strength",
            "relation_type",
            "strength",
            postgresql_where="strength > 0.7",
        ),
    )


# =============================================================================
# Observability Domain
# =============================================================================


class CostLedger(Base):
    """Per-request token counts and costs with provider, model, task_id."""

    __tablename__ = "cost_ledger"

    id: Mapped[uuid.UUID] = mapped_column(default=uuid.uuid4, primary_key=True)
    task_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("task_executions.id"), nullable=True
    )
    # Per-run correlation key — the graph ``thread_id`` of the run that issued
    # the call (``cli-{run_id}`` for a ``--run-id`` run, else ``cli-{pid}-{obj}``
    # so even a run with no explicit id is attributable to one process). Unlike
    # ``task_id`` (a UUID FK into task_executions, which the agent never
    # populates), ``run_id`` is a free Text column that always carries the run
    # identifier, enabling per-run cost attribution via get_run_spend/runs_summary.
    run_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    provider: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(Text, nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cached_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost_usd: Mapped[float] = mapped_column(Numeric(10, 6), nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    # Relationships
    task_execution: Mapped[TaskExecution | None] = relationship(
        back_populates="cost_ledger_entries"
    )

    # Indexes
    __table_args__ = (
        Index("idx_cost_ledger_task", "task_id", "created_at"),
        Index("idx_cost_ledger_run", "run_id", "created_at"),
        Index("idx_cost_ledger_provider", "provider", "model", "created_at"),
        Index("idx_cost_ledger_date", "created_at"),
    )


class EvolutionTelemetry(Base):
    """Evolution cycle metrics."""

    __tablename__ = "evolution_telemetry"

    id: Mapped[uuid.UUID] = mapped_column(default=uuid.uuid4, primary_key=True)
    chain_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("mutation_chains.id"), nullable=True
    )
    event_type: Mapped[str] = mapped_column(Text, nullable=False)
    event_data: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    # Relationships
    mutation_chain: Mapped[MutationChain | None] = relationship(
        back_populates="evolution_telemetry"
    )

    # Indexes
    __table_args__ = (
        Index("idx_evolution_telemetry_chain", "chain_id", "created_at"),
        Index("idx_evolution_telemetry_type", "event_type", "created_at"),
    )


class AgentConfigVersion(Base):
    """Configuration versioning."""

    __tablename__ = "agent_config_versions"

    id: Mapped[uuid.UUID] = mapped_column(default=uuid.uuid4, primary_key=True)
    config_snapshot: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    change_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    # Relationships
    config_snapshots: Mapped[list[ConfigSnapshot]] = relationship(
        back_populates="config_version", cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (Index("idx_agent_config_versions_created", "created_at"),)


class ConfigSnapshot(Base):
    """Configuration state snapshots."""

    __tablename__ = "config_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(default=uuid.uuid4, primary_key=True)
    config_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("agent_config_versions.id"), nullable=False
    )
    key_path: Mapped[str] = mapped_column(Text, nullable=False)
    old_value: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    new_value: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    # Relationships
    config_version: Mapped[AgentConfigVersion] = relationship(
        back_populates="config_snapshots"
    )

    # Indexes
    __table_args__ = (
        Index("idx_config_snapshots_version", "config_version_id"),
        Index("idx_config_snapshots_key", "key_path", "created_at"),
    )


# =============================================================================
# Evaluation Domain (Phase 3 correctness harness)
# =============================================================================


class EvalResult(Base):
    """Per-check correctness result row for the evaluation harness.

    One row per (run, goal, check) so the eval store is queryable by goal or
    run for regression tracking and the Phase-8 evolution canary. The full
    per-run aggregate lives in ``BenchmarkResult`` (in-memory/JSON export); this
    table is the durable, queryable projection.
    """

    __tablename__ = "eval_results"

    id: Mapped[uuid.UUID] = mapped_column(default=uuid.uuid4, primary_key=True)
    goal_id: Mapped[str] = mapped_column(Text, nullable=False)
    run_id: Mapped[str] = mapped_column(Text, nullable=False)
    # Per-run-attempt discriminator (Phase-3 follow-up). ``run_id`` is the graph
    # ``thread_id`` and is STABLE across re-runs of the same ``--run_id`` (it is
    # the resume key), so without this column re-running q03 would blend every
    # attempt's rows under one ``run_id`` and ``cli-q03 = 0.750`` would be a
    # blend, not one attempt's score. ``attempt_id`` is generated once per
    # invocation (main.py) so a score means ONE attempt. Nullable for back-compat
    # with legacy rows + any future unattributed write.
    attempt_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    spec_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    check_name: Mapped[str] = mapped_column(Text, nullable=False)
    check_type: Mapped[str] = mapped_column(Text, nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    score: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False, default=0.0)
    skipped: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    evidence: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    cost_usd: Mapped[float] = mapped_column(Numeric(10, 6), nullable=False, default=0.0)
    # Producer-model attribution (Phase-2): the model id that ran the goal's
    # execute step, so the capability curve can be sliced per-model
    # (``curve --model glm-4.7``) instead of reading a blended system-wide trend
    # — the thesis ("self-improvement") is model-specific. Nullable for back-compat
    # with legacy rows + unattributed writes (the verify node resolves it only when
    # a gateway is present).
    producer_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    # Indexes
    __table_args__ = (
        Index("idx_eval_results_goal", "goal_id", "created_at"),
        Index("idx_eval_results_run", "run_id", "check_name"),
        # (run_id, attempt_id) so query_latest_attempt can pick the newest
        # attempt of a run without scanning every row.
        Index("idx_eval_results_attempt", "run_id", "attempt_id", "created_at"),
    )


class ScheduledTask(Base):
    """An agent-authored durable cron task (Phase 5 I1).

    One row = one future run the agent asked to fire on a cron schedule via the
    ``create_scheduled_task`` builtin. The scheduler consumer
    (``src.scheduler.cron_consumer``) polls this table, registers/refreshes an
    APScheduler ``CronTrigger`` job per enabled row, and on each fire enqueues a
    ``RunJob`` through the SAME ``RunsQueue.enqueue`` seam the API/battery use —
    so the run goes through the real deployed worker stack (lease-lock,
    checkpoint, eval-resolution all apply unchanged). The table is the durable
    substrate; APScheduler is the in-memory trigger.

    ``name`` is the agent's stable handle: re-calling the tool with the same
    name UPSERTs (updates cron/goal/model) instead of duplicating, so an agent
    can revise its own schedule without accumulating stale rows. The unique
    index backs that upsert. ``owner_run_id`` is provenance (the run that
    authored the task), read from the ``_active_run_id`` contextvar when bound.
    """

    __tablename__ = "scheduled_tasks"

    id: Mapped[uuid.UUID] = mapped_column(default=uuid.uuid4, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    # 5-field crontab, validated by CronTrigger.from_crontab at creation time.
    cron: Mapped[str] = mapped_column(Text, nullable=False)
    # The goal the fired run will accomplish (the enqueued RunJob.goal).
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    # Optional pinned model (registry key / litellm id). NULL → tiered default.
    model: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Provenance: the run that authored this task (None for operator/manual rows).
    owner_run_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Whether the consumer should register/fire it. Disabled rows persist as history.
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    # IANA zone for the cron (CronTrigger.from_crontab takes a timezone).
    timezone: Mapped[str] = mapped_column(Text, nullable=False, default="UTC")
    # Informational next fire (UTC); APScheduler is authoritative. Refreshed by
    # the consumer sync. Nullable so a freshly-inserted row (before first sync)
    # is valid, and so a disabled row has no misleading future time.
    next_fire_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    # Indexes
    __table_args__ = (
        # The consumer's primary read: every enabled row, soonest-first.
        Index("idx_scheduled_tasks_enabled", "enabled", "next_fire_at"),
        # Provenance lookups ("which tasks did run X author?").
        Index("idx_scheduled_tasks_owner", "owner_run_id"),
        # The agent's stable handle — backs the upsert-by-name semantics.
        UniqueConstraint("name", name="uq_scheduled_tasks_name"),
    )
