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
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("task_executions.id", ondelete="CASCADE"), nullable=False
    )
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

    # Capability embedding for semantic dedup/consolidation (B3). Nullable:
    # rows created before this migration, or while no embedding API key is
    # available (hash-fallback vectors are not stored — see
    # ``EmbeddingGenerator.last_source``), are NULL. ``find_similar`` filters
    # them. ``capability_text`` is the embedded text, kept for re-embedding/debug.
    capability_embedding: Mapped[Vector] = mapped_column(Vector(768), nullable=True)
    capability_text: Mapped[str | None] = mapped_column(Text, nullable=True)

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
