"""Pydantic models for structured LLM output from graph nodes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.graph.enums import MutationType, Strategy, TaskComplexity


class TaskClassification(BaseModel):
    """Structured output from the classify node's LLM call."""

    complexity: TaskComplexity = Field(description="Estimated task complexity")
    strategy: Strategy = Field(description="Recommended execution strategy")
    estimated_steps: int = Field(ge=1, le=20, description="Estimated number of steps")
    confidence: float = Field(ge=0.0, le=1.0, description="Classification confidence")
    reasoning: str = Field(default="", description="Brief reasoning for the classification")

    # ── Intent refinement + ambiguity assessment (Feature A) ────────────
    # Additive, all defaulted so a legacy 5-field JSON still parses. This is
    # the detector that feeds the ambiguity-resolution cascade (Feature B).
    # The literal goal text is NEVER replaced — refined_intent is advisory
    # only; it surfaces the real/desired outcome behind the wording.
    refined_intent: str = Field(
        default="",
        description="The real/desired outcome behind the literal goal wording; "
        "empty string if the literal goal already says exactly what is wanted",
    )
    ambiguity_type: str = Field(
        default="none",
        description="none | referential | scope | intent | constraint",
    )
    ambiguity_severity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How under-specified the goal is (0.0 = clear, "
        "1.0 = unresolvable without more input)",
    )
    ambiguity_notes: list[str] = Field(
        default_factory=list,
        description="Specific unresolved points (missing entity / scope / "
        "constraint). Empty when ambiguity_type is none",
    )


class GeneratedStep(BaseModel):
    """A single step in an LLM-generated execution plan."""

    description: str = Field(description="What this step accomplishes")
    tool_name: str | None = Field(default=None, description="Tool to use, if any")
    expected_output: str = Field(default="", description="Expected result of this step")
    depends_on: list[str] = Field(
        default_factory=list,
        description="Descriptions of earlier steps this step depends on (the "
        "'dependencies' decomposition pass); empty if none",
    )


class GeneratedPlan(BaseModel):
    """Structured output from the plan node's LLM call."""

    steps: list[GeneratedStep] = Field(description="Ordered list of plan steps")
    rationale: str = Field(default="", description="Why this plan was chosen")


class DisambiguationResolution(BaseModel):
    """Structured output from the disambiguate node's LLM calls (Feature B).

    One schema serves BOTH cascade LLM calls — the self-resolve pass (which may
    emit ``grounding_queries``) and the re-score pass (which consumes evidence).
    Every field is defaulted so a partial/legacy response still parses.
    """

    proposed_interpretation: str = Field(
        default="",
        description="The most-likely intended outcome of the goal, stated plainly",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Explicit assumptions this interpretation rests on",
    )
    grounding_queries: list[str] = Field(
        default_factory=list,
        description="Web search queries that would resolve the ambiguity "
        "(empty if grounding would not help — e.g. pure intent/constraint gaps)",
    )
    resolved: bool = Field(
        default=False,
        description="True if the ambiguity is resolved to actionable certainty",
    )
    remaining_severity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Residual ambiguity after this pass (0.0 = clear)",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Remaining unresolved points (for the advisory context / HITL)",
    )


class StepAtomicity(BaseModel):
    """Per-step atomicity verdict (Feature C plan-quality validator)."""

    step_id: str = Field(default="", description="The PlanStep.id this verdict covers")
    description: str = Field(default="", description="Truncated step description")
    flag: str = Field(
        default="atomic",
        description="atomic | too_coarse | too_fine",
    )
    reason: str = Field(default="", description="Why the step received this flag")


class PlanQuality(BaseModel):
    """Whole-plan atomicity assessment (Feature C).

    Computed by the plan node's pure heuristic ``_validate_step_atomicity`` and
    attached to state as ``plan_quality`` (a ``model_dump()`` dict —
    AsyncPostgresSaver checkpoint-safe). Serves as advisory telemetry on
    decomposition quality; reflect/verify may read it to decide whether a
    re-plan is warranted.
    """

    per_step: list[StepAtomicity] = Field(default_factory=list)
    atomic: bool = Field(default=True, description="True iff every step is atomic")
    too_coarse_count: int = Field(default=0, ge=0)
    too_fine_count: int = Field(default=0, ge=0)


class ReflectionAnalysis(BaseModel):
    """Structured output from the reflect node's LLM call."""

    progress_assessment: str = Field(description="Assessment of current progress")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in achieving the goal")
    should_replan: bool = Field(default=False, description="Whether to generate a new plan")
    should_evolve: bool = Field(default=False, description="Whether to trigger evolution")
    lessons_learned: list[str] = Field(default_factory=list, description="Key takeaways")
    memory_observations: list[str] = Field(default_factory=list, description="Observations worth storing")
    next_action: str = Field(default="continue", description="Recommended next action")
    missing_tools: list[str] = Field(
        default_factory=list,
        description="Capabilities/tools the agent needed but did not have",
    )
    missing_sub_agents: list[str] = Field(
        default_factory=list,
        description="Descriptions of specialized sub-agents that would help",
    )


class VerificationResult(BaseModel):
    """Structured output from the verify node's LLM call."""

    is_complete: bool = Field(description="Whether the goal is fully achieved")
    completion_percentage: float = Field(ge=0.0, le=100.0, description="Estimated completion")
    gaps: list[str] = Field(default_factory=list, description="Remaining gaps or issues")
    quality_assessment: str = Field(default="", description="Quality of the results")
    should_evolve: bool = Field(default=False, description="Whether evolution could improve results")


class MutationProposal(BaseModel):
    """Structured output from the evolution generate node's LLM call."""

    mutation_type: MutationType = Field(description="Type of mutation to generate")
    target_path: str | None = Field(default=None, description="File path to modify (relative to src/)")
    mutated_content: str = Field(description="The complete modified code or prompt text")
    description: str = Field(description="What this mutation changes")
    rationale: str = Field(default="", description="Why this change should improve performance")


# ── Sub-Agent Schemas ──────────────────────────────────────────────────


class SubAgentProposal(BaseModel):
    """Structured output from agent_spawn node's LLM call."""

    name: str = Field(description="Snake_case sub-agent name (e.g. data_analyst)")
    description: str = Field(description="What this sub-agent specializes in")
    template_type: Literal["fixed", "custom"] = Field(default="fixed", description="fixed or custom")
    tool_scope: Literal["inherit_all", "inherit_subset", "self_create"] = Field(
        default="inherit_all",
        description="inherit_all, inherit_subset, or self_create",
    )
    tool_subset: list[str] = Field(default_factory=list, description="Tool names if inherit_subset")
    model_tier: Literal["trivial", "simple", "complex", "critical"] = Field(
        default="simple",
        description="trivial, simple, complex, or critical",
    )
    goal_description: str = Field(description="The subtask category this agent handles")
    rationale: str = Field(default="", description="Why a dedicated sub-agent is needed")


class DelegationPlan(BaseModel):
    """Structured output from delegate node's LLM call for agent selection."""

    sub_agent_name: str = Field(description="Name of sub-agent to delegate to")
    goal: str = Field(description="Specific subtask goal")
    expected_output: str = Field(description="What the sub-agent should produce")
