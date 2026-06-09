"""Pydantic models for structured LLM output from graph nodes."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.graph.enums import MutationType, Strategy, TaskComplexity


class TaskClassification(BaseModel):
    """Structured output from the classify node's LLM call."""

    complexity: TaskComplexity = Field(description="Estimated task complexity")
    strategy: Strategy = Field(description="Recommended execution strategy")
    estimated_steps: int = Field(ge=1, le=20, description="Estimated number of steps")
    confidence: float = Field(ge=0.0, le=1.0, description="Classification confidence")
    reasoning: str = Field(default="", description="Brief reasoning for the classification")


class GeneratedStep(BaseModel):
    """A single step in an LLM-generated execution plan."""

    description: str = Field(description="What this step accomplishes")
    tool_name: str | None = Field(default=None, description="Tool to use, if any")
    expected_output: str = Field(default="", description="Expected result of this step")


class GeneratedPlan(BaseModel):
    """Structured output from the plan node's LLM call."""

    steps: list[GeneratedStep] = Field(description="Ordered list of plan steps")
    rationale: str = Field(default="", description="Why this plan was chosen")


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
