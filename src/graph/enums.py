"""Enums for graph state and routing."""

from __future__ import annotations

from enum import Enum


class Phase(str, Enum):
    """Graph execution phases — drives conditional edge routing."""

    CLASSIFY = "classify"
    DISAMBIGUATE = "disambiguate"
    PLAN = "plan"
    RETRIEVE_MEMORY = "retrieve_memory"
    RESEARCH = "research"
    STRUCTURE_ANALYSIS = "structure_analysis"
    EXECUTE = "execute"
    REFLECT = "reflect"
    VERIFY = "verify"
    EVOLVE = "evolve"
    STORE_MEMORY = "store_memory"
    TOOL_CREATE = "tool_create"
    AGENT_SPAWN = "agent_spawn"
    DELEGATE = "delegate"
    HITL_GATE = "hitl_gate"
    ERROR_HANDLER = "error_handler"
    COMPLETE = "complete"


class Strategy(str, Enum):
    """Task execution strategies."""

    DIRECT = "direct"
    REACT = "react"
    REFLECTION = "reflection"
    PLANNING = "planning"
    TOT = "tree_of_thought"
    REWOO = "rewoo"
    DEBATE = "debate"


class TaskComplexity(str, Enum):
    """Task complexity levels for model routing."""

    TRIVIAL = "trivial"
    SIMPLE = "simple"
    COMPLEX = "complex"
    CRITICAL = "critical"


class Confidence(str, Enum):
    """Confidence levels for reflection results."""

    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class GoalStatus(str, Enum):
    """Status of goals and plan steps."""

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    DELEGATED = "delegated"


class MutationType(str, Enum):
    """Types of evolution mutations."""

    PROMPT = "prompt"
    CODE = "code"
    TOOL = "tool"
    WORKFLOW = "workflow"
    MEMORY = "memory"
    CONFIG = "config"
    SUB_AGENT_PROMPT = "sub_agent_prompt"
    SUB_AGENT_TOOLS = "sub_agent_tools"
    SUB_AGENT_CONFIG = "sub_agent_config"
    SUB_AGENT_MODEL_TIER = "sub_agent_model_tier"
