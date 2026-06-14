"""Prompt template package with jinja2-backed templates.

All prompt constants are PromptTemplate instances that support ``.format(**kwargs)``
for backward compatibility with the old string-constant approach. Node files
importing from ``src.graph.prompts`` work unchanged.
"""

from __future__ import annotations

from src.graph.prompts.builder import build_messages, splice_techniques
from src.graph.prompts.loader import PromptManager, PromptTemplate
from src.graph.prompts.technique_selector import (
    JSON_SCHEMA_MARKER,
    NODE_EXECUTE,
    NODE_PLAN,
    NODE_REFLECT,
    NODE_VERIFY,
    TECHNIQUE_REGISTRY,
    Technique,
    TechniqueSelector,
)

# ── Singleton manager ──────────────────────────────────────────────────
prompt_manager = PromptManager()

# ── Task Classification ────────────────────────────────────────────────
CLASSIFY_SYSTEM = PromptTemplate("classify_system")
CLASSIFY_USER = PromptTemplate("classify_user")

# ── Planning ───────────────────────────────────────────────────────────
PLAN_SYSTEM = PromptTemplate("plan_system")
PLAN_USER = PromptTemplate("plan_user")

# ── Execution ──────────────────────────────────────────────────────────
EXECUTE_SYSTEM = PromptTemplate("execute_system")

# ── Reflection ─────────────────────────────────────────────────────────
REFLECT_SYSTEM = PromptTemplate("reflect_system")
REFLECT_USER = PromptTemplate("reflect_user")

# ── Verification ───────────────────────────────────────────────────────
VERIFY_SYSTEM = PromptTemplate("verify_system")
VERIFY_USER = PromptTemplate("verify_user")

# ── Evolution Generation ───────────────────────────────────────────────
EVOLUTION_GENERATE_SYSTEM = PromptTemplate("evolution_generate_system")
EVOLUTION_GENERATE_USER = PromptTemplate("evolution_generate_user")

# ── Dynamic Tool Generation ────────────────────────────────────────────
TOOL_GENERATE_SYSTEM = PromptTemplate("tool_generate_system")
TOOL_GENERATE_USER = PromptTemplate("tool_generate_user")

# ── Sub-Agent Spawning ─────────────────────────────────────────────────
AGENT_SPAWN_SYSTEM = PromptTemplate("agent_spawn_system")
AGENT_SPAWN_USER = PromptTemplate("agent_spawn_user")

# ── Task Delegation ────────────────────────────────────────────────────
DELEGATE_SYSTEM = PromptTemplate("delegate_system")
DELEGATE_USER = PromptTemplate("delegate_user")

__all__ = [
    "PromptManager",
    "PromptTemplate",
    "prompt_manager",
    # ── §5 prompting-technique layer ────────────────────────────────────
    "build_messages",
    "splice_techniques",
    "Technique",
    "TechniqueSelector",
    "TECHNIQUE_REGISTRY",
    "JSON_SCHEMA_MARKER",
    "NODE_PLAN",
    "NODE_EXECUTE",
    "NODE_REFLECT",
    "NODE_VERIFY",
    "CLASSIFY_SYSTEM",
    "CLASSIFY_USER",
    "PLAN_SYSTEM",
    "PLAN_USER",
    "EXECUTE_SYSTEM",
    "REFLECT_SYSTEM",
    "REFLECT_USER",
    "VERIFY_SYSTEM",
    "VERIFY_USER",
    "EVOLUTION_GENERATE_SYSTEM",
    "EVOLUTION_GENERATE_USER",
    "TOOL_GENERATE_SYSTEM",
    "TOOL_GENERATE_USER",
    "AGENT_SPAWN_SYSTEM",
    "AGENT_SPAWN_USER",
    "DELEGATE_SYSTEM",
    "DELEGATE_USER",
]
