"""Sub-agent system — persistent, evolvable sub-agents.

Sub-agents are dynamically created LangGraph subgraphs that run isolated
tasks on behalf of the main agent. They are persisted to PostgreSQL with
versioning and rolling performance metrics, and optimized by the main
agent's evolution engine over time.

Note: SubAgentRunner and run_parallel are imported lazily at runtime via
module __getattr__ (PEP 562) to avoid circular imports
(runner → subgraph → graph.nodes → agents). They are also declared under
TYPE_CHECKING so static analyzers agree with __all__.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.agents.registry import SubAgentRegistry
from src.agents.persister import SubAgentPersister

if TYPE_CHECKING:
    # Provided at runtime via __getattr__ below; declared here only so type
    # checkers recognize these as part of the public surface (see __all__).
    from src.agents.runner import SubAgentRunner, run_parallel

__all__ = [
    "SubAgentRegistry",
    "SubAgentRunner",
    "SubAgentPersister",
    "run_parallel",
]


def __getattr__(name: str) -> object:
    """Lazy imports to avoid circular dependency with graph.nodes."""
    if name == "SubAgentRunner":
        from src.agents.runner import SubAgentRunner
        return SubAgentRunner
    if name == "run_parallel":
        from src.agents.runner import run_parallel
        return run_parallel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
