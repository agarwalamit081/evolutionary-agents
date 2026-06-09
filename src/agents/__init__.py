"""Sub-agent system — persistent, evolvable sub-agents.

Sub-agents are dynamically created LangGraph subgraphs that run isolated
tasks on behalf of the main agent. They are persisted to PostgreSQL with
versioning and rolling performance metrics, and optimized by the main
agent's evolution engine over time.

Note: SubAgentRunner and run_parallel are imported lazily to avoid
circular imports (runner → subgraph → graph.nodes → agents).
"""

from src.agents.registry import SubAgentRegistry
from src.agents.persister import SubAgentPersister

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
