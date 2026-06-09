"""Evolution package — self-evolution engine and safety pipeline."""

from src.evolution.engine import SelfEvolutionEngine
from src.evolution.git_tracker import GitTracker

__all__ = ["SelfEvolutionEngine", "GitTracker"]
