"""Evolve node — triggers self-evolution pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from src.graph.enums import Phase
from src.graph.state import AgentState

if TYPE_CHECKING:
    from src.llm.gateway import LLMGateway


async def evolve_node(
    state: AgentState,
    *,
    gateway: LLMGateway | None = None,
) -> dict[str, Any]:
    """Trigger the self-evolution pipeline.

    When reflection indicates evolution is beneficial, this node
    initiates the mutation analysis via SelfEvolutionEngine.
    Falls back to placeholder recording if the engine is unavailable.

    Args:
        state: Current agent state with reflection results.
        gateway: Optional LLM gateway for mutation generation.

    Returns:
        Partial state update with evolution results.
    """
    reflection = state.get("reflection")
    generation = state.get("generation", 0)
    execution_history = state.get("execution_history", [])

    logger.info(f"Evolution triggered (generation {generation})")

    # Try running the evolution engine
    if gateway is not None:
        result = await _run_evolution_engine(gateway, state, generation)
        if result is not None:
            return result

    # Fallback: record evolution attempt without execution
    evolution_record = {
        "generation": generation,
        "trigger": "reflection_recommended",
        "summary": reflection.summary if reflection else "no reflection",
        "lessons": reflection.lessons_learned if reflection else [],
        "outcome": "skipped_no_gateway",
    }

    logger.info("Evolution skipped: no gateway available")

    return {
        "phase": Phase.STORE_MEMORY,
        "evolution_history": [evolution_record],
        "generation": generation + 1,
    }


async def _run_evolution_engine(
    gateway: LLMGateway,
    state: AgentState,
    generation: int,
) -> dict[str, Any] | None:
    """Run the SelfEvolutionEngine. Returns None on failure."""
    try:
        from src.evolution.engine import SelfEvolutionEngine
        from src.safety.pipeline import SafetyPipeline

        reflection = state.get("reflection")
        execution_history = state.get("execution_history", [])

        safety = SafetyPipeline()
        engine = SelfEvolutionEngine(gateway=gateway, safety_pipeline=safety)

        # Run one evolution cycle
        cycle_result = await engine.run_cycle(
            execution_history=execution_history,
            reflection=reflection,
        )

        evolution_record = {
            "generation": generation,
            "trigger": "reflection_recommended",
            "summary": reflection.summary if reflection else "no reflection",
            "lessons": reflection.lessons_learned if reflection else [],
            "outcome": "completed",
            "mutations_proposed": cycle_result.get("mutations_proposed", 0),
            "mutations_deployed": cycle_result.get("mutations_deployed", 0),
        }

        logger.info(
            f"Evolution cycle complete: "
            f"{evolution_record['mutations_proposed']} proposed, "
            f"{evolution_record['mutations_deployed']} deployed"
        )

        return {
            "phase": Phase.STORE_MEMORY,
            "evolution_history": [evolution_record],
            "generation": generation + 1,
        }
    except Exception as e:
        logger.warning(f"Evolution engine failed: {e}")
        return None
