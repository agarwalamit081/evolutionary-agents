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
    _execution_history = state.get("execution_history", [])

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
        from pathlib import Path

        from src.config import get_settings
        from src.evolution.engine import SelfEvolutionEngine
        from src.evolution.git_tracker import GitTracker
        from src.safety.pipeline import SafetyPipeline
        from src.sandbox.executor import SandboxExecutor

        reflection = state.get("reflection")
        execution_history = state.get("execution_history", [])

        settings = get_settings()
        safety = SafetyPipeline()
        engine = SelfEvolutionEngine(gateway=gateway, safety_pipeline=safety)

        # Create sandbox executor
        sandbox: SandboxExecutor | None = None
        try:
            sandbox = SandboxExecutor(settings.evolution)
            await sandbox.ensure_image()
        except Exception as e:
            logger.debug(f"Sandbox executor not available: {e}")
            sandbox = None

        # Create git tracker for shadow repo
        git_tracker: GitTracker | None = None
        try:
            source_dir = Path(getattr(settings.evolution, "evolution_source_dir", "src"))
            repo_dir = Path(getattr(settings.evolution, "evolution_shadow_repo_path", ".turing/evolution-repo"))
            git_tracker = GitTracker(source_dir=source_dir, repo_dir=repo_dir)
            await git_tracker.initialize()
        except Exception as e:
            logger.debug(f"Git tracker not available: {e}")
            git_tracker = None

        # Run one evolution cycle with full pipeline
        cycle_result = await engine.run_cycle(
            execution_history=execution_history,
            reflection=reflection,
            sandbox=sandbox,
            git_tracker=git_tracker,
        )

        # Generate human-readable evolution report
        from src.evolution.report import generate_report

        report = generate_report(
            cycle_result=cycle_result,
            generation=generation,
            trigger="reflection_recommended",
        )
        logger.info(f"\n{report}")

        # Crystallize deployed mutations as warm memory skills
        if cycle_result.get("deployed") and cycle_result.get("proposal"):
            await _crystallize_mutation_skill(cycle_result["proposal"])

        evolution_record = {
            "generation": generation,
            "trigger": "reflection_recommended",
            "summary": reflection.summary if reflection else "no reflection",
            "lessons": reflection.lessons_learned if reflection else [],
            "outcome": cycle_result.get("status", "unknown"),
            "mutations_proposed": cycle_result.get("mutations_proposed", 0),
            "mutations_deployed": cycle_result.get("mutations_deployed", 0),
            "commit_hash": cycle_result.get("deployment", {}).get("commit_hash"),
            "rationale": cycle_result.get("proposal", {}).get("rationale", ""),
            "report": report,
        }

        logger.info(
            f"Evolution cycle complete: "
            f"{evolution_record['mutations_proposed']} proposed, "
            f"{evolution_record['mutations_deployed']} deployed, "
            f"status={evolution_record['outcome']}"
        )

        return {
            "phase": Phase.STORE_MEMORY,
            "evolution_history": [evolution_record],
            "generation": generation + 1,
        }
    except Exception as e:
        logger.warning(f"Evolution engine failed: {e}")
        return None


async def _crystallize_mutation_skill(proposal: dict[str, Any]) -> None:
    """Store a deployed mutation as a warm memory skill for future runs.

    This enables the retrieve_memory_node to load evolved prompts and
    configurations on subsequent agent runs via the existing memory pipeline.
    """
    try:
        from src.memory.warm import WarmMemoryStore

        mutation_type = proposal.get("mutation_type")
        mutated_content = proposal.get("mutated_content", "")
        target_path = proposal.get("target_path") or "evolution/latest_mutation.json"

        if not mutated_content:
            return

        # Determine memory type from mutation type
        if str(mutation_type) == "prompt":
            memory_type = "evolved_prompt"
        elif str(mutation_type) in ("workflow", "tool", "config"):
            memory_type = "evolved_config"
        else:
            memory_type = "evolved_skill"

        # Use warm memory store directly (no Redis/pgvector needed for skills)
        from src.db.session import get_session

        async with get_session() as session:
            warm_store = WarmMemoryStore(session)
            await warm_store.store(
                name=f"evolved_{memory_type}_{proposal.get('priority', 'normal')}",
                content=mutated_content,
                memory_type=memory_type,
                tags=["evolution", str(mutation_type), target_path],
                fitness_score=0.6,
            )

        logger.info(
            f"Crystallized mutation as warm memory skill: "
            f"type={memory_type}, target={target_path}"
        )
    except Exception as e:
        logger.debug(f"Skill crystallization skipped (non-critical): {e}")
