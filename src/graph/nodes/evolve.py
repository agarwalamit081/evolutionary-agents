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
    tools: Any | None = None,
    sub_agent_registry: Any | None = None,
) -> dict[str, Any]:
    """Trigger the self-evolution pipeline.

    When reflection indicates evolution is beneficial, this node
    initiates the mutation analysis via SelfEvolutionEngine.
    Falls back to placeholder recording if the engine is unavailable.

    Args:
        state: Current agent state with reflection results.
        gateway: Optional LLM gateway for mutation generation.
        tools: Optional ToolRegistry — wired into the Phase-8 promotion canary
            (a golden benchmark needs the live tool registry) when promotion is
            opted in. ``None`` → no canary → no promotion (safe).
        sub_agent_registry: Optional SubAgentRegistry — likewise for the canary.

    Returns:
        Partial state update with evolution results.
    """
    reflection = state.get("reflection")
    generation = state.get("generation", 0)
    _execution_history = state.get("execution_history", [])

    logger.info(f"Evolution triggered (generation {generation})")

    # Try running the evolution engine
    if gateway is not None:
        result = await _run_evolution_engine(
            gateway, state, generation, tools=tools, sub_agent_registry=sub_agent_registry
        )
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
    *,
    tools: Any | None = None,
    sub_agent_registry: Any | None = None,
) -> dict[str, Any] | None:
    """Run the SelfEvolutionEngine. Returns None on failure."""
    try:
        from pathlib import Path

        from src.config import get_settings
        from src.evolution.engine import SelfEvolutionEngine
        from src.evolution.git_tracker import GitTracker
        from src.evolution.persister import EvolutionPersister
        from src.safety.pipeline import SafetyPipeline
        from src.sandbox.executor import SandboxExecutor

        reflection = state.get("reflection")
        execution_history = state.get("execution_history", [])

        settings = get_settings()
        safety = SafetyPipeline()
        # Construct the persister unconditionally (no I/O until a method is
        # called); the engine swallows every persister exception, so a DB outage
        # can never abort an evolution cycle.
        persister = EvolutionPersister()
        engine = SelfEvolutionEngine(
            gateway=gateway,
            safety_pipeline=safety,
            persister=persister,
            max_retries=getattr(settings.evolution, "max_evolution_retries", 2),
        )

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

        # Phase 8: build the promotion gate when live promotion is opted in AND
        # the canary has its deps (gateway is present here; tools/registry needed
        # for the golden benchmark). Missing deps → no gate → no promotion (the
        # engine logs it; promotion is never required for a healthy cycle).
        promotion_gate: Any | None = None
        if (
            getattr(settings.evolution, "evolution_promote_to_live", False)
            and tools is not None
            and sub_agent_registry is not None
        ):
            try:
                from src.evolution.promote import GoldenCanary, PromotionGate

                canary = GoldenCanary(gateway, tools, sub_agent_registry)
                promotion_gate = PromotionGate(canary=canary.score, settings=settings)
            except Exception as e:
                logger.debug(f"Promotion gate not wired: {e}")
                promotion_gate = None

        # Run one evolution cycle with full pipeline
        cycle_result = await engine.run_cycle(
            execution_history=execution_history,
            reflection=reflection,
            sandbox=sandbox,
            git_tracker=git_tracker,
            promotion_gate=promotion_gate,
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

        proposal = cycle_result.get("proposal") or {}
        raw_mutation_type = proposal.get("mutation_type")
        # MutationType is a (str, Enum); prefer ``.value`` ("tool") over the enum
        # repr, but tolerate a plain-string mutation_type already in the proposal.
        # ``getattr`` is None-safe so this never trips Optional-member access.
        mutation_type_str: str | None
        if raw_mutation_type is None:
            mutation_type_str = None
        else:
            value = getattr(raw_mutation_type, "value", None)
            mutation_type_str = str(value) if value is not None else str(raw_mutation_type)
        evolution_record = {
            "generation": generation,
            "trigger": "reflection_recommended",
            "summary": reflection.summary if reflection else "no reflection",
            "lessons": reflection.lessons_learned if reflection else [],
            "outcome": cycle_result.get("status", "unknown"),
            "mutation_type": mutation_type_str,
            "mutations_proposed": cycle_result.get("mutations_proposed", 0),
            "mutations_deployed": cycle_result.get("mutations_deployed", 0),
            "commit_hash": cycle_result.get("deployment", {}).get("commit_hash"),
            "rationale": proposal.get("rationale", ""),
            "report": report,
            "promotion": cycle_result.get("promotion", {}),
        }

        logger.info(
            f"Evolution cycle complete: "
            f"{evolution_record['mutations_proposed']} proposed, "
            f"{evolution_record['mutations_deployed']} deployed, "
            f"status={evolution_record['outcome']}"
        )

        # Phase 4 E — evolve→execute for deployed TOOL mutations (opt-in, default
        # off; settings.evolution.evolution_reexecute_tool). After a TOOL
        # mutation deploys (it has already passed the engine's safety + sandbox +
        # post-deploy smoke gates), live-register its handler in the ToolRegistry
        # and signal route_after_evolve to run ONE execute pass so the new tool is
        # reachable in-run. Fail-closed: any hiccup → no re-execute (falls through
        # to store_memory). The gate is TOOL-specific: PROMPT/CODE mutations and
        # config-JSON TOOL templates (target_path not ending in .py) never
        # re-execute. ``evolve_reexecute_done`` bounds re-execution to once per
        # run (monotonic-True; a second evolve cycle never re-offers).
        reexecute_offered = False
        reexecute_done = bool(state.get("evolve_reexecute_done", False))
        if (
            getattr(settings.evolution, "evolution_reexecute_tool", False)
            and tools is not None
            and cycle_result.get("deployed")
            and mutation_type_str == "tool"
            and (proposal.get("target_path") or "").endswith(".py")
            and not reexecute_done
        ):
            registered = await _try_register_deployed_tool(proposal, tools, gateway)
            reexecute_offered = registered
            if registered:
                reexecute_done = True
                evolution_record["reexecute_registered_tool"] = proposal.get("target_path")

        return {
            "phase": Phase.STORE_MEMORY,
            "evolution_history": [evolution_record],
            "generation": generation + 1,
            "evolve_reexecute_offered": reexecute_offered,
            "evolve_reexecute_done": reexecute_done,
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


def _sanitize_tool_name(raw: str) -> str:
    """Coerce a deployed-module stem into a valid dynamic-tool name.

    The generator's name rule is ``replace("_", "").isalnum()``; this collapses
    any non-alphanumeric run to ``_`` and lower-cases. Falls back to a stable
    default and guarantees a leading alpha so the name is a legal identifier.
    """
    import re

    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", raw).strip("_").lower()
    if not cleaned:
        cleaned = "evolved_tool"
    if not (cleaned[0].isalpha() or cleaned[0] == "_"):
        cleaned = f"t_{cleaned}"
    return cleaned


def _derive_input_schema(handler_code: str) -> dict[str, Any]:
    """Best-effort JSON Schema for a materialized handler's parameters.

    AST-parses ``handler_code`` and describes the first async def's positional
    parameters as loose string fields (the LLM-facing schema is intentionally
    permissive; the handler validates its own inputs). Falls back to an empty
    object schema when the signature can't be read. ``self``/``cls`` are skipped,
    and parameters with defaults are excluded from ``required`` (the last
    ``len(defaults)`` args carry defaults).
    """
    import ast

    schema: dict[str, Any] = {"type": "object", "properties": {}}
    try:
        tree = ast.parse(handler_code)
    except SyntaxError:
        return schema
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            args = node.args.args
            n_defaults = len(node.args.defaults)
            # The first (len(args) - n_defaults) args have no default → required.
            first_optional = len(args) - n_defaults
            properties: dict[str, Any] = {}
            required: list[str] = []
            for idx, arg in enumerate(args):
                name = arg.arg
                if name in ("self", "cls"):
                    continue
                properties[name] = {"type": "string"}
                if idx < first_optional:
                    required.append(name)
            schema = {
                "type": "object",
                "properties": properties,
                "required": required,
            }
            break
    return schema


async def _try_register_deployed_tool(
    proposal: dict[str, Any],
    registry: Any,
    gateway: Any,
) -> bool:
    """Live-register a deployed TOOL mutation's handler (Phase 4 E1).

    The mutation has already passed the engine's safety + sandbox + post-deploy
    smoke gates (``cycle_result["deployed"]`` is the smoke-verified effective
    deploy). This re-runs the dynamic-tool double-barrier (``SafetyPipeline`` +
    ``_materialize_handler``) for defense-in-depth and makes the tool callable in
    the live ``ToolRegistry`` for the current run, then best-effort persists it
    to ``tool_registrations`` so future runs load it via ``load_active_tools``.

    Fail-closed: ANY error or validation failure → ``False`` (the caller does not
    signal re-execution). The mutation reaches this point only when
    ``mutation_type == "tool"`` and ``target_path`` ends in ``.py`` (the runnable
    LLM-gen path), so config-JSON TOOL templates are excluded upstream.
    """
    try:
        from pathlib import Path

        from src.safety.pipeline import SafetyPipeline
        from src.tools.dynamic.generator import GeneratedTool, ToolGenerator

        target = proposal.get("target_path") or ""
        handler_code = proposal.get("mutated_content", "")
        if not handler_code or not target.endswith(".py"):
            return False

        tool_name = _sanitize_tool_name(Path(target).stem)
        description = (
            proposal.get("description")
            or f"Evolved tool {tool_name} deployed by self-evolution"
        )
        gen_tool = GeneratedTool(
            tool_name=tool_name,
            description=str(description),
            input_schema=_derive_input_schema(handler_code),
            handler_code=handler_code,
            test_code="",
        )

        generator = ToolGenerator(
            gateway=gateway,
            safety_pipeline=SafetyPipeline(),
            sandbox=None,
        )
        result = await generator.validate_and_register(gen_tool, registry)
        if not result.get("success"):
            logger.warning(
                f"Deployed TOOL mutation '{tool_name}' failed live registration: "
                f"{result.get('reason')}; not re-executing"
            )
            return False

        # Best-effort DB persistence so load_active_tools materializes this tool
        # on future runs. Non-fatal — the live registry already has it for this
        # run, so a persistence failure never blocks re-execution.
        await _persist_deployed_tool(
            tool_name, str(description), gen_tool.input_schema, handler_code
        )

        logger.info(
            f"Deployed TOOL mutation live-registered as '{tool_name}'; "
            f"signaling one execute re-execution pass"
        )
        return True
    except Exception as e:
        logger.warning(
            f"Deployed TOOL live-registration failed; not re-executing: {e}"
        )
        return False


async def _persist_deployed_tool(
    tool_name: str,
    description: str,
    input_schema: dict[str, Any],
    handler_code: str,
) -> None:
    """Best-effort persist of a live-registered evolved tool to the DB.

    Computes a capability embedding via the shared ``embed_capability`` path and
    stores it ONLY when the source is ``"api"`` (hash fallbacks are not
    semantically meaningful — matching the dedup contract in
    ``embed_capability``). Any failure is logged at DEBUG and swallowed so the
    live registration (the correctness-critical part) always wins.
    """
    try:
        from src.tools.dynamic.persister import ToolPersister

        capability_text = f"{tool_name}: {description}"
        embedding: list[float] | None = None
        try:
            from src.memory.embeddings import embed_capability

            vector, source = await embed_capability(capability_text)
            embedding = vector if source == "api" else None
        except Exception as embed_exc:
            logger.debug(f"Deployed tool capability embedding skipped: {embed_exc}")
            embedding = None

        await ToolPersister().persist(
            tool_name=tool_name,
            description=description,
            input_schema=input_schema,
            handler_code=handler_code,
            test_code="",
            capability_embedding=embedding,
            capability_text=capability_text,
        )
    except Exception as e:
        logger.debug(f"Deployed tool DB persistence skipped: {e}")
