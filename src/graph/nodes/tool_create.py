"""Tool create node — generates, validates, and registers missing tools at runtime.

When reflection detects tool gaps (e.g. the LLM called a tool that doesn't exist),
this node attempts to create the missing tool via LLM code generation, safety
validation, sandbox testing, and dynamic registration.

Flow:
    reflect (detects gap) → tool_create → plan (re-plan with new tool) → execute
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from src.graph.enums import Phase
from src.graph.state import AgentState

if TYPE_CHECKING:
    from src.llm.gateway import LLMGateway
    from src.tools.registry import ToolRegistry


# Bounded retry count for LLM tool-handler generation. Cheaper models
# frequently truncate the generated handler on the first attempt (unclosed
# paren / partial function → AST fail) but succeed once told what broke.
# Feeding the validation error back and regenerating mirrors the evolution
# retry-with-feedback pattern (commit d4c9951). Bounded so a stubborn failure
# degrades gracefully instead of looping. Without this, a single truncated
# handler fails validation and the tool is never registered — breaking
# cross-run persistence+recall for tool-create goals (observed: N5's
# duplicate_finder handler came back at 156 chars, "'(' was never closed").
# Operator-configurable via AgentSettings (TOOL_GEN_MAX_ATTEMPTS); read at
# call-time below.


async def tool_create_node(
    state: AgentState,
    *,
    gateway: LLMGateway | None = None,
    tools: ToolRegistry | None = None,
) -> dict[str, Any]:
    """Generate and register missing tools identified during reflection.

    Iterates over pending_tool_gaps, attempts LLM-based tool generation,
    validates through safety pipeline + sandbox, and registers successful
    tools in the ToolRegistry for immediate use.

    Args:
        state: Current agent state with pending_tool_gaps populated.
        gateway: Optional LLM gateway for code generation.
        tools: Optional ToolRegistry for dynamic registration.

    Returns:
        Partial state update clearing pending gaps and recording created tools.
    """
    pending_gaps = state.get("pending_tool_gaps", [])
    goal = state.get("current_goal")
    tool_results = state.get("tool_results", [])

    # Defense-in-depth: never re-attempt a gap already recorded in
    # attempted_tool_gaps this run. Upstream paths can re-seed the SAME gap into
    # pending_tool_gaps even after a failed attempt — notably agent_spawn's
    # failed-spawn → tool-gap conversion (agent_spawn.py), which writes
    # pending_tool_gaps directly and so bypasses reflect's tool-gap dedup.
    # Without this guard each re-seed re-runs the bounded 3-attempt regeneration
    # loop (_MAX_GENERATION_ATTEMPTS), burning the budget (battery-02 N6: 19
    # node entries, ~56 generations, 764s). attempted_tool_gaps is an
    # operator.add accumulator, so a gap recorded once stays recorded for the
    # whole run — consistent with the "don't retry failed gaps in-run" intent
    # below. The companion root-cause fix is reflect's agent-gap dedup
    # (attempted_agent_gaps); this guard is the backstop.
    already_attempted = state.get("attempted_tool_gaps", [])
    if already_attempted and pending_gaps:
        fresh = [g for g in pending_gaps if g not in already_attempted]
        skipped = len(pending_gaps) - len(fresh)
        if skipped:
            logger.info(
                f"Skipping {skipped} tool gap(s) already attempted this run "
                f"(defense against spawn→tool_create churn)"
            )
        pending_gaps = fresh

    if not pending_gaps or gateway is None or tools is None:
        logger.info(
            f"Tool creation skipped: gaps={len(pending_gaps)}, "
            f"gateway={'yes' if gateway else 'no'}, tools={'yes' if tools else 'no'}"
        )
        return {
            "phase": Phase.EXECUTE,
            "pending_tool_gaps": [],
        }

    logger.info(f"Attempting to create {len(pending_gaps)} missing tool(s)")

    created_tools: list[dict[str, Any]] = []
    failed_gaps: list[str] = []

    for gap_description in pending_gaps:
        result = await _create_single_tool(
            gateway=gateway,
            registry=tools,
            gap_description=gap_description,
            goal_text=goal.text if goal else "",
            tool_results=tool_results,
        )

        if result["success"]:
            created_tools.append(result)
            logger.info(f"Successfully created tool: {result['tool_name']}")
        else:
            failed_gaps.append(gap_description)
            logger.warning(
                f"Failed to create tool for '{gap_description}': "
                f"{result.get('reason', 'unknown')}"
            )

    # Always clear pending_tool_gaps after attempting — failed gaps are logged
    # and should NOT be retried in the same run to prevent infinite loops.
    if failed_gaps:
        logger.info(
            f"Clearing {len(failed_gaps)} failed tool gap(s) to prevent retry loops"
        )

    return {
        "phase": Phase.PLAN if created_tools else Phase.EXECUTE,
        "pending_tool_gaps": [],
        "attempted_tool_gaps": list(pending_gaps),  # record all attempted to prevent re-detection
        "tools_created": created_tools,
    }


async def _create_single_tool(
    gateway: LLMGateway,
    registry: ToolRegistry,
    gap_description: str,
    goal_text: str,
    tool_results: list[Any],
) -> dict[str, Any]:
    """Attempt to create a single tool for a specific capability gap.

    Args:
        gateway: LLM gateway for code generation.
        registry: ToolRegistry to register the tool into.
        gap_description: What capability the tool should provide.
        goal_text: Current goal text for context.
        tool_results: Previous tool results for error context.

    Returns:
        Result dict with 'success' bool and tool details or failure reason.
    """
    try:
        from src.safety.pipeline import SafetyPipeline
        from src.tools.dynamic.generator import ToolGenerator

        safety = SafetyPipeline()

        # ── Semantic dedup (B3) ──────────────────────────────────────────
        # Before spending an LLM generation call, embed the capability gap and
        # reuse an existing tool whose capability is semantically identical
        # (cosine >= capability_dedup_threshold). Only real ("api") embeddings
        # participate — hash-fallback vectors are not semantically meaningful.
        # The reused tool must already be in the in-memory registry (loaded at
        # startup) so the agent can actually call it this run. Best-effort: any
        # failure degrades to generation and never blocks the run.
        from src.memory.embeddings import embed_capability

        gap_embedding, emb_source = await embed_capability(gap_description)
        if gap_embedding is not None and emb_source == "api":
            try:
                from src.config import get_settings
                from src.tools.dynamic.persister import ToolPersister

                threshold = get_settings().agent.capability_dedup_threshold
                similar = await ToolPersister().find_similar(
                    gap_embedding, threshold=threshold
                )
                for cand in similar:
                    if registry.has(cand["tool_name"]):
                        logger.info(
                            f"Reusing existing tool '{cand['tool_name']}' for "
                            f"gap '{gap_description[:60]}' "
                            f"(similarity={cand['similarity']:.3f}) — skipping "
                            f"generation"
                        )
                        return {
                            "success": True,
                            "tool_name": cand["tool_name"],
                            "description": cand["description"],
                            "reused": True,
                            "safety_passed": True,
                            "sandbox_passed": True,
                        }
            except Exception as e:
                logger.debug(f"Tool capability dedup skipped: {e}")

        # Sandbox is optional — best-effort
        sandbox: Any = None
        try:
            from src.config import get_settings
            from src.sandbox.executor import SandboxExecutor

            sandbox = SandboxExecutor(get_settings().evolution)
        except Exception:
            logger.debug("SandboxExecutor not available for tool creation")

        generator = ToolGenerator(
            gateway=gateway,
            safety_pipeline=safety,
            sandbox=sandbox,
        )

        # Gather context about failed tools
        failed_tools: list[str] = []
        error_details: list[str] = []
        for tr in tool_results:
            if hasattr(tr, "error") and tr.error:
                if "Unknown tool" in str(tr.error):
                    failed_tools.append(f"{tr.tool_name}: {tr.error}")
                error_details.append(str(tr.error)[:200])

        context = {
            "goal_text": goal_text,
            "failed_tools": "; ".join(failed_tools[-3:]) or "none",
            "error_details": "; ".join(error_details[-3:]) or "none",
            "existing_tools": registry.list_names(),
        }

        # Bounded regeneration loop: generate → validate → (on failure) feed the
        # specific validation error back and regenerate. A truncated handler is
        # recoverable — the model usually emits a complete function once it sees
        # what broke. ``validate_and_register`` only registers on success, so a
        # failed attempt never pollutes the registry nor counts toward
        # ``max_tools_per_run`` (that counter increments inside register only).
        last_reason = ""
        max_attempts = get_settings().agent.tool_gen_max_attempts
        for attempt in range(1, max_attempts + 1):
            if last_reason:
                context["error_details"] = (
                    f"Previous generation attempt failed validation: "
                    f"{last_reason}. Regenerate the COMPLETE handler_code as "
                    f"valid Python defining exactly one async function — do not "
                    f"truncate or emit a partial function."
                )
                logger.info(
                    f"Retrying tool generation for '{gap_description}' "
                    f"(attempt {attempt}/{max_attempts}) with "
                    f"validation feedback"
                )

            generated = await generator.generate(gap_description, context)
            if generated is None:
                return {
                    "success": False,
                    "reason": "LLM generation failed or returned invalid output",
                    "gap": gap_description,
                }

            result = await generator.validate_and_register(generated, registry)
            if result["success"]:
                # F2: framework-mandated DB persistence (resilient retry). Store
                # the capability embedding (only when a real "api" vector was
                # produced) so future semantically-identical gaps reuse this
                # tool instead of generating a duplicate (B3). ``persisted``
                # flows into tools_created so a silent DB failure is observable
                # downstream rather than masked.
                persisted = await _persist_tool(
                    generated,
                    capability_embedding=gap_embedding
                    if emb_source == "api"
                    else None,
                    capability_text=gap_description if emb_source == "api" else None,
                )
                return {
                    "success": True,
                    "tool_name": generated.tool_name,
                    "description": generated.description,
                    "safety_passed": True,
                    "sandbox_passed": result.get("sandbox_result", {}).get("passed", True),
                    "persisted": persisted,
                }

            last_reason = result.get("reason", "validation failed")
            logger.warning(
                f"Tool generation attempt {attempt}/{max_attempts} "
                f"for '{gap_description}' failed: {last_reason}"
            )

        return {
            "success": False,
            "reason": (
                f"All {max_attempts} generation attempts failed: "
                f"{last_reason}"
            ),
            "gap": gap_description,
        }

    except Exception as e:
        logger.warning(f"Tool creation error for '{gap_description}': {e}")
        return {
            "success": False,
            "reason": str(e),
            "gap": gap_description,
        }


async def _persist_tool(
    generated: Any,
    capability_embedding: list[float] | None = None,
    capability_text: str | None = None,
) -> bool:
    """Framework-mandated DB persistence of a validated tool (F2).

    A successful ``validate_and_register`` must ALWAYS reach the database — a
    tool is only useful this run if it persists for cross-run recall. Persist is
    retried with a fresh session each attempt: ``ToolPersister.persist`` opens
    its own session, so a transient connection error / poisoned session recovers
    on the next call (CostTracker-resilience pattern). Without this a single DB
    blip silently dropped the tool — the run recorded it "created" yet it was
    gone next run, breaking the tool-create persistence+recall contract.

    Args:
        generated: GeneratedTool with handler_code and test_code.
        capability_embedding: Optional capability vector to store (B3 dedup).
        capability_text: The text the embedding was derived from.

    Returns:
        True once a row is written (or already present); False only if every
        attempt fails — in which case the tool stays in-memory for this run and
        a WARNING is logged (observable, never fatal).
    """
    from src.config import get_settings
    from src.tools.dynamic.persister import ToolPersister

    attempts = max(get_settings().agent.tool_persist_max_attempts, 1)
    persister = ToolPersister()
    for attempt in range(1, attempts + 1):
        try:
            row_id = await persister.persist(
                tool_name=generated.tool_name,
                description=generated.description,
                input_schema=generated.input_schema,
                handler_code=generated.handler_code,
                test_code=generated.test_code,
                capability_embedding=capability_embedding,
                capability_text=capability_text,
            )
            if row_id is not None:
                if attempt > 1:
                    logger.info(
                        f"Persisted tool '{generated.tool_name}' to database "
                        f"(on retry attempt {attempt}/{attempts})"
                    )
                else:
                    logger.info(
                        f"Persisted tool '{generated.tool_name}' to database"
                    )
                return True
        except Exception as e:
            # persist() swallows its own errors → None; this covers a raise
            # before its try/except (e.g. an import / wiring fault).
            logger.warning(
                f"Tool persistence attempt {attempt}/{attempts} for "
                f"'{generated.tool_name}' raised: {e}"
            )
        if attempt < attempts:
            logger.warning(
                f"Tool persistence for '{generated.tool_name}' returned no row — "
                f"retrying (attempt {attempt + 1}/{attempts})"
            )

    logger.warning(
        f"Could not persist tool '{generated.tool_name}' after {attempts} "
        f"attempt(s) — tool is in-memory only this run (F2 self-heal exhausted)"
    )
    return False
