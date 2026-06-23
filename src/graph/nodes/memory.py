"""Memory node — retrieve and store memories across 3 tiers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from src.graph.enums import Confidence, Phase
from src.graph.state import AgentState, objective_goal_text


def _episode_importance(is_complete: bool, confidence: Confidence | None) -> float:
    """Derive a non-flat episode importance from completion + confidence (A6).

    Episodes were stored with flat 0.5 importance, so cold recall ranked them by
    similarity alone — a hard-won success ranked the same as a dead-end.
    Completion is the dominant signal (0.9 vs 0.4); HIGH/VERY_HIGH confidence
    nudges it up. Clamped to [0,1] for ``cold.store``. A ``None`` confidence
    (heuristic-fallback path) keeps the base value unchanged.
    """
    base = 0.9 if is_complete else 0.4
    if confidence in (Confidence.HIGH, Confidence.VERY_HIGH):
        base += 0.05
    return max(0.0, min(1.0, base))

if TYPE_CHECKING:
    from src.llm.gateway import LLMGateway
    from src.memory.manager import MemoryManager


async def retrieve_memory_node(
    state: AgentState,
    *,
    memory: MemoryManager | None = None,
) -> dict[str, Any]:
    """Retrieve relevant memories for the current goal.

    When a MemoryManager is provided, queries all 3 tiers (Redis hot,
    PostgreSQL warm, pgvector cold) for context relevant to the task.

    Args:
        state: Current agent state.
        memory: Optional MemoryManager for 3-tier memory queries.

    Returns:
        Partial state update with retrieved memories.
    """
    # Key recall on the IMMUTABLE objective (submitted_goal), not the mutable
    # current_goal.text. If current_goal ever drifts (a recalled skill/episode
    # leaking into it), recalling on the drifted text would compound the drift
    # and pull in ever-more-irrelevant context. The anchor keeps recall pure.
    goal_text = objective_goal_text(state)

    logger.info(f"Retrieving memories for: {goal_text[:60]}...")

    retrieved: list[dict[str, Any]] = []
    # Skill ids recalled this run — surfaced to state so store_memory_node can
    # feed the skill-fitness EMA (findings-05 D). Empty when nothing recalled.
    recalled_skill_ids: list[str] = []

    if memory is not None:
        try:
            results = await memory.retrieve_context(query=goal_text, limit=5)
            if results:
                retrieved = [
                    {"content": r.get("content", ""), "tier": r.get("tier", ""), "score": r.get("score", 0.0)}
                    for r in results
                    if isinstance(r, dict) and "content" in r
                ]
                # Fallback: results might be plain objects
                if not retrieved:
                    retrieved = [
                        {"content": str(r)}
                        for r in results[:5]
                    ]
                logger.info(f"Retrieved {len(retrieved)} memories from 3-tier system")
        except Exception as e:
            logger.warning(f"Memory retrieval failed: {e}")

        # Load evolved prompts from warm memory (crystallized by evolution)
        try:
            evolved = await memory.warm.retrieve(
                memory_type="evolved_prompt",
                min_fitness=0.5,
                limit=3,
            )
            for entry in evolved:
                retrieved.append({
                    "content": entry.get("content", ""),
                    "tier": "evolved",
                    "score": entry.get("fitness_score", 0.6),
                })
            if evolved:
                logger.info(f"Loaded {len(evolved)} evolved prompt(s) from warm memory")
        except Exception as e:
            logger.debug(f"Evolved prompt loading skipped: {e}")

        # Recall folded-memory summaries persisted by earlier runs. Each fold
        # stores compact episode/working/tool JSON as warm memory so later
        # runs can reuse compressed context instead of re-deriving it.
        try:
            folded = await memory.warm.retrieve(
                memory_type="folded_memory",
                min_fitness=0.5,
                limit=3,
            )
            for entry in folded:
                retrieved.append({
                    "content": entry.get("content", ""),
                    "tier": "folded",
                    "score": entry.get("fitness_score", 0.5),
                })
            if folded:
                logger.info(
                    f"Loaded {len(folded)} folded memory summary/summaries from warm memory"
                )
        except Exception as e:
            logger.debug(f"Folded memory loading skipped: {e}")

        # Recall durable facts (Phase 5). Facts are entity-ish knowledge mined
        # from prior folds/runs — distinct from skills and from episodic cold
        # memory — and ranked semantically against the goal when possible.
        try:
            facts = await memory.retrieve_facts(query=goal_text, limit=3)
            for entry in facts:
                value = entry.get("value", "")
                key = entry.get("key", "")
                content = f"{key}: {value}" if key else value
                retrieved.append({
                    "content": content,
                    "tier": "fact",
                    "score": entry.get("confidence", 0.5),
                })
            if facts:
                logger.info(f"Loaded {len(facts)} fact(s) from the semantic tier")
        except Exception as e:
            logger.debug(f"Fact recall skipped: {e}")

        # Recall crystallized skills/procedures/workflows, ranked semantically
        # against the goal when possible (findings-05 C). Skills carry real
        # capability embeddings (store() persists them for every type), so this
        # is the recall counterpart to fact recall — ranked by cosine distance,
        # not just fitness. Distinct from facts (procedural HOW vs entity-ish
        # WHAT) and from folded-memory episode summaries.
        try:
            skills = await memory.retrieve_skills(query=goal_text, limit=3)
            for entry in skills:
                skill_id = entry.get("id")
                if skill_id:
                    recalled_skill_ids.append(str(skill_id))
                retrieved.append({
                    "content": entry.get("content", ""),
                    "tier": "skill",
                    "score": entry.get("fitness_score", 0.5),
                })
            if skills:
                logger.info(
                    f"Loaded {len(skills)} skill(s) from the semantic tier"
                )
        except Exception as e:
            logger.debug(f"Skill recall skipped: {e}")

        # A7: recall prior error episodes (the cross-run failure / anti-pattern
        # tier). store_memory_node persists one episode_type='error' cold episode
        # per failed run; this surfaces them ranked by semantic similarity to the
        # objective so planning sees what approaches already failed. Distinct
        # from positive recall (skills/facts = what worked) and best-effort
        # (search_by_query returns [] when no generator is wired).
        try:
            failed = await memory.cold.search_by_query(
                query=goal_text, episode_type="error", limit=2
            )
            for entry in failed:
                retrieved.append({
                    "content": entry.get("content", ""),
                    "tier": "error_episode",
                    "score": entry.get("similarity", 0.5),
                })
            if failed:
                logger.info(f"Loaded {len(failed)} prior error episode(s)")
        except Exception as e:
            logger.debug(f"Error-episode recall skipped: {e}")
    else:
        logger.debug("No MemoryManager available, returning empty memories")

    return {
        "phase": Phase.EXECUTE,
        "retrieved_memories": retrieved,
        "recalled_skill_ids": recalled_skill_ids,
    }


async def store_memory_node(
    state: AgentState,
    *,
    memory: MemoryManager | None = None,
    gateway: LLMGateway | None = None,
) -> dict[str, Any]:
    """Store execution learnings and observations to memory.

    When a MemoryManager is provided, persists observations as hot
    memories and lessons learned as warm memories.

    Args:
        state: Current agent state with reflection and observations.
        memory: Optional MemoryManager for 3-tier memory storage.

    Returns:
        Partial state update confirming storage.
    """
    reflection = state.get("reflection")
    memory_observations = state.get("memory_observations", [])
    is_complete = state.get("is_complete", False)

    observations_count = len(memory_observations)
    lessons_count = len(reflection.lessons_learned) if reflection else 0

    logger.info(
        f"Storing memory: {observations_count} observations, "
        f"{lessons_count} lessons learned"
    )

    if memory is not None:
        stored_count = 0

        # A6: non-flat importance. An episode's weight in cold recall now reflects
        # whether the run completed and how confident it was — not a flat 0.5.
        episode_importance = _episode_importance(
            is_complete, state.get("confidence")
        )

        # Store each observation as hot memory
        for obs in memory_observations:
            try:
                await memory.store_observation(
                    content=obs,
                    importance=episode_importance,
                    tags=["reflection", "complete" if is_complete else "incomplete"],
                )
                stored_count += 1
            except Exception as e:
                logger.warning(f"Failed to store observation: {e}")

        # Store lessons learned as warm memory skills
        if reflection and reflection.lessons_learned:
            try:
                await memory.store_skill(
                    name=f"lesson_{stored_count}",
                    content="; ".join(reflection.lessons_learned),
                    tags=["lesson", str(state.get("current_goal", ""))[:50]],
                )
            except Exception as e:
                logger.warning(f"Failed to store lessons: {e}")

        logger.info(f"Stored {stored_count}/{observations_count} observations to memory")

        # A7: cross-run failure tier. When the run accumulated errors, persist one
        # ``episode_type='error'`` cold episode describing the failed approach so a
        # later run on the same objective can recall "this approach failed last
        # time" (see retrieve_memory_node's error_episode recall). store_memory is
        # terminal, so this writes once per run — no bloat. The most recent error
        # carries the actionable signal. Fixed 0.7 importance — a failure is
        # valuable learning regardless of confidence. Non-fatal (a hiccup never
        # aborts the terminal sink).
        errors = state.get("errors", [])
        if errors:
            try:
                goal_text = objective_goal_text(state)
                reason = str(errors[-1])[:300]
                await memory.store_observation(
                    content=f"Failed approach for {goal_text[:80]}: {reason}",
                    episode_type="error",
                    importance=0.7,
                    tags=["error", goal_text[:50]],
                )
                logger.info(f"Stored error episode (reason: {reason[:60]})")
            except Exception as e:
                logger.warning(f"Failed to store error episode: {e}")

        # Feed the skill-fitness EMA (findings-05 D). Each skill recalled this
        # run (retrieve_memory_node populated recalled_skill_ids) gets
        # success=is_complete — a completed run bumps fitness, an incomplete one
        # decays it — the signal governance fitness-retirement + semantic recall
        # ranking improve on. ``is_complete`` is the durable success signal
        # verify persists (``goal_satisfied`` is verify-local, not in state).
        # Non-fatal (CostTracker-resilience pattern): a hiccup on one skill
        # never aborts the terminal sink; no skill↔tool mapping needed.
        for skill_id in state.get("recalled_skill_ids", []):
            try:
                await memory.update_skill_fitness(str(skill_id), success=is_complete)
            except Exception as e:
                logger.debug(f"Skill fitness update skipped: {e}")

        # Feature E: persist classify's refined_intent (Feature A) as a durable
        # semantic fact so later runs recall the real desired outcome behind a
        # recurring goal. Opt-in (persist_intent_facts, default off); best-effort
        # and non-fatal (CostTracker-resilience pattern — a store hiccup never
        # aborts the terminal sink). Skipped when no refined_intent was surfaced
        # (heuristic classify path) so it costs nothing on default runs.
        try:
            from src.config import get_settings

            if get_settings().agent.persist_intent_facts:
                refined_intent = str(state.get("refined_intent") or "").strip()
                if refined_intent:
                    goal = state.get("current_goal")
                    goal_id = getattr(goal, "id", "") or "unknown"
                    ambiguity_type = str(state.get("ambiguity_type") or "none")
                    await memory.store_fact(
                        key=f"intent::{goal_id}",
                        value=refined_intent,
                        source="classify",
                        tags=["intent", ambiguity_type],
                    )
                    logger.info("Persisted refined_intent as a fact (Feature E)")
        except Exception as e:
            logger.debug(f"Intent-fact persist skipped: {e}")
    else:
        logger.debug("No MemoryManager available, skipping memory storage")

    result: dict[str, Any] = {
        "phase": Phase.COMPLETE if is_complete else Phase.HITL_GATE,
    }

    # Flush accumulated LLM cost/token records into graph state. This node is
    # reached on every terminating path (complete / partial-accepted /
    # evolve→store), so it is the single sink that populates cost_records and
    # total_tokens_used for run-history, eval, and report consumers. No other
    # node writes these fields, so the operator.add reducer sees one append.
    if gateway is not None:
        cost_records = gateway.get_cost_records()
        if cost_records:
            result["cost_records"] = cost_records
            result["total_tokens_used"] = sum(
                r.input_tokens + r.output_tokens for r in cost_records
            )

    return result
