"""research node — bounded multi-hop research loop (Phase 5a; default-off).

Runs once between ``retrieve_memory`` and ``structure_analysis`` (only when the
master switch ``research_loop_enabled`` is on), and gathers external grounding
for the goal through a retrieve→refine loop:

  hop 1 query = the refined intent (or the goal text)
  each hop:
    1. retrieve — query the first available retrieval tool
       (``web_search`` → ``corpus_search`` → ``arxiv_search``) for top-K results
    2. refine — one LLM call distils the fresh evidence into concise findings and
       decides whether the evidence is now ``sufficient`` or emits one sharper
       ``next_query`` for the biggest remaining gap
  the loop stops at ``research_max_hops`` hops, when the refine step marks the
  evidence sufficient, when it emits no/empty/duplicate next query, or when a
  hop's LLM call fails.

The distilled findings are carried forward as ADVISORY ``research_context``
(rendered into planning on re-plan + surfaced alongside retrieved memory); the
literal goal is NEVER rewritten — it stays the OBJECTIVE. ``research_node`` never
mutates ``current_goal``.

Loop safety: ``research_done`` is a single-shot guard (mirror
``structure_analysis_done``). Every hop is wrapped so a tool/gateway failure
degrades to stopping with the evidence gathered so far — a hiccup never aborts
the run (CostTracker-resilience pattern). ``asyncio.TimeoutError`` from the
gateway is non-retriable (the gateway already does not retry it), so it is
caught here and the loop stops rather than amplifying.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from src.config import get_settings
from src.graph.enums import Phase, TaskComplexity
from src.graph.state import AgentState, objective_goal_text

if TYPE_CHECKING:
    from src.llm.gateway import LLMGateway
    from src.tools.registry import ToolRegistry


# Cap the raw evidence kept from a single retrieval hop so one verbose search
# result cannot blow the refine prompt's context (the refine step distils it).
_HOP_EVIDENCE_CHAR_LIMIT = 4000
# Cap the seed query derived from the goal/intent — a search query, not prose.
_INITIAL_QUERY_CHAR_LIMIT = 200


async def research_node(
    state: AgentState,
    *,
    gateway: LLMGateway | None = None,
    tools: ToolRegistry | None = None,
) -> dict[str, Any]:
    """Run a bounded multi-hop retrieve→refine research loop.

    Args:
        state: Current agent state (reads the immutable objective +
            ``refined_intent``; the ``research_done`` guard). Writes the
            ``research_done`` guard + the advisory ``research_context``.
        gateway: Optional LLM gateway for the refine step. Without it the node
            degrades to a no-op (carries no research context) — it never breaks
            the run.
        tools: Optional ToolRegistry for retrieval. Without a loaded retrieval
            tool every hop's evidence is empty, so the loop still runs its refine
            steps but gathers nothing useful.

    Returns:
        Partial state update. Always sets ``research_done=True`` and the
        ``RESEARCH`` phase transition; includes the assembled ``research_context``
        (empty string when nothing was gathered).
    """
    # Single-shot guard — never re-run (prevents retrieve_memory↔research cycles).
    if state.get("research_done"):
        return {"phase": Phase.RESEARCH, "research_done": True}

    # No gateway → no refine step → nothing to loop over. Carry no context.
    if gateway is None:
        logger.debug("Research: no gateway; skipping research loop")
        return {"phase": Phase.RESEARCH, "research_done": True, "research_context": ""}

    settings = get_settings().agent
    goal_text = objective_goal_text(state)
    refined_intent = str(state.get("refined_intent", "") or "").strip()

    # Seed query: prefer the refined intent (the real desired outcome), else the
    # goal text. An empty goal means there is nothing to research.
    query = (refined_intent or goal_text).strip()[:_INITIAL_QUERY_CHAR_LIMIT]
    if not query:
        return {"phase": Phase.RESEARCH, "research_done": True, "research_context": ""}

    max_hops = settings.research_max_hops
    top_k = settings.research_top_k

    prior_queries: list[str] = []
    findings: list[str] = []

    hop = 0
    while hop < max_hops and query:
        hop += 1
        prior_queries.append(query)
        evidence = await _retrieve(tools, query, top_k)

        decision = await _refine(
            gateway,
            goal_text=goal_text,
            refined_intent=refined_intent,
            hop=hop,
            max_hops=max_hops,
            prior_queries=prior_queries,
            findings=findings,
            query=query,
            evidence=evidence,
            max_tokens=settings.research_max_tokens,
        )
        if decision is None:
            # Refine failure (gateway/parse) → stop with what we have so far.
            logger.debug(f"Research: refine returned nothing at hop {hop}; stopping")
            break

        if decision.findings:
            findings.extend(decision.findings)

        if decision.sufficient:
            logger.info(f"Research: sufficient at hop {hop}/{max_hops}")
            break

        next_query = (decision.next_query or "").strip()
        if not next_query:
            logger.info(f"Research: no next query at hop {hop}/{max_hops}; stopping")
            break
        if next_query in prior_queries:
            logger.info(f"Research: next query already run at hop {hop}; stopping")
            break
        query = next_query

    logger.info(
        f"Research: {hop} hop(s) run, {len(findings)} finding(s) gathered, "
        f"{len(prior_queries)} query/queries"
    )

    return {
        "phase": Phase.RESEARCH,
        "research_done": True,
        "research_context": _assemble(prior_queries, findings),
    }


# ── Loop helpers ────────────────────────────────────────────────────────


async def _retrieve(
    tools: ToolRegistry | None,
    query: str,
    top_k: int,
) -> str:
    """Query the first available retrieval tool; return its (capped) text.

    Tries ``web_search`` → ``corpus_search`` → ``arxiv_search`` in preference
    order and returns the first non-empty result. Best-effort: a tool failure or
    an unavailable registry yields empty evidence, never an exception.
    """
    if tools is None:
        return ""
    # Each tool takes a slightly different kwarg shape; dispatch per-tool.
    candidates: list[tuple[str, dict[str, Any]]] = [
        ("web_search", {"queries": [query], "max_results": top_k}),
        ("corpus_search", {"query": query, "top_k": top_k}),
        ("arxiv_search", {"query": query, "max_results": top_k}),
    ]
    for name, kwargs in candidates:
        handler = tools.get_handler(name)
        if handler is None:
            continue
        try:
            out = await handler(**kwargs)
            text = str(out).strip()
            if text:
                return text[:_HOP_EVIDENCE_CHAR_LIMIT]
        except Exception as exc:  # noqa: BLE001 — grounding is opportunistic
            logger.debug(f"Research: {name} failed for '{query[:50]}': {exc}")
            continue
    return ""


async def _refine(
    gateway: LLMGateway,
    *,
    goal_text: str,
    refined_intent: str,
    hop: int,
    max_hops: int,
    prior_queries: list[str],
    findings: list[str],
    query: str,
    evidence: str,
    max_tokens: int,
) -> Any:
    """One LLM refine-or-stop call. Returns the ``ResearchRefine`` or ``None``.

    Returns ``None`` on any failure (gateway error, parse failure, timeout) so
    the loop stops gracefully — a refine hiccup never aborts the run.
    """
    try:
        from src.graph.prompts import RESEARCH_SYSTEM, RESEARCH_USER
        from src.graph.schemas import ResearchRefine
        from src.llm.structured_output import StructuredOutputManager

        user_prompt = RESEARCH_USER.format(
            goal_text=goal_text,
            refined_intent=refined_intent or "(none surfaced)",
            hop_index=hop,
            max_hops=max_hops,
            prior_queries="\n".join(f"- {q}" for q in prior_queries) or "- (none)",
            accumulated_findings="\n".join(f"- {f}" for f in findings)
            or "- (none yet)",
            current_query=query,
            current_evidence=evidence or "(no results returned)",
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": str(RESEARCH_SYSTEM)},
            {"role": "user", "content": user_prompt},
        ]
        response = await gateway.acompletion(
            messages=messages,
            node="research",
            complexity=TaskComplexity.SIMPLE,
            max_tokens=max_tokens,
        )
        return await StructuredOutputManager().extract(
            response.content, ResearchRefine, gateway=gateway, messages=messages
        )
    except Exception as exc:  # noqa: BLE001 — a refine failure degrades, never aborts
        logger.debug(f"Research refine failed at hop {hop}: {exc}")
        return None


def _assemble(
    prior_queries: list[str],
    findings: list[str],
) -> str:
    """Assemble the advisory ``research_context`` markdown (empty when no findings).

    The context is advisory grounding only — it explains the goal's domain, never
    rewrites it. Findings are the model's distilled facts (not raw snippets) so
    the downstream prompt stays compact.
    """
    if not findings:
        return ""
    lines: list[str] = [
        "RESEARCH CONTEXT — multi-hop evidence gathered before planning. "
        "Treat as ADVISORY grounding; verify any specific claim before relying "
        "on it. The literal goal above remains what you must deliver.",
        "Queries run:",
    ]
    lines += [f"- {q}" for q in prior_queries]
    lines.append("Findings:")
    lines += [f"- {f}" for f in findings]
    return "\n".join(lines)
