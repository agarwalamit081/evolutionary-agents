"""Tool retrieval-before-selection (findings-05).

Default-OFF capability: instead of injecting EVERY active tool into the
execute/plan prompt, keep the built-in tools (always) plus the top-k
dynamically-created tools whose capability embeddings are semantically nearest
the current goal/step. Reuses the existing ``capability_embedding`` index for
RECALL — before this, ``find_similar`` was the index's only consumer and it
gated dedup, never ranked recall. Falls back to the full set when retrieval is
disabled, the embedding provider is unavailable, or any step errors — so
behavior is unchanged until toggled on (``TOOL_RETRIEVAL_ENABLED``) and a
retrieval hiccup can never starve a run of tools.
"""

from __future__ import annotations

from typing import Any, Protocol

from loguru import logger

from src.config.settings import Settings
from src.memory.embeddings import embed_capability
from src.tools.builtin import ALL_TOOL_DEFINITIONS
from src.tools.dynamic.persister import ToolPersister
from src.tools.registry import ToolRegistry


class ToolRetriever(Protocol):
    """Structural type: anything that can RECALL tool names by capability embedding.

    ``ToolPersister`` satisfies this structurally. Declared as a Protocol so the
    selection layer depends on the *capability* (recall names by vector), not
    the concrete persister — and tests can inject a lightweight fake.
    """

    async def retrieve_tools(self, query_embedding: list[float], top_k: int = 8) -> list[str]:
        ...


# Built-in tools are always available. They are NOT in the capability_embedding
# index (only ``tool_create``/``agent_spawn`` persist embeddings), so a naive
# "retrieve → filter" would silently drop web_search / file_writer /
# code_executor / etc. — the core tools. This set guarantees they survive
# retrieval. ``list_tools(names=...)`` silently skips any built-in not currently
# registered, so an over-broad set is harmless.
_BUILTIN_TOOL_NAMES: frozenset[str] = frozenset(d["name"] for d in ALL_TOOL_DEFINITIONS)


async def select_tools_for_query(
    query: str,
    registry: ToolRegistry,
    settings: Settings,
    persister: ToolRetriever | None = None,
) -> list[dict[str, Any]]:
    """Return the tool set to inject for a goal/step.

    When ``settings.agent.tool_retrieval_enabled`` is False (the default),
    returns the full registered set — identical to pre-retrieval behavior.
    When enabled: built-in tools (always) ∪ the top-k dynamically-created tools
    whose capability embeddings are nearest ``query``.

    The ``persister`` arg is optional purely for tests (inject a fake); the
    production callers omit it and a stateless ``ToolPersister()`` is built on
    demand, mirroring ``find_similar``'s own-session pattern.

    Graceful full-set fallback on: disabled flag, embedding unavailable (the
    hash fallback is not semantically meaningful), empty/failed retrieval, or
    any exception — so retrieval can never break a run.
    """
    agent = settings.agent
    full = registry.list_tools()
    if not agent.tool_retrieval_enabled:
        return full

    try:
        embedding, source = await embed_capability(query)
        # Only real ("api") vectors are semantically rankable; the hash fallback
        # makes unrelated texts dissimilar but carries no meaning, so it cannot
        # rank tools — fall back to the full set rather than ranking on noise.
        if embedding is None or source != "api":
            logger.debug("Tool retrieval skipped this call: no API embedding available")
            return full

        pers = persister or ToolPersister()
        retrieved = await pers.retrieve_tools(embedding, top_k=agent.tool_retrieval_top_k)
        if not retrieved:
            return full

        # Built-ins always + retrieved dynamic tools. list_tools preserves
        # REGISTRY order regardless of the names arg's order, so output order
        # stays stable for the prompt.
        wanted = _BUILTIN_TOOL_NAMES | set(retrieved)
        selected = registry.list_tools(names=list(wanted))
        if not selected:
            return full
        logger.debug(
            f"Tool retrieval selected {len(selected)}/{len(full)} tools "
            f"(built-ins + top-{agent.tool_retrieval_top_k} dynamic by similarity)"
        )
        return selected
    except Exception as e:
        logger.debug(f"Tool retrieval failed, using full tool set: {e}")
        return full
