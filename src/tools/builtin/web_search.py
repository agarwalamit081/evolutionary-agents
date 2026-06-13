"""Web search tool — searches the web via the ``ddgs`` (DuckDuckGo) package.

Replaces the previous hand-rolled DuckDuckGo HTML regex scraper with the
maintained ``ddgs`` client, adding backend fallback and structured exception
handling. Output shape (``N. title / snippet / URL``) is unchanged so callers
and tests are unaffected.
"""

from __future__ import annotations

import asyncio

from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException
from loguru import logger


# Best-practice query tuning (per tmp-code/ddgs-search.md): a specific region
# yields cleaner results than the worldwide 'wt-wt' default, and moderate
# safe-search filters low-quality domains.
_REGION = "us-en"
_SAFESEARCH = "moderate"

# Backend fallback chain — if one backend is rate-limited or blocked, pivot to
# the next ('auto' first, then the lighter 'lite'/'html' backends).
_BACKENDS = ("auto", "lite", "html")


def _ddgs_text(query: str, max_results: int) -> list[dict[str, str]]:
    """Run a synchronous ddgs text search with backend fallback.

    Returns a list of result dicts (keys: ``title``, ``href``, ``body``).
    Raises a ``DDGSException`` only when every backend has failed.
    """
    last_exc: Exception | None = None
    for backend in _BACKENDS:
        try:
            with DDGS() as ddgs:
                return list(
                    ddgs.text(
                        query,
                        region=_REGION,
                        safesearch=_SAFESEARCH,
                        backend=backend,
                        max_results=max_results,
                    )
                )
        except (RatelimitException, TimeoutException, DDGSException) as exc:
            last_exc = exc
            logger.warning(f"ddgs backend '{backend}' failed for '{query[:50]}': {exc}")
            continue
    raise DDGSException(f"All ddgs backends failed: {last_exc}")


async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo (via the ``ddgs`` package).

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.

    Returns:
        Formatted search results (title / snippet / URL), one per line.
    """
    logger.info(f"Web search: {query[:60]}...")

    try:
        # ddgs 9.x is synchronous — run it off the event loop so we never block.
        results = await asyncio.to_thread(_ddgs_text, query, max_results)
    except DDGSException as exc:
        return f"ERROR: Search failed: {exc}"
    except Exception as exc:
        return f"ERROR: Search failed: {exc}"

    if not results:
        return f"No results found for: {query}"

    # ddgs result keys are title/href/body; map to the legacy output shape
    # (title / snippet / URL) for backward compatibility.
    formatted = "\n".join(
        f"{i + 1}. {r.get('title', '')}\n"
        f"   {r.get('body', '')}\n"
        f"   URL: {r.get('href', '')}"
        for i, r in enumerate(results)
    )
    return formatted


TOOL_DEFINITION = {
    "name": "web_search",
    "handler": web_search,
    "description": (
        "Search the web using DuckDuckGo. Returns top results with titles, "
        "snippets, and URLs. Useful for finding current information, documentation, "
        "or answers to factual questions."
    ),
    # Idempotent read-only network fetch — safe to cache within/across runs.
    "cacheable": True,
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (default: 5).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}
