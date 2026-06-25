"""arXiv paper search builtin.

arXiv is the primary preprint server for CS / physics / math research. A
builtin lets the agent search it directly (relevance-ranked) and cite real
papers — instead of fabricating citations or burning one of its 3-per-run
generated-tool slots re-creating an arxiv client every run.

Uses the ``arxiv`` package (``arxiv==4.0.0``, installed + allowlisted). The
dependency is resolved **lazily** (``importlib.util.find_spec``), so a worker
image without ``arxiv`` still imports this module — the call degrades to a
graceful ``ERROR:`` string rather than crashing on import.
``arxiv.Client().results()`` is a blocking HTTP iterator, so it runs off the
event loop via :func:`asyncio.to_thread`.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from typing import Any

from loguru import logger

# Clamp max_results so an over-large request does not page through hundreds of
# arXiv result pages (the client pages ~100 per request over HTTP). 50 is ample
# for grounding a research summary or citation list.
_MAX_RESULTS = 50


def _result_to_dict(result: Any) -> dict[str, Any]:
    """Project an ``arxiv.Result`` onto the documented output shape.

    Fields are defensively coerced (``getattr`` + ``str()``) so a partially-
    populated entry never raises — the call returns the rows it could parse
    rather than failing wholesale.
    """
    authors = getattr(result, "authors", None) or []
    published = getattr(result, "published", None)
    return {
        "title": str(getattr(result, "title", "") or ""),
        "authors": [str(getattr(a, "name", "")) for a in authors],
        "summary": str(getattr(result, "summary", "") or "").strip(),
        "published": published.isoformat() if published is not None else "",
        "pdf_url": str(getattr(result, "pdf_url", "") or ""),
        "entry_id": str(getattr(result, "entry_id", "") or ""),
    }


def _arxiv_search_sync(query: str, max_results: int) -> list[dict[str, Any]]:
    """Blocking arXiv search — run off the event loop via ``asyncio.to_thread``."""
    import arxiv

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    client = arxiv.Client()
    return [_result_to_dict(r) for r in client.results(search)]


async def arxiv_search(query: str = "", max_results: int = 5) -> str:
    """Search arXiv for research papers matching ``query``.

    Args:
        query: arXiv search query (accepts arXiv's fielded syntax, e.g.
            ``"transformer attention"`` or ``cat:cs.CL AND llama``). Whitespace
            is normalized; an empty query is rejected.
        max_results: Maximum papers to return (default 5), clamped to [1, 50].

    Returns:
        A JSON array of ``{title, authors, summary, published, pdf_url,
        entry_id}`` objects (most relevant first), or an ``ERROR:`` string on
        failure / empty query / missing dependency.
    """
    base = " ".join((query or "").split())
    if not base:
        return "ERROR: empty arxiv query"

    n = max(1, min(_MAX_RESULTS, int(max_results)))

    # Lazy dependency check — a worker image without `arxiv` still imports this
    # module; the call degrades to a friendly ERROR instead of crashing.
    if importlib.util.find_spec("arxiv") is None:
        logger.warning("arxiv_search: 'arxiv' package not installed")
        return "ERROR: arxiv package not installed"

    logger.info(f"arxiv_search: query='{base[:60]}' max_results={n}")
    try:
        rows = await asyncio.to_thread(_arxiv_search_sync, base, n)
    except Exception as exc:  # arxiv HTTP / parse / network errors are non-fatal
        logger.warning(f"arxiv_search failed for '{base[:60]}': {exc}")
        return f"ERROR: arxiv search failed: {exc}"

    if not rows:
        return f"No arxiv results found for: {query}"

    return json.dumps(rows, ensure_ascii=False, indent=2)


TOOL_DEFINITION = {
    "name": "arxiv_search",
    "handler": arxiv_search,
    "description": (
        "Search arXiv (the primary CS/physics/math preprint server) for real "
        "research papers by relevance, to ground citations and technical claims "
        "in actual literature rather than memory (which risks fabricated "
        "citations). Returns JSON objects with title, authors, abstract, "
        "publication date, PDF URL, and arXiv id. Use this before citing any "
        "paper, surveying prior work, or answering a 'what does the research "
        "say about X' question. Accepts arXiv's fielded query syntax "
        "(e.g. 'cat:cs.CL AND llama')."
    ),
    # Idempotent read-only network fetch — safe to cache within/across runs.
    "cacheable": True,
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "arXiv search query. Supports arXiv's fielded syntax, e.g. "
                    "'transformer attention mechanism' or 'cat:cs.CL AND llama'."
                ),
            },
            "max_results": {
                "type": "integer",
                "description": (
                    "Maximum number of papers to return (default 5, clamped to 50)."
                ),
                "default": 5,
            },
        },
        "required": ["query"],
    },
}
