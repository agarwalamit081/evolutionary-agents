"""Web search tool — searches the web via DuckDuckGo HTML."""

from __future__ import annotations

import httpx
from loguru import logger


_DDG_URL = "https://html.duckduckgo.com/html/"
_USER_AGENT = "Mozilla/5.0 (compatible; TuringAgent/1.0)"


async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo HTML endpoint.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.

    Returns:
        Formatted search results as text.
    """
    logger.info(f"Web search: {query[:60]}...")

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                _DDG_URL,
                data={"q": query, "b": ""},
                headers={"User-Agent": _USER_AGENT},
            )
            response.raise_for_status()

        # Parse HTML results (basic extraction)
        results = _parse_ddg_results(response.text, max_results)

        if not results:
            return f"No results found for: {query}"

        formatted = "\n".join(
            f"{i+1}. {r['title']}\n   {r['snippet']}\n   URL: {r['url']}"
            for i, r in enumerate(results)
        )
        return formatted

    except httpx.TimeoutException:
        return f"ERROR: Search timed out for query: {query}"
    except httpx.HTTPStatusError as exc:
        return f"ERROR: Search failed with status {exc.response.status_code}"
    except Exception as exc:
        return f"ERROR: Search failed: {exc}"


def _parse_ddg_results(html: str, max_results: int) -> list[dict[str, str]]:
    """Parse DuckDuckGo HTML response into structured results."""
    results: list[dict[str, str]] = []

    # Basic HTML parsing — extract result blocks
    import re

    # Find result containers
    pattern = re.compile(
        r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
        re.DOTALL,
    )

    for match in pattern.finditer(html):
        url = match.group(1)
        title = re.sub(r"<[^>]+>", "", match.group(2)).strip()
        snippet = re.sub(r"<[^>]+>", "", match.group(3)).strip()

        results.append({"url": url, "title": title, "snippet": snippet})
        if len(results) >= max_results:
            break

    return results


TOOL_DEFINITION = {
    "name": "web_search",
    "handler": web_search,
    "description": (
        "Search the web using DuckDuckGo. Returns top results with titles, "
        "snippets, and URLs. Useful for finding current information, documentation, "
        "or answers to factual questions."
    ),
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
