"""Tool that fetches a URL and returns its main content as clean markdown.

Distinct from ``web_search`` (which returns search snippets + links): this
fetches a *specific* URL and strips HTML/JS boilerplate down to readable text
using ``trafilatura``.
"""

from __future__ import annotations

import asyncio

import httpx
import trafilatura
from loguru import logger

from src.tools.builtin._net_safety import assert_public_host

_MAX_BYTES = 5 * 1024 * 1024  # 5 MB download cap
_FETCH_TIMEOUT = 20.0
_USER_AGENT = "Mozilla/5.0 (compatible; TuringAgent/1.0)"


def _extract(url: str, max_chars: int) -> str:
    """Fetch + extract main content (sync; run off the event loop)."""
    err = assert_public_host(url)
    if err:
        return err

    # Fetch with httpx so we enforce our own size cap, user-agent, and timeout
    # (rather than trafilatura's bundled fetcher).
    try:
        with httpx.Client(
            timeout=_FETCH_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            with client.stream("GET", url) as resp:
                resp.raise_for_status()
                chunks: list[bytes] = []
                size = 0
                for chunk in resp.iter_bytes():
                    size += len(chunk)
                    if size > _MAX_BYTES:
                        return (
                            f"ERROR: Page exceeds {_MAX_BYTES // (1024 * 1024)} MB "
                            f"download cap"
                        )
                    chunks.append(chunk)
                html = b"".join(chunks).decode("utf-8", errors="replace")
    except httpx.HTTPStatusError as exc:
        return f"ERROR: HTTP {exc.response.status_code} fetching {url}"
    except httpx.HTTPError as exc:
        return f"ERROR: Fetch failed: {exc}"

    extracted = trafilatura.extract(
        html, output_format="markdown", include_tables=True, include_links=True
    )
    if not extracted:
        return f"ERROR: Could not extract readable content from {url}"

    if len(extracted) > max_chars:
        extracted = extracted[:max_chars] + f"\n\n... (truncated at {max_chars} chars)"
    return extracted


async def web_scraper(url: str, max_chars: int = 8000) -> str:
    """Fetch a URL and return its main content as clean markdown.

    Args:
        url: Absolute ``http(s)`` URL to scrape.
        max_chars: Maximum characters of extracted text to return (default 8000).

    Returns:
        Clean markdown of the page's main content, or an ``ERROR:`` string.
        Private/loopback URLs are blocked (SSRF guard).
    """
    logger.info(f"web_scraper: {url[:80]}")
    try:
        return await asyncio.to_thread(_extract, url, max_chars)
    except Exception as exc:
        return f"ERROR: Scrape failed: {exc}"


TOOL_DEFINITION = {
    "name": "web_scraper",
    "handler": web_scraper,
    "description": (
        "Fetch a specific URL and return its main content as clean markdown, "
        "stripping navigation, ads, and HTML/JS boilerplate. Use this AFTER "
        "web_search to read a chosen result page, or on any known URL. Only "
        "public http(s) URLs are allowed (private/loopback hosts are blocked)."
    ),
    # Idempotent read-only URL fetch — safe to cache within/across runs.
    "cacheable": True,
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Absolute http(s) URL to fetch and extract.",
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum characters of extracted text (default: 8000).",
                "default": 8000,
            },
        },
        "required": ["url"],
    },
}
