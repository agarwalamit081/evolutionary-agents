"""Tool that fetches a URL and returns its main content as clean markdown.

Distinct from ``web_search`` (which returns search snippets + links): this
fetches a *specific* URL and strips HTML/JS boilerplate down to readable text
using ``trafilatura``.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import httpx
import trafilatura
from loguru import logger

from src.config.settings import get_settings
from src.tools.builtin._net_safety import assert_public_host

_USER_AGENT = "Mozilla/5.0 (compatible; TuringAgent/1.0)"
# Download-byte cap and fetch timeout are operator-configurable via
# ToolLimitsSettings (WEB_SCRAPER_MAX_BYTES / WEB_SCRAPER_TIMEOUT). The
# schema display default below mirrors the settings default; enforcement
# reads settings at call-time via _tool_limits().
_SCHEMA_DEFAULT_MAX_CHARS = 8000  # mirrors ToolLimitsSettings.web_scraper_max_chars


def _tool_limits():
    """Call-time accessor — never capture get_settings() at module import."""
    return get_settings().tools


def _extract(url: str, max_chars: int, fetch_timeout: float, max_bytes: int) -> str:
    """Fetch + extract main content (sync; run off the event loop)."""
    err = assert_public_host(url)
    if err:
        return err

    # Fetch with httpx so we enforce our own size cap, user-agent, and timeout
    # (rather than trafilatura's bundled fetcher).
    try:
        with httpx.Client(
            timeout=fetch_timeout,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            with client.stream("GET", url) as resp:
                resp.raise_for_status()
                chunks: list[bytes] = []
                size = 0
                for chunk in resp.iter_bytes():
                    size += len(chunk)
                    if size > max_bytes:
                        return (
                            f"ERROR: Page exceeds {max_bytes // (1024 * 1024)} MB "
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


async def web_scraper(url: str, max_chars: Optional[int] = None) -> str:
    """Fetch a URL and return its main content as clean markdown.

    Args:
        url: Absolute ``http(s)`` URL to scrape.
        max_chars: Maximum characters of extracted text to return. ``None``
            resolves to ``WEB_SCRAPER_MAX_CHARS`` (ToolLimitsSettings, default 8000).

    Returns:
        Clean markdown of the page's main content, or an ``ERROR:`` string.
        Private/loopback URLs are blocked (SSRF guard).
    """
    limits = _tool_limits()
    if max_chars is None:
        max_chars = limits.web_scraper_max_chars
    logger.info(f"web_scraper: {url[:80]}")
    try:
        return await asyncio.to_thread(
            _extract, url, max_chars, limits.web_scraper_timeout, limits.web_scraper_max_bytes
        )
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
                "description": "Maximum characters of extracted text (default: 8000, configurable via WEB_SCRAPER_MAX_CHARS).",
                "default": _SCHEMA_DEFAULT_MAX_CHARS,
            },
        },
        "required": ["url"],
    },
}
