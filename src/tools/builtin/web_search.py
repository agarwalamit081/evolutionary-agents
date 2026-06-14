"""Web search tool — searches the web via the ``ddgs`` (DuckDuckGo) package.

Implements the ``ddgs`` best practices (tmp-code/ddgs-search.md, §6):
  * strict safe-search; ``region`` / ``timelimit`` / ``max_results`` surfaced as
    tool params;
  * query normalization + a per-request politeness delay;
  * tenacity retry/backoff around the auto→lite→html backend fallback chain;
  * post-fetch cleanup: DDG-redirect unwrapping, URL canonicalization + dedup,
    and a spam/low-value-domain blocklist.

Output shape (``N. title / snippet / URL``) is unchanged so callers and tests
that mock the fetcher are unaffected.
"""

from __future__ import annotations

import asyncio
import time
from random import uniform
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)


# ── ddgs defaults (per tmp-code/ddgs-search.md) ──────────────────────────
# A specific region yields cleaner results than the worldwide 'wt-wt' default.
_REGION = "us-en"
# strict filters a large share of low-quality/spammy domains (§6 gap: was moderate).
_SAFESEARCH = "strict"

# Backend fallback chain — if one backend is rate-limited or blocked, pivot to
# the next ('auto' first, then the lighter 'lite'/'html' backends).
_BACKENDS = ("auto", "lite", "html")

# Politeness delay window (seconds) before each ddgs request to avoid IP bans.
_REQUEST_DELAY = (0.2, 0.6)

# Low-value / content-farm domains filtered from results post-fetch. Deliberately
# conservative: social platforms with near-zero technical-research value.
# Quora/Reddit/Medium are intentionally NOT blocked here — they frequently carry
# useful Q&A and technical write-ups. Easy to extend per use case.
_SPAM_DOMAINS = frozenset(
    {
        "pinterest.com",
        "pin.it",
        "facebook.com",
        "fb.com",
        "instagram.com",
        "tiktok.com",
        "9gag.com",
    }
)

# Query-string parameters stripped when canonicalizing a URL for dedup.
_TRACKING_PARAMS = frozenset(
    {
        "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
        "fbclid", "gclid", "ref", "ref_src", "referrer",
    }
)


# ── result cleanup helpers ──────────────────────────────────────────────


def _build_query(query: str) -> str:
    """Normalize the query before sending (§6 query-builder entry point).

    Currently collapses whitespace. This is the extension point for future
    operator injection (``site:``, ``filetype:``, exact-match quoting) — those
    are intentionally NOT applied by default because they drastically narrow
    natural-language queries. Spam domains are filtered post-fetch (robust and
    deterministic) rather than via fragile query-level ``-site:`` exclusion.
    """
    return " ".join(query.split())


def _unwrap_redirect(url: str) -> str:
    """Strip DuckDuckGo's own redirect wrapper (``duckduckgo.com/l/?uddg=``).

    DDG occasionally wraps result URLs in its own redirect with the real URL
    url-encoded in the ``uddg`` query param. Returns the inner URL when present.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return url
    host = (parsed.hostname or "").lower()
    if "duckduckgo.com" in host and parsed.path.startswith("/l/"):
        params = dict(parse_qsl(parsed.query, keep_blank_values=True))
        inner = params.get("uddg")
        if inner:
            return inner
    return url


def _canonicalize_url(url: str) -> str:
    """Normalize a URL for deduplication.

    Lowercases scheme/host, drops the fragment and tracking query params, strips
    a trailing slash on the path, and drops default ports (80/443).
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return url
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    port = parsed.port
    netloc = host if (not port or port in (80, 443)) else f"{host}:{port}"
    kept = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if k.lower() not in _TRACKING_PARAMS
    ]
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((scheme, netloc, path, "", urlencode(kept), ""))


def _is_spam_url(url: str) -> bool:
    """True if the URL's host is (a subdomain of) a blocked spam domain."""
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False
    return any(host == d or host.endswith("." + d) for d in _SPAM_DOMAINS)


def _clean_results(results: list[dict[str, str]]) -> list[dict[str, str]]:
    """Apply redirect-unwrap → canonicalize → spam-filter → dedup.

    Dedup keys on the canonical URL and on a title+body-prefix hash (to also
    collapse syndicated near-duplicates). Order-preserving; returns dicts with
    the legacy ``title``/``href``/``body`` keys.
    """
    seen_urls: set[str] = set()
    seen_content: set[str] = set()
    cleaned: list[dict[str, str]] = []
    for r in results:
        raw_href = _unwrap_redirect((r.get("href") or "").strip())
        canonical = _canonicalize_url(raw_href)
        if _is_spam_url(canonical):
            continue
        if canonical in seen_urls:
            continue
        title = (r.get("title") or "").strip()
        body = (r.get("body") or "").strip()
        content_key = f"{title.lower()}|{body[:120].lower()}"
        if content_key in seen_content:
            continue
        seen_urls.add(canonical)
        seen_content.add(content_key)
        # Output the cleaned (canonical) URL — tracking params and fragments
        # are stripped so callers get tidy links (§6 URL hygiene).
        cleaned.append({"title": title, "href": canonical, "body": body})
    return cleaned


def _log_retry(state: object) -> None:
    """tenacity ``before_sleep`` hook — logs each retry attempt via loguru."""
    outcome = getattr(state, "outcome", None)
    exc = outcome.exception() if outcome is not None and getattr(outcome, "failed", False) else None
    attempt = getattr(state, "attempt_number", "?")
    logger.warning(f"ddgs retry attempt #{attempt} after: {exc}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=0.4, max=2.0),
    retry=retry_if_exception_type((RatelimitException, TimeoutException, DDGSException)),
    before_sleep=_log_retry,
    reraise=True,
)
def _ddgs_text(
    query: str,
    max_results: int,
    region: str,
    safesearch: str,
    timelimit: str,
) -> list[dict[str, str]]:
    """Run a synchronous ddgs text search with backend fallback + retry.

    Returns a list of result dicts (keys: ``title``, ``href``, ``body``).
    Raises a ``DDGSException`` only when every backend has failed across all
    retry attempts. The whole chain is wrapped in tenacity so a transient
    rate-limit/timeout retries before bubbling up (§6).
    """
    # Per-request politeness delay to avoid IP bans (§6). Lives inside the
    # retried function so mocked tests (which replace this name) are unaffected.
    time.sleep(uniform(*_REQUEST_DELAY))

    last_exc: Exception | None = None
    for backend in _BACKENDS:
        try:
            with DDGS() as ddgs:
                text_kwargs: dict[str, object] = {
                    "region": region,
                    "safesearch": safesearch,
                    "backend": backend,
                    "max_results": max_results,
                }
                if timelimit:
                    text_kwargs["timelimit"] = timelimit
                return list(ddgs.text(query, **text_kwargs))
        except (RatelimitException, TimeoutException, DDGSException) as exc:
            last_exc = exc
            logger.warning(f"ddgs backend '{backend}' failed for '{query[:50]}': {exc}")
            continue
    raise DDGSException(f"All ddgs backends failed: {last_exc}")


async def web_search(
    query: str,
    max_results: int = 5,
    region: str = _REGION,
    timelimit: str = "",
) -> str:
    """Search the web using DuckDuckGo (via the ``ddgs`` package).

    Args:
        query: Search query string (normalized before sending).
        max_results: Maximum number of results to return.
        region: Result region (default ``us-en``); ``wt-wt`` for worldwide.
        timelimit: Recency filter — ``d`` (day), ``w`` (week), ``m`` (month),
            ``y`` (year). Empty string disables recency filtering.

    Returns:
        Formatted search results (title / snippet / URL), one per line, after
        spam filtering, DDG-redirect unwrapping, and canonical-URL dedup.
    """
    built_query = _build_query(query)
    if not built_query:
        return "ERROR: empty search query"
    logger.info(f"Web search: {built_query[:60]}...")

    try:
        # ddgs 9.x is synchronous — run it off the event loop so we never block.
        results = await asyncio.to_thread(
            _ddgs_text, built_query, max_results, region, _SAFESEARCH, timelimit
        )
    except DDGSException as exc:
        return f"ERROR: Search failed: {exc}"
    except Exception as exc:
        return f"ERROR: Search failed: {exc}"

    # ddgs result keys are title/href/body. Clean results (strip DDG redirect
    # wrappers, dedup by canonical URL, drop spam domains) before formatting.
    cleaned = _clean_results(results)
    if not cleaned:
        return f"No results found for: {query}"

    formatted = "\n".join(
        f"{i + 1}. {r['title']}\n"
        f"   {r['body']}\n"
        f"   URL: {r['href']}"
        for i, r in enumerate(cleaned)
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
            "region": {
                "type": "string",
                "description": (
                    "Result region, e.g. 'us-en' (default) or 'wt-wt' for "
                    "worldwide."
                ),
                "default": "us-en",
            },
            "timelimit": {
                "type": "string",
                "description": (
                    "Recency filter: 'd' (day), 'w' (week), 'm' (month), "
                    "'y' (year). Omit for no recency filter."
                ),
                "enum": ["", "d", "w", "m", "y"],
                "default": "",
            },
        },
        "required": ["query"],
    },
}
