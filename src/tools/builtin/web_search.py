"""Web search tool — SearXNG-primary with automatic lightweight-paid fallback.

Architecture (Phase 1 overhaul):
  * SearXNG (service port 8080; host-mapped to 8081) is the primary keyless
    live-search service.
  * On failure / empty / throttle, an ordered chain of lightweight paid providers
    is tried automatically (each only if its API key is set):
    tavily -> serper -> brave -> serpapi -> serpstack -> llmlayer.
  * Heavy providers (firecrawl / apify) stay OFF by default — they engage only
    when the caller passes ``deep_crawl=True`` AND ``DEEP_CRAWL_ENABLED=true``
    AND the provider key is set.
  * Batch mode: ``web_search(queries=[...])`` fans out concurrently under a
    ``SEARCH_BATCH_CONCURRENCY`` semaphore.
  * Post-fetch cleanup is unchanged: redirect-unwrap -> canonicalize ->
    spam-filter -> dedup, with the same ``N. title / snippet / URL`` output
    shape so callers and result-cache keys are unaffected.

All HTTP calls go through tenacity retry on **transient** errors only
(rate-limit / 5xx / timeout / network). Auth / bad-request (4xx) errors never
retry — they mark the provider unavailable and the chain moves on.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from typing import TypeVar
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx
from loguru import logger
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.config.settings import get_settings


# ── ddgs-defaults carried over for the legacy region/safesearch contract ──
# A specific region yields cleaner results than the worldwide 'wt-wt' default.
_REGION = "us-en"


# ── exceptions ──────────────────────────────────────────────────────────
class SearchProviderError(Exception):
    """Base for any search-provider failure handled by the fallback chain."""


class TransientSearchError(SearchProviderError):
    """Transient failure (rate-limit / 5xx / timeout / network) — retried."""


class ProviderUnavailable(SearchProviderError):
    """Non-transient failure (no key / 4xx auth / bad-request) — not retried."""


def _search_settings():
    """Call-time accessor for the SearchSettings group — never capture at import."""
    return get_settings().search


def _tool_limits():
    """Call-time accessor — never capture get_settings() at module import."""
    return get_settings().tools


def _log_retry(state: object) -> None:
    """tenacity ``before_sleep`` hook — logs each retry attempt via loguru."""
    outcome = getattr(state, "outcome", None)
    exc = outcome.exception() if outcome is not None and getattr(outcome, "failed", False) else None
    attempt = getattr(state, "attempt_number", "?")
    logger.warning(f"web_search retry attempt #{attempt} after: {exc}")


_T = TypeVar("_T")


async def _run_with_retry(
    provider: str, coro_factory: Callable[[], Awaitable[_T]]
) -> _T:
    """Run a provider call under tenacity, retrying transient errors only.

    ``coro_factory`` is a zero-arg callable returning a fresh coroutine each
    invocation (re-invoked per retry). The retry attempt count is the operator-
    configurable ``WEB_SEARCH_MAX_ATTEMPTS`` (ToolLimitsSettings), read at call
    time so ``.env`` changes take effect without a restart.
    """
    limits = _tool_limits()
    attempts = max(1, limits.web_search_max_attempts)
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential_jitter(initial=0.4, max=2.0),
        retry=retry_if_exception_type(TransientSearchError),
        before_sleep=_log_retry,
        reraise=True,
    ):
        with attempt:
            return await coro_factory()
    # Unreachable: AsyncRetrying with reraise=True always returns or re-raises
    # from within the loop above. Present so the return-type checker is satisfied.
    raise TransientSearchError(f"{provider}: retry loop exited without a result")


# ── low-value domain blocklist + tracking-param strip (unchanged) ────────
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


# ── result cleanup helpers (unchanged — preserved verbatim) ─────────────


def _build_query(query: str) -> str:
    """Normalize the query before sending.

    Currently collapses whitespace. This is the extension point for future
    operator injection (``site:``, ``filetype:``, exact-match quoting) — those
    are intentionally NOT applied by default because they drastically narrow
    natural-language queries. Spam domains are filtered post-fetch (robust and
    deterministic) rather than via fragile query-level ``-site:`` exclusion.
    """
    return " ".join(query.split())


def _unwrap_redirect(url: str) -> str:
    """Strip a search-engine redirect wrapper (e.g. ``duckduckgo.com/l/?uddg=``).

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
        # are stripped so callers get tidy links.
        cleaned.append({"title": title, "href": canonical, "body": body})
    return cleaned


# ── query-param mapping (legacy ddgs tokens → SearXNG params) ───────────


def _region_to_language(region: str) -> str:
    """Map the legacy ddgs region token to a SearXNG ``language`` code.

    ddgs ``us-en`` (country=us, lang=en) → ``en-US``; ``wt-wt`` (worldwide) →
    ``""`` (no language filter); a BCP47-ish ``en-US`` / ``en`` passes through.
    """
    r = (region or "").strip().lower()
    if not r or r == "wt-wt":
        return ""
    if "-" in r:
        parts = r.split("-", 1)
        # ddgs shape is '<country>-<lang>' (e.g. 'us-en'); reorder to '<lang>-<Country>'.
        if len(parts) == 2 and len(parts[0]) == 2 and len(parts[1]) == 2:
            country, lang = parts
            return f"{lang}-{country.upper()}"
        return r
    return r


def _timelimit_to_time_range(timelimit: str) -> str:
    """Map the legacy ddgs recency token to a SearXNG ``time_range`` value."""
    return {
        "d": "day", "w": "week", "m": "month", "y": "year",
    }.get((timelimit or "").strip().lower(), "")


# ── provider key lookup ─────────────────────────────────────────────────
# Optional third-party integration credentials, read at call time (mirrors how
# litellm reads provider keys from the environment by convention). Not central
# config values — their presence/absence only selects which fallback runs.
_PROVIDER_KEY_ENV = {
    "tavily": "TAVILY_API_KEY",
    "serper": "SERPER_API_KEY",
    "brave": "BRAVE_SEARCH_API_KEY",
    "serpapi": "SERPAPI_API_KEY",
    "serpstack": "SERPSTACK_API_KEY",
    "llmlayer": "LLMLAYER_API_KEY",
    "firecrawl": "FIRECRAWL_API_KEY",
    "apify": "APIFY_API_KEY",
}


def _provider_key(name: str) -> str | None:
    """Return a configured provider key, or None if unset."""
    return os.getenv(_PROVIDER_KEY_ENV.get(name, "")) or None


# ── HTTP helpers (status → exception mapping, DRY across providers) ─────


async def _get_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    provider: str,
    params: dict[str, str | int] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 10.0,
) -> dict[str, object]:
    try:
        resp = await client.get(url, params=params, headers=headers, timeout=timeout)
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        raise TransientSearchError(f"{provider} transport error: {exc}") from exc
    return _parse_json(resp, provider)


async def _post_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    provider: str,
    json_body: dict[str, object],
    headers: dict[str, str] | None = None,
    timeout: float = 10.0,
) -> dict[str, object]:
    try:
        resp = await client.post(url, json=json_body, headers=headers, timeout=timeout)
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        raise TransientSearchError(f"{provider} transport error: {exc}") from exc
    return _parse_json(resp, provider)


def _parse_json(resp: httpx.Response, provider: str) -> dict[str, object]:
    status = resp.status_code
    if status == 429 or 500 <= status < 600:
        raise TransientSearchError(f"{provider} HTTP {status}")
    if 400 <= status < 500:
        # Auth (401/403) or bad-request (400/422) — never retry, skip provider.
        raise ProviderUnavailable(f"{provider} HTTP {status}")
    try:
        return resp.json()
    except Exception as exc:  # malformed JSON — non-transient, skip provider.
        raise ProviderUnavailable(f"{provider} returned non-JSON: {exc}") from exc


def _norm(title: object, href: object, body: object) -> dict[str, str]:
    """Coerce a raw provider field triple to the legacy title/href/body shape."""
    return {
        "title": str(title or ""),
        "href": str(href or ""),
        "body": str(body or ""),
    }


def _rows(data: dict[str, object], *keys: str) -> list[dict[str, object]]:
    """Descend ``data`` through ``keys`` and return the list at that path.

    Each level is isinstance-narrowed so untrusted provider JSON never raises
    on a missing/mistyped field — it just yields an empty list.
    """
    cur: object = data
    for k in keys:
        if not isinstance(cur, dict):
            return []
        cur = cur.get(k)
    return cur if isinstance(cur, list) else []


# ── SearXNG primary fetcher ─────────────────────────────────────────────


async def _searxng_call(
    client: httpx.AsyncClient,
    query: str,
    max_results: int,
    region: str,
    timelimit: str,
) -> list[dict[str, str]]:
    s = _search_settings()
    params: dict[str, str | int] = {
        "q": query,
        "format": "json",
        "safesearch": 2,  # strict (preserves the prior ddgs 'strict' intent)
        "pageno": 1,
    }
    language = _region_to_language(region)
    if language:
        params["language"] = language
    time_range = _timelimit_to_time_range(timelimit)
    if time_range:
        params["time_range"] = time_range
    url = f"{s.searxng_url.rstrip('/')}/search"
    data = await _get_json(
        client, url, provider="searxng", params=params, timeout=s.searxng_timeout
    )
    results: list[dict[str, str]] = []
    for r in _rows(data, "results")[:max_results]:
        results.append(_norm(r.get("title"), r.get("url"), r.get("content")))
    return results


async def _searxng_fetch(
    client: httpx.AsyncClient,
    query: str,
    max_results: int,
    region: str,
    timelimit: str,
) -> list[dict[str, str]]:
    """SearXNG primary search under transient-only retry."""
    return await _run_with_retry(
        "searxng",
        lambda: _searxng_call(client, query, max_results, region, timelimit),
    )


# ── lightweight paid provider adapters (each: keyed → normalized results) ─


async def _tavily_fetch(
    client: httpx.AsyncClient, key: str, query: str, max_results: int,
) -> list[dict[str, str]]:
    data = await _run_with_retry(
        "tavily",
        lambda: _post_json(
            client, "https://api.tavily.com/search", provider="tavily",
            json_body={"api_key": key, "query": query, "max_results": max_results},
        ),
    )
    return [
        _norm(r.get("title"), r.get("url"), r.get("content"))
        for r in _rows(data, "results")[:max_results]
    ]


async def _serper_fetch(
    client: httpx.AsyncClient, key: str, query: str, max_results: int,
) -> list[dict[str, str]]:
    headers = {"X-API-KEY": key, "Content-Type": "application/json"}
    data = await _run_with_retry(
        "serper",
        lambda: _post_json(
            client, "https://google.serper.dev/search", provider="serper",
            json_body={"q": query, "num": max_results}, headers=headers,
        ),
    )
    return [
        _norm(r.get("title"), r.get("link"), r.get("snippet"))
        for r in _rows(data, "organic")[:max_results]
    ]


async def _brave_fetch(
    client: httpx.AsyncClient, key: str, query: str, max_results: int,
) -> list[dict[str, str]]:
    headers = {"X-Subscription-Token": key, "Accept": "application/json"}
    data = await _run_with_retry(
        "brave",
        lambda: _get_json(
            client, "https://api.search.brave.com/res/v1/web/search", provider="brave",
            params={"q": query, "count": max_results}, headers=headers,
        ),
    )
    return [
        _norm(r.get("title"), r.get("url"), r.get("description"))
        for r in _rows(data, "web", "results")[:max_results]
    ]


async def _serpapi_fetch(
    client: httpx.AsyncClient, key: str, query: str, max_results: int,
) -> list[dict[str, str]]:
    data = await _run_with_retry(
        "serpapi",
        lambda: _get_json(
            client, "https://serpapi.com/search", provider="serpapi",
            params={"api_key": key, "q": query, "engine": "google", "num": max_results},
        ),
    )
    return [
        _norm(r.get("title"), r.get("link"), r.get("snippet"))
        for r in _rows(data, "organic_results")[:max_results]
    ]


async def _serpstack_fetch(
    client: httpx.AsyncClient, key: str, query: str, max_results: int,
) -> list[dict[str, str]]:
    data = await _run_with_retry(
        "serpstack",
        lambda: _get_json(
            client, "https://api.serpstack.com/search", provider="serpstack",
            params={"access_key": key, "query": query, "num": max_results},
        ),
    )
    return [
        _norm(r.get("title"), r.get("url"), r.get("snippet") or r.get("description"))
        for r in _rows(data, "organic_results")[:max_results]
    ]


# Heavy provider: Firecrawl (deep-crawl search). Off by default — engages only
# when deep_crawl=True AND DEEP_CRAWL_ENABLED=true AND the key is set.
async def _firecrawl_fetch(
    client: httpx.AsyncClient, key: str, query: str, max_results: int,
) -> list[dict[str, str]]:
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    data = await _run_with_retry(
        "firecrawl",
        lambda: _post_json(
            client, "https://api.firecrawl.dev/v1/search", provider="firecrawl",
            json_body={"query": query, "limit": max_results}, headers=headers,
            timeout=20.0,
        ),
    )
    out: list[dict[str, str]] = []
    for r in _rows(data, "data")[:max_results]:
        raw_meta = r.get("metadata")
        meta: dict[str, object] = raw_meta if isinstance(raw_meta, dict) else {}
        out.append(_norm(
            r.get("title") or meta.get("title"),
            r.get("url"),
            str(r.get("description") or r.get("markdown") or "")[:300],
        ))
    return out


# Registry: provider name → async adapter(client, key, query, max_results, ...).
# Providers keyed in .env but absent here (e.g. llmlayer, apify — no verified
# search endpoint) are skipped with a debug log rather than called against an
# invented endpoint.
PROVIDER_ADAPTERS = {
    "tavily": _tavily_fetch,
    "serper": _serper_fetch,
    "brave": _brave_fetch,
    "serpapi": _serpapi_fetch,
    "serpstack": _serpstack_fetch,
}

# Heavy / deep-crawl providers (separate chain; default-off).
HEAVY_ADAPTERS = {
    "firecrawl": _firecrawl_fetch,
}


# ── fallback orchestrator ───────────────────────────────────────────────


async def _search_with_fallback(
    client: httpx.AsyncClient,
    query: str,
    max_results: int,
    region: str,
    timelimit: str,
    deep_crawl: bool,
) -> list[dict[str, str]]:
    """Try SearXNG, then the (optional heavy, then lightweight) paid chain.

    Returns the first non-empty result set as raw ``{title, href, body}`` dicts.
    A provider with no key, or no registered adapter, is skipped. Returns ``[]``
    when every available source is empty/unavailable — never raises (the caller
    surfaces a friendly 'no results' message).
    """
    # 1. SearXNG primary.
    try:
        results = await _searxng_fetch(client, query, max_results, region, timelimit)
        if results:
            return results
        logger.info(f"web_search: searxng returned 0 results for '{query[:40]}'; trying fallback")
    except SearchProviderError as exc:
        logger.warning(f"web_search: searxng failed for '{query[:40]}': {exc}; trying fallback")

    # 2. Heavy deep-crawl providers (only when explicitly requested + enabled).
    if deep_crawl and _search_settings().deep_crawl_enabled:
        for name in ("firecrawl", "apify"):
            adapter = HEAVY_ADAPTERS.get(name)
            if adapter is None:
                if _provider_key(name):
                    logger.debug(f"web_search: deep-crawl provider '{name}' keyed but no adapter; skip")
                continue
            key = _provider_key(name)
            if not key:
                continue
            try:
                results = await adapter(client, key, query, max_results)
                if results:
                    return results
            except SearchProviderError as exc:
                logger.warning(f"web_search: {name} failed: {exc}")

    # 3. Lightweight paid fallback chain (configured order, keyed + adapter only).
    for name in _search_settings().fallback_providers:
        adapter = PROVIDER_ADAPTERS.get(name)
        if adapter is None:
            if _provider_key(name):
                logger.debug(f"web_search: provider '{name}' keyed but no adapter; skip")
            continue
        key = _provider_key(name)
        if not key:
            continue
        try:
            results = await adapter(client, key, query, max_results)
            if results:
                return results
            logger.info(f"web_search: provider '{name}' returned 0 results; next")
        except SearchProviderError as exc:
            logger.warning(f"web_search: provider '{name}' failed: {exc}; next")

    return []


async def _fetch_results(
    query: str,
    max_results: int,
    region: str,
    timelimit: str,
    deep_crawl: bool,
) -> list[dict[str, str]]:
    """Single-query fetch over a fresh client. The clean test/replace seam."""
    async with httpx.AsyncClient() as client:
        return await _search_with_fallback(
            client, query, max_results, region, timelimit, deep_crawl
        )


async def _fetch_batch(
    queries: list[str],
    max_results: int,
    region: str,
    timelimit: str,
    deep_crawl: bool,
) -> list[list[dict[str, str]]]:
    """Fan out queries concurrently under the SEARCH_BATCH_CONCURRENCY semaphore.

    Each query gets its own client; a transient failure on one query yields an
    empty list for that slot (others are unaffected) — gather never aborts early.
    """
    concurrency = max(1, _search_settings().search_batch_concurrency)
    sem = asyncio.Semaphore(concurrency)

    async def _one(q: str) -> list[dict[str, str]]:
        async with sem:
            try:
                return await _fetch_results(q, max_results, region, timelimit, deep_crawl)
            except Exception as exc:  # keep one bad query from sinking the batch
                logger.warning(f"web_search: batch query '{q[:40]}' failed: {exc}")
                return []

    return await asyncio.gather(*[_one(q) for q in queries])


def _format_results(cleaned: list[dict[str, str]]) -> str:
    return "\n".join(
        f"{i + 1}. {r['title']}\n"
        f"   {r['body']}\n"
        f"   URL: {r['href']}"
        for i, r in enumerate(cleaned)
    )


async def web_search(
    query: str = "",
    max_results: int = 5,
    region: str = _REGION,
    timelimit: str = "",
    queries: list[str] | None = None,
    deep_crawl: bool = False,
) -> str:
    """Search the web via SearXNG (primary) with automatic paid fallback.

    Args:
        query: Single search query (normalized before sending). Ignored when
            ``queries`` is provided.
        max_results: Maximum number of results per query.
        region: Result region (default ``us-en``); ``wt-wt`` for worldwide.
        timelimit: Recency filter — ``d`` (day), ``w`` (week), ``m`` (month),
            ``y`` (year). Empty disables recency filtering.
        queries: Batch mode — a list of queries fanned out concurrently under
            ``SEARCH_BATCH_CONCURRENCY``. When set, ``query`` is ignored.
        deep_crawl: Engage heavy providers (Firecrawl/Apify). No-op unless
            ``DEEP_CRAWL_ENABLED=true`` and the provider key is set.

    Returns:
        Formatted search results (title / snippet / URL), one per line, after
        spam filtering, redirect unwrapping, and canonical-URL dedup. In batch
        mode each query's results are a ``Query: "..."``-prefixed block.
    """
    if queries:
        built = [q for q in (_build_query(str(q)) for q in queries) if q]
        if not built:
            return "ERROR: empty search queries"
        logger.info(f"Web search batch: {len(built)} queries...")
        per_query = await _fetch_batch(built, max_results, region, timelimit, deep_crawl)
        blocks: list[str] = []
        for q, raw in zip(built, per_query):
            cleaned = _clean_results(raw)
            if not cleaned:
                blocks.append(f'Query: "{q}"\nNo results found for: {q}')
            else:
                blocks.append(f'Query: "{q}"\n{_format_results(cleaned)}')
        return "\n\n".join(blocks)

    built_query = _build_query(query)
    if not built_query:
        return "ERROR: empty search query"
    logger.info(f"Web search: {built_query[:60]}...")

    try:
        results = await _fetch_results(
            built_query, max_results, region, timelimit, deep_crawl
        )
    except Exception as exc:
        return f"ERROR: Search failed: {exc}"

    cleaned = _clean_results(results)
    if not cleaned:
        return f"No results found for: {query}"

    return _format_results(cleaned)


TOOL_DEFINITION = {
    "name": "web_search",
    "handler": web_search,
    "description": (
        "Search the web via SearXNG (primary) with automatic paid fallback. "
        "Returns top results with titles, snippets, and URLs. Useful for finding "
        "current information, documentation, or answers to factual questions. "
        "Pass `queries` (a list) to run several searches in parallel, or "
        "`deep_crawl=true` for heavy providers (off by default)."
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
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Batch mode: list of queries run concurrently. When set, "
                    "`query` is ignored."
                ),
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results per query (default: 5).",
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
            "deep_crawl": {
                "type": "boolean",
                "description": (
                    "Engage heavy providers (Firecrawl). Off by default — "
                    "requires DEEP_CRAWL_ENABLED=true and a provider key."
                ),
                "default": False,
            },
        },
        "required": ["query"],
    },
}
