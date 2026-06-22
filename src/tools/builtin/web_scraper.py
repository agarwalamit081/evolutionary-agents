"""Tool that fetches a URL and returns its main content as clean markdown.

Distinct from ``web_search`` (which returns search snippets + links): this
fetches a *specific* URL, strips HTML/JS boilerplate, and exposes the
AI-format extraction layer (Phase 1): structured metadata, content-hash
dedup key, and overlapping chunking for LLM/corpus indexing.

Extraction stack:
  * ``trafilatura`` for main-content extraction (markdown) + metadata
    (``bare_extraction``: title/description/sitename/author/date/hostname).
  * ``markdownify`` as a fallback for pages trafilatura can't parse.
  * HTTP client is ``httpx``; on an anti-bot signal (HTTP 403/429) the fetch
    retries once via ``curl_cffi`` TLS impersonation (Chrome JA3) — the
    Cloudflare/bot-WAF bypass tier (Gap 7). Both deps are OPTIONAL and imported
    defensively so a missing one degrades gracefully (markdownify →
    trafilatura-only; curl_cffi → httpx-only) and never crashes graph build.

Public surface reused by ``corpus.py``: ``extract_page`` (structured),
``chunk_text`` (overlapping chunks), ``compute_content_hash`` (dedup key).
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
import tiktoken
import trafilatura
from loguru import logger

# markdownify is an OPTIONAL fallback (HTML→markdown for pages trafilatura can't
# parse). It is imported DEFENSIVELY so a missing optional extraction lib NEVER
# crashes graph build: an eager top-level ``from markdownify import ...`` made
# the entire agent fail to start on EVERY goal whenever the dep was absent (Bug
# A — the worker's run executor died at iter 0 with ModuleNotFoundError before
# any node ran). When absent, extraction degrades to trafilatura-only
# (see ``_extract_markdown``).
try:
    from markdownify import markdownify as _markdownify

    _MARKDOWNIFY_AVAILABLE = True
except ModuleNotFoundError:  # optional fallback dep not installed
    _markdownify = None  # type: ignore[assignment]
    _MARKDOWNIFY_AVAILABLE = False

# curl_cffi is the OPTIONAL anti-bot tier (Gap 7): on a 403/429 from httpx, the
# fetch is retried once with a Chrome-impersonated TLS session (JA3), bypassing
# Cloudflare/bot-WAF TLS blocking that httpx cannot. Imported defensively so a
# missing curl_cffi degrades to httpx-only and never crashes graph build.
try:
    from curl_cffi import requests as _curl_cffi_requests

    _CURL_CFFI_AVAILABLE = True
except ModuleNotFoundError:  # optional anti-bot dep not installed
    _curl_cffi_requests = None  # type: ignore[assignment]
    _CURL_CFFI_AVAILABLE = False

from src.config.settings import get_settings
from src.tools.builtin._net_safety import assert_public_host

_USER_AGENT = "Mozilla/5.0 (compatible; TuringAgent/1.0)"
# Anti-bot signals: Cloudflare/bot-WAF challenge (403) and rate-limit (429).
# On these, httpx is retried once via curl_cffi TLS impersonation (see
# ``_fetch_html``). Other 4xx/5xx are NOT retried — they aren't TLS-blocked.
_ANTIBOT_STATUS: frozenset[int] = frozenset({403, 429})
# Download-byte cap and fetch timeout are operator-configurable via
# ToolLimitsSettings (WEB_SCRAPER_MAX_BYTES / WEB_SCRAPER_TIMEOUT). The
# schema display default below mirrors the settings default; enforcement
# reads settings at call-time via _tool_limits().
_SCHEMA_DEFAULT_MAX_CHARS = 8000  # mirrors ToolLimitsSettings.web_scraper_max_chars

# Lazily-initialized tiktoken encoder for token-mode chunking. None until
# first use; False if the BPE file can't be loaded (offline) → char fallback.
_TIKTOKEN_ENC: object | None = None


def _tool_limits():
    """Call-time accessor — never capture get_settings() at module import."""
    return get_settings().tools


def _search_settings():
    """Call-time accessor for the SearchSettings group (chunk params)."""
    return get_settings().search


class _FetchError(Exception):
    """Carries a full user-facing ERROR: message from the fetch/extract path."""


# ── AI-format helpers ───────────────────────────────────────────────────


def compute_content_hash(content: str) -> str:
    """Stable 32-char SHA-256 of whitespace-normalized content (dedup key)."""
    return hashlib.sha256(" ".join(content.split()).encode("utf-8")).hexdigest()[:32]


def chunk_text(
    text: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
    mode: str = "char",
) -> list[str]:
    """Split ``text`` into overlapping chunks for LLM/corpus indexing.

    Args:
        text: Source text to chunk.
        chunk_size: Target chunk size (characters in "char" mode, tokens in
            "token" mode). Defaults to ``SearchSettings.chunk_size``.
        chunk_overlap: Overlap between consecutive chunks (same units). Defaults
            to ``SearchSettings.chunk_overlap``.
        mode: ``"char"`` (deterministic, no deps — default) or ``"token"``
            (tiktoken ``cl100k_base``; falls back to char mode if the encoder
            can't load offline).

    Returns:
        Non-empty overlapping chunks, or ``[]`` for empty/blank input.
    """
    if not text or not text.strip():
        return []
    size = max(1, int(chunk_size))
    overlap = max(0, min(int(chunk_overlap), size - 1))
    if mode == "token":
        token_chunks = _chunk_tokens(text, size, overlap)
        if token_chunks:
            return token_chunks
        # tiktoken unavailable offline → fall through to char mode.
    return _chunk_chars(text, size, overlap)


def _chunk_chars(text: str, size: int, overlap: int) -> list[str]:
    step = max(1, size - overlap)
    out: list[str] = []
    for i in range(0, len(text), step):
        piece = text[i : i + size]
        if piece.strip():
            out.append(piece)
        if i + size >= len(text):
            break
    return out or [text[:size]]


def _get_tiktoken() -> object | None:
    global _TIKTOKEN_ENC
    if _TIKTOKEN_ENC is None:
        try:
            _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _TIKTOKEN_ENC = False  # unavailable (offline) — token mode degrades
    return _TIKTOKEN_ENC if _TIKTOKEN_ENC is not False else None


def _chunk_tokens(text: str, size: int, overlap: int) -> list[str]:
    enc = _get_tiktoken()
    if enc is None:
        return []
    encode = getattr(enc, "encode")
    decode = getattr(enc, "decode")
    tokens = encode(text)
    if not tokens:
        return []
    step = max(1, size - overlap)
    out: list[str] = []
    for i in range(0, len(tokens), step):
        piece = decode(tokens[i : i + size])
        if piece.strip():
            out.append(piece)
        if i + size >= len(tokens):
            break
    return out or [decode(tokens[:size])]


# ── fetch + extraction ─────────────────────────────────────────────────


@dataclass
class ExtractedPage:
    """Structured extraction result for a fetched page."""

    url: str
    title: str
    description: str
    markdown: str
    text: str
    content_hash: str
    metadata: dict[str, str] = field(default_factory=dict)


def _fetch_html_curl_cffi(url: str, timeout: float, max_bytes: int, impersonate: str) -> str:
    """Retry a fetch with a TLS-impersonated Chrome session (anti-bot tier).

    Called only from ``_fetch_html`` after httpx hit an anti-bot status
    (403/429). ``curl_cffi`` replays the request with a Chrome JA3 fingerprint,
    bypassing Cloudflare/bot-WAF TLS blocking that httpx cannot. Enforces the
    same byte cap + timeout as the httpx path. Raises ``_FetchError`` (carrying
    a full ``ERROR:`` message) on any failure — the caller has already fallen
    through, so this error surfaces to the user rather than masking the block.
    """
    assert _curl_cffi_requests is not None  # checked by caller; narrows for type-checkers
    try:
        # ``impersonate`` is operator config (a str read from .env); splat it
        # through a loosely-typed kwargs dict so pyright doesn't narrow it
        # against curl_cffi's ``BrowserTypeLiteral`` (curl_cffi validates the
        # value at runtime, raising on an unknown browser identifier).
        session_kwargs: dict[str, Any] = {"impersonate": impersonate}
        with _curl_cffi_requests.Session(**session_kwargs) as session:
            resp = session.get(url, timeout=timeout, allow_redirects=True)
            status = resp.status_code
            # Still blocked (or any non-success) → surface it; no further retry.
            if not (200 <= status < 400):
                raise _FetchError(f"ERROR: HTTP {status} fetching {url}")
            # Byte cap: Content-Length pre-check (best-effort) + post-download cap.
            cl = resp.headers.get("Content-Length") if resp.headers else None
            if cl and cl.strip().isdigit() and int(cl) > max_bytes:
                raise _FetchError(
                    f"ERROR: Page exceeds {max_bytes // (1024 * 1024)} MB download cap"
                )
            body = resp.content or b""
            if len(body) > max_bytes:
                raise _FetchError(
                    f"ERROR: Page exceeds {max_bytes // (1024 * 1024)} MB download cap"
                )
            return body.decode("utf-8", errors="replace")
    except _FetchError:
        raise
    except Exception as exc:  # curl_cffi raises CurlError etc.; surface as ERROR:
        raise _FetchError(f"ERROR: curl_cffi fetch failed: {exc}") from exc


def _fetch_html(url: str, timeout: float, max_bytes: int) -> str:
    """Fetch the raw HTML with an SSRF guard, size cap, timeout, and anti-bot tier.

    On an anti-bot signal (HTTP 403/429 from httpx) and when ``curl_cffi`` is
    enabled + importable, retries once via a Chrome-impersonated TLS session
    (Cloudflare/bot-WAF bypass — Gap 7). Raises ``_FetchError`` (carrying a full
    ``ERROR:`` message) on any failure.
    """
    err = assert_public_host(url)
    if err:
        raise _FetchError(err)
    try:
        with httpx.Client(
            timeout=timeout,
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
                        raise _FetchError(
                            f"ERROR: Page exceeds {max_bytes // (1024 * 1024)} MB "
                            f"download cap"
                        )
                    chunks.append(chunk)
                return b"".join(chunks).decode("utf-8", errors="replace")
    except _FetchError:
        raise
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        # Anti-bot tier: on a Cloudflare block (403) or rate-limit (429), retry
        # once with a TLS-impersonated Chrome session if available. Only these
        # TLS-blocked statuses qualify — a genuine 404/500 is not a fingerprint
        # block and must not be retried.
        if status in _ANTIBOT_STATUS:
            limits = _tool_limits()
            if (
                limits.web_scraper_curl_cffi_enabled
                and _CURL_CFFI_AVAILABLE
                and _curl_cffi_requests is not None
            ):
                logger.debug(
                    f"web_scraper: anti-bot {status} from {url[:80]}, retrying via curl_cffi"
                )
                return _fetch_html_curl_cffi(
                    url, timeout, max_bytes, limits.web_scraper_curl_cffi_impersonate
                )
        raise _FetchError(f"ERROR: HTTP {status} fetching {url}") from exc
    except httpx.HTTPError as exc:
        raise _FetchError(f"ERROR: Fetch failed: {exc}") from exc


def _extract_markdown(html: str) -> str:
    """Main-content markdown via trafilatura, with an optional markdownify fallback.

    markdownify is the fallback for pages trafilatura can't parse; it is OPTIONAL
    (imported defensively at module top). When absent, this returns trafilatura's
    result (or ``""`` when trafilatura also yielded nothing) — a missing optional
    fallback must never raise, so a graph build never fails on it (Bug A).
    """
    extracted = trafilatura.extract(
        html, output_format="markdown", include_tables=True, include_links=True
    )
    if extracted and extracted.strip():
        return extracted
    if _MARKDOWNIFY_AVAILABLE and _markdownify is not None:
        try:
            alt = _markdownify(html)
            if alt and alt.strip():
                return alt
        except Exception:
            pass
    return ""


def extract_page(url: str, timeout: float, max_bytes: int) -> ExtractedPage:
    """Fetch + extract a page into structured fields (sync; run off the loop).

    Raises ``_FetchError`` on fetch/size failure. ``markdown`` may be empty when
    no readable content could be extracted (callers decide how to handle that).
    Metadata is best-effort: a missing field is simply omitted.

    Uses ``trafilatura.extract_metadata`` (stable ``Document`` with string fields:
    title/description/author/sitename/hostname/language/date) for metadata and
    ``trafilatura.extract`` for plain text. ``bare_extraction`` is avoided — in
    trafilatura 2.x its ``.body`` is an lxml Element, not a string.
    """
    html = _fetch_html(url, timeout, max_bytes)

    meta: object | None = None
    try:
        meta = trafilatura.extract_metadata(html, default_url=url)
    except Exception:
        meta = None

    def _field(name: str) -> str:
        val = getattr(meta, name, None) if meta else None
        return val.strip() if isinstance(val, str) else ""

    title = _field("title")
    description = _field("description")

    raw_text = ""
    try:
        raw_text = trafilatura.extract(
            html, include_tables=True, include_links=True
        )
    except Exception:
        raw_text = ""
    text = raw_text.strip() if isinstance(raw_text, str) else ""
    markdown = _extract_markdown(html) or text

    metadata: dict[str, str] = {}
    for key in ("sitename", "author", "date", "hostname", "language"):
        val = _field(key)
        if val:
            metadata[key] = val
    if title:
        metadata["title"] = title
    if description:
        metadata["description"] = description

    return ExtractedPage(
        url=url,
        title=title,
        description=description,
        markdown=markdown,
        text=text,
        content_hash=compute_content_hash(markdown or text),
        metadata=metadata,
    )


def _extract(url: str, max_chars: int, timeout: float, max_bytes: int) -> str:
    """Backward-compat seam: markdown (possibly truncated) or an ``ERROR:`` string."""
    try:
        page = extract_page(url, timeout, max_bytes)
    except _FetchError as exc:
        return str(exc)
    markdown = page.markdown
    if not markdown:
        return f"ERROR: Could not extract readable content from {url}"
    if len(markdown) > max_chars:
        markdown = markdown[:max_chars] + f"\n\n... (truncated at {max_chars} chars)"
    return markdown


async def web_scraper(
    url: str, max_chars: Optional[int] = None, chunk: bool = False
) -> str:
    """Fetch a URL and return its main content as clean markdown.

    Args:
        url: Absolute ``http(s)`` URL to scrape.
        max_chars: Maximum characters of extracted markdown to return (single
            mode). ``None`` resolves to ``WEB_SCRAPER_MAX_CHARS``
            (ToolLimitsSettings, default 8000). Ignored in chunk mode.
        chunk: When True, return the page content as overlapping chunks
            (``CHUNK_SIZE`` / ``CHUNK_OVERLAP``) joined as ``[Chunk N/M]``
            blocks — for large pages or corpus indexing.

    Returns:
        Clean markdown of the page's main content (or joined chunks), or an
        ``ERROR:`` string. Private/loopback URLs are blocked (SSRF guard).
    """
    limits = _tool_limits()
    if max_chars is None:
        max_chars = limits.web_scraper_max_chars
    logger.info(f"web_scraper: {url[:80]} (chunk={chunk})")
    try:
        if chunk:
            page = await asyncio.to_thread(
                extract_page, url, limits.web_scraper_timeout, limits.web_scraper_max_bytes
            )
            s = _search_settings()
            chunks = chunk_text(page.markdown or page.text, s.chunk_size, s.chunk_overlap)
            if not chunks:
                return f"ERROR: Could not extract readable content from {url}"
            return "\n\n---\n\n".join(
                f"[Chunk {i + 1}/{len(chunks)}]\n{c}" for i, c in enumerate(chunks)
            )
        return await asyncio.to_thread(
            _extract, url, max_chars, limits.web_scraper_timeout, limits.web_scraper_max_bytes
        )
    except _FetchError as exc:
        return str(exc)
    except Exception as exc:
        return f"ERROR: Scrape failed: {exc}"


TOOL_DEFINITION = {
    "name": "web_scraper",
    "handler": web_scraper,
    "description": (
        "Fetch a specific URL and return its main content as clean markdown, "
        "stripping navigation, ads, and HTML/JS boilerplate. Use this AFTER "
        "web_search to read a chosen result page, or on any known URL. Only "
        "public http(s) URLs are allowed (private/loopback hosts are blocked). "
        "Pass chunk=true to return the content as overlapping chunks for large "
        "pages or corpus indexing."
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
                "description": "Maximum characters of extracted text (single mode; default 8000, configurable via WEB_SCRAPER_MAX_CHARS). Ignored when chunk=true.",
                "default": _SCHEMA_DEFAULT_MAX_CHARS,
            },
            "chunk": {
                "type": "boolean",
                "description": (
                    "Return the content as overlapping chunks "
                    "(CHUNK_SIZE/CHUNK_OVERLAP) instead of one truncated "
                    "markdown string. Default false."
                ),
                "default": False,
            },
        },
        "required": ["url"],
    },
}
