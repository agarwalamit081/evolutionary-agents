"""Tool that fetches a URL and returns its main content as clean markdown.

Distinct from ``web_search`` (which returns search snippets + links): this
fetches a *specific* URL, strips HTML/JS boilerplate, and exposes the
AI-format extraction layer (Phase 1): structured metadata, content-hash
dedup key, and overlapping chunking for LLM/corpus indexing.

Extraction stack:
  * ``trafilatura`` for main-content extraction (markdown) + metadata
    (``bare_extraction``: title/description/sitename/author/date/hostname).
  * ``markdownify`` as a fallback for pages trafilatura can't parse.
  * No ``selectolax`` / ``curl-cffi`` dependency (not installed in this env);
    metadata comes from trafilatura and the HTTP client is ``httpx``
    (the curl-cffi anti-bot tier is a documented future enhancement).

Public surface reused by ``corpus.py``: ``extract_page`` (structured),
``chunk_text`` (overlapping chunks), ``compute_content_hash`` (dedup key).
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import Optional

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

from src.config.settings import get_settings
from src.tools.builtin._net_safety import assert_public_host

_USER_AGENT = "Mozilla/5.0 (compatible; TuringAgent/1.0)"
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


def _fetch_html(url: str, timeout: float, max_bytes: int) -> str:
    """Fetch the raw HTML with an SSRF guard, size cap, and timeout.

    Raises ``_FetchError`` (carrying a full ``ERROR:`` message) on any failure.
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
        raise _FetchError(f"ERROR: HTTP {exc.response.status_code} fetching {url}") from exc
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
