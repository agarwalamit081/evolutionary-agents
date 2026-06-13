"""Tool making controlled HTTP requests to external APIs/services.

Distinct from ``web_search`` (search) and ``web_scraper`` (page → markdown):
this performs structured HTTP with an explicit method, headers, and body, and
returns the status line + raw response text. Method allowlist + SSRF guard keep
it safer than ad-hoc ``code_executor`` scripts.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import httpx
from loguru import logger

from src.tools.builtin._net_safety import assert_public_host

_ALLOWED_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE"})
_MAX_RESPONSE_CHARS = 8000
_MAX_BODY_BYTES = 1_000_000  # 1 MB request body cap
_USER_AGENT = "Mozilla/5.0 (compatible; TuringAgent/1.0)"


async def http_request(
    url: str,
    method: str = "GET",
    headers: Optional[dict[str, str]] = None,
    body: Optional[str] = None,
    timeout: float = 15.0,
    max_chars: int = _MAX_RESPONSE_CHARS,
) -> str:
    """Make a controlled HTTP request and return the status + response text.

    Args:
        url: Absolute ``http(s)`` URL.
        method: HTTP method — one of GET, POST, PUT, PATCH, DELETE (default GET).
        headers: Optional request headers. ``User-Agent`` is set automatically.
        body: Optional request body (sent verbatim; capped at 1 MB).
        timeout: Request timeout in seconds (default 15.0).
        max_chars: Maximum response characters to return (default 8000).

    Returns:
        ``HTTP <status>\\nContent-Type: ...\\n\\n<body>`` or an ``ERROR:`` string.
        Private/loopback URLs are blocked (SSRF guard).
    """
    method_upper = (method or "GET").upper()
    if method_upper not in _ALLOWED_METHODS:
        return (
            f"ERROR: Method '{method}' not allowed. "
            f"Use one of: {', '.join(sorted(_ALLOWED_METHODS))}."
        )

    err = await asyncio.to_thread(assert_public_host, url)
    if err:
        return err

    request_headers = {"User-Agent": _USER_AGENT}
    if headers:
        # Merge caller headers last; never log them (may contain credentials).
        request_headers.update({str(k): str(v) for k, v in headers.items()})

    content: Optional[bytes] = None
    if body is not None:
        content = body.encode("utf-8")
        if len(content) > _MAX_BODY_BYTES:
            return f"ERROR: Request body exceeds {_MAX_BODY_BYTES} bytes"

    logger.info(f"http_request: {method_upper} {url[:80]}")

    try:
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=True, headers=request_headers
        ) as client:
            response = await client.request(method_upper, url, content=content)
    except httpx.TimeoutException:
        return f"ERROR: Request timed out after {timeout}s: {url}"
    except httpx.HTTPError as exc:
        return f"ERROR: Request failed: {exc}"
    except Exception as exc:
        return f"ERROR: Request failed: {exc}"

    text = response.text
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n... (truncated at {max_chars} chars)"

    content_type = response.headers.get("content-type", "")
    return f"HTTP {response.status_code}\nContent-Type: {content_type}\n\n{text}"


TOOL_DEFINITION = {
    "name": "http_request",
    "handler": http_request,
    "description": (
        "Make a structured HTTP request (GET/POST/PUT/PATCH/DELETE) to an "
        "external API or service, returning the status code, content-type, and "
        "response text. Use this for JSON/REST APIs instead of writing fetch "
        "code. Only public http(s) URLs are allowed (private/loopback hosts "
        "are blocked)."
    ),
    # Handles mutating methods (POST/PUT/DELETE) — never cache.
    "cacheable": False,
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Absolute http(s) URL to request.",
            },
            "method": {
                "type": "string",
                "description": "HTTP method: GET, POST, PUT, PATCH, or DELETE.",
                "default": "GET",
                "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
            },
            "headers": {
                "type": "object",
                "description": "Optional request headers (object of name→value).",
            },
            "body": {
                "type": "string",
                "description": "Optional request body (capped at 1 MB).",
            },
            "timeout": {
                "type": "number",
                "description": "Request timeout in seconds (default: 15.0).",
                "default": 15.0,
            },
        },
        "required": ["url"],
    },
}
