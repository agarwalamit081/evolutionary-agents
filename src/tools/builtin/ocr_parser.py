"""Tool running GLM-OCR (Z.AI layout_parsing) over an image or PDF.

Distinct from ``document_parser`` (which text-mines an already-text-layered
file locally via pypdf/openpyxl/trafilatura): this calls the GLM-OCR model for
true OCR + layout understanding of scanned/image-only documents — returning
markdown that preserves headings, tables, and formulas. A layout-parsing API is
NOT a chat-completions model, so it cannot live in ``MODEL_REGISTRY`` (no
``acompletion`` route); it is a dedicated fetch tool, same shape as
``http_request``/``web_scraper``.

Input is exactly one of a sandboxed ``file_path`` (base64-encoded and sent to
the API) or a public ``url`` (SSRF-guarded). Returns the API's ``md_results``
markdown, capped. Path traversal and private-host requests are blocked.
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Optional

import httpx
from loguru import logger

from src.config.settings import get_settings
from src.tools.builtin._net_safety import assert_public_host

# Z.AI (international) layout-parsing endpoint. Fixed host; the tool only ever
# talks to this one allowlisted API, so the SSRF guard is applied to the *input*
# url (the document), never to this base.
_ZAI_LAYOUT_PARSING_URL = "https://api.z.ai/api/paas/v4/layout_parsing"
# API limits: single image ≤10 MB, PDF ≤50 MB. Cap inputs at the image ceiling
# so a base64 payload never balloons an already-large file through the gateway.
_MAX_INPUT_BYTES = 10_000_000
_ALLOWED_EXT = frozenset({".png", ".jpg", ".jpeg", ".pdf"})
# Z.AI layout_parsing needs a data-URI to detect an inline file's format — raw
# base64 is rejected with HTTP 400 ("OCR only supports PDF, JPG, PNG, JPEG
# formats"). Map each allowed extension to its MIME for the prefix.
_MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".pdf": "application/pdf",
}


def _tool_limits():
    """Call-time accessor — never capture get_settings() at module import."""
    return get_settings().tools


def _resolve_file(file_path: str, sandbox_root: str) -> tuple[Path, str]:
    """Resolve a sandboxed file → (path, base64 payload) or raise ValueError."""
    root = Path(sandbox_root).resolve()
    target = (root / file_path).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"path traversal blocked: {file_path}")
    if not target.exists() or not target.is_file():
        raise ValueError(f"file not found: {file_path}")
    if target.suffix.lower() not in _ALLOWED_EXT:
        raise ValueError(
            f"unsupported type '{target.suffix}' — use one of {sorted(_ALLOWED_EXT)}"
        )
    if target.stat().st_size > _MAX_INPUT_BYTES:
        raise ValueError(f"file too large ({target.stat().st_size} bytes, max {_MAX_INPUT_BYTES})")
    raw = target.read_bytes()
    # Z.AI layout_parsing rejects raw base64 (HTTP 400 "OCR only supports PDF,
    # JPG, PNG, JPEG formats") — it needs a data-URI to detect inline format.
    mime = _MIME_BY_EXT.get(target.suffix.lower(), "application/octet-stream")
    payload = f"data:{mime};base64,{base64.b64encode(raw).decode()}"
    return target, payload


async def ocr_parser(
    file_path: Optional[str] = None,
    url: Optional[str] = None,
    max_chars: Optional[int] = None,
) -> str:
    """Run GLM-OCR over an image (PNG/JPG) or PDF and return extracted markdown.

    Args:
        file_path: Relative path to an image/PDF within the sandbox root
            (defaults to ``settings.agent.workspace_root``). Base64-encoded and
            sent to Z.AI. Path traversal is blocked.
        url: Public https URL of an image/PDF to OCR. Private/loopback hosts are
            blocked (SSRF guard). Provide exactly one of ``file_path``/``url``.
        max_chars: Maximum markdown characters to return. ``None`` resolves to
            ``OCR_PARSER_MAX_CHARS`` (default 8000).

    Returns:
        Extracted markdown (``md_results``), or an ``ERROR:`` string. Requires
        ``ZAI_API_KEY``.
    """
    limits = _tool_limits()
    if max_chars is None:
        max_chars = limits.ocr_parser_max_chars
    timeout = limits.ocr_parser_timeout
    api_key = get_settings().llm.zai_api_key
    if not api_key:
        return "ERROR: ZAI_API_KEY not configured — GLM-OCR unavailable"

    if (file_path is None) == (url is None):
        return "ERROR: Provide exactly one of file_path or url."

    if file_path is not None:
        sandbox_root = get_settings().agent.workspace_root
        try:
            target, file_payload = await asyncio.to_thread(
                _resolve_file, file_path, sandbox_root
            )
        except ValueError as exc:
            return f"ERROR: {exc}"
        source_label = target.name
    else:
        assert url is not None  # narrowed by the exactly-one check above
        err = await asyncio.to_thread(assert_public_host, url)
        if err:
            return err
        file_payload = url
        source_label = url[:80]

    logger.info(f"ocr_parser: {source_label}")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {"model": "glm-ocr", "file": file_payload}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(_ZAI_LAYOUT_PARSING_URL, headers=headers, json=body)
    except httpx.TimeoutException:
        return f"ERROR: GLM-OCR request timed out after {timeout}s"
    except httpx.HTTPError as exc:
        return f"ERROR: GLM-OCR request failed: {exc}"
    except Exception as exc:  # noqa: BLE001 — surface any failure, never leak the key
        return f"ERROR: GLM-OCR request failed: {exc}"

    if response.status_code != 200:
        # Sanitized error — never echo headers or the full body (may carry trace ids).
        return f"ERROR: GLM-OCR returned HTTP {response.status_code}"

    try:
        payload = response.json()
    except ValueError:
        return "ERROR: GLM-OCR returned a non-JSON response"

    md = (payload.get("md_results") or "").strip()
    if not md:
        return "ERROR: GLM-OCR returned no extractable text"
    if len(md) > max_chars:
        md = md[:max_chars] + f"\n\n... (truncated at {max_chars} chars)"
    return md


TOOL_DEFINITION = {
    "name": "ocr_parser",
    "handler": ocr_parser,
    "description": (
        "Run the GLM-OCR model over a scanned/image-only document (PNG, JPG, or "
        "PDF) and return the extracted text as markdown, preserving headings, "
        "tables, and formulas. Use this when document_parser finds no text layer "
        "(a scanned PDF or a photo of a page). Provide exactly one of file_path "
        "(sandboxed) or url (public https). Requires ZAI_API_KEY. Path traversal "
        "and private-host requests are blocked."
    ),
    # OCR is a deterministic transform of one input — safe to cache by payload.
    "cacheable": True,
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Relative path to a PNG/JPG/PDF within the project directory.",
            },
            "url": {
                "type": "string",
                "description": "Public https URL of a PNG/JPG/PDF to OCR.",
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum markdown characters to return (default: 8000, configurable via OCR_PARSER_MAX_CHARS).",
                "default": 8000,
            },
        },
        # Neither is individually required, but exactly one must be supplied
        # (enforced in the handler). Omiting ``required`` lets the schema match
        # either call shape.
    },
}
