"""Image-generation builtin — text→image via litellm ``aimage_generation``.

Unlike ``ocr_parser`` (which calls Z.AI's layout-parsing HTTP endpoint
directly), image generation has a first-class litellm primitive
(``litellm.aimage_generation``), so this tool is provider-agnostic: the model,
size, and quality are all operator knobs (``IMAGE_GEN_*``), defaulting to
OpenAI ``gpt-image-1`` (a verified litellm image-gen model, ~$0.011/image at
low/1024²). Swap ``IMAGE_GEN_MODEL`` for e.g. ``dashscope/qwen-image-2.0`` to
reuse the existing DASHSCOPE key.

The generated PNG is written into the per-run results subdir
(``normalize(filename, base=results_root())`` — run-isolated when
``RESULTS_PER_RUN_SUBDIR`` is on) and its on-disk path returned so the agent
can cite the deliverable. If a provider returns only a URL (no base64), that
URL is returned verbatim instead — we never auto-download it, to avoid a
second network hop and SSRF nuance.
"""

from __future__ import annotations

import base64
import secrets
from pathlib import Path
from typing import Any

import litellm
from loguru import logger

from src.config.settings import get_settings
from src.tools._paths import normalize, results_root

#: Filename-safe extension for generated images.
_IMAGE_EXT = ".png"
#: Max length of a caller-supplied filename stem (defensive cap).
_MAX_STEM_LEN = 80


def _safe_stem(filename: str | None) -> str:
    """Reduce a caller-supplied filename to a safe basename stem.

    Strips path separators / traversal and clamps length. ``None`` → a short
    random default (``image_<hex>``) so repeated calls never collide on disk.
    """
    if not filename:
        return f"image_{secrets.token_hex(3)}"
    # Take only the final path component, drop any explicit extension, then
    # remove anything that isn't word/dash/dot so it can't escape the results root.
    stem = Path(filename).name
    if stem.lower().endswith(_IMAGE_EXT):
        stem = stem[: -len(_IMAGE_EXT)]
    cleaned = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in stem).strip("._")
    if not cleaned:
        cleaned = f"image_{secrets.token_hex(3)}"
    return cleaned[:_MAX_STEM_LEN]


async def image_generator(
    prompt: str,
    size: str | None = None,
    quality: str | None = None,
    filename: str | None = None,
) -> str:
    """Generate one image from ``prompt`` and persist it under the results root.

    Args:
        prompt: The text-to-image prompt (required).
        size: Output size, e.g. ``"1024x1024"``. Defaults to
            ``IMAGE_GEN_DEFAULT_SIZE``.
        quality: Generation quality, e.g. ``"low"`` / ``"medium"`` / ``"high"``.
            Defaults to ``IMAGE_GEN_DEFAULT_QUALITY`` (cost-tier selectable).
        filename: Optional output basename (``.png`` appended if absent). A
            random stem is used when omitted.

    Returns:
        The on-disk path of the written PNG (e.g.
        ``results/<run_id>/image_abc.png``), or a provider URL if the model
        returned one instead of base64. A sanitized ``ERROR: ...`` string on
        any failure (no key/path/stack trace leaked).
    """
    if not prompt or not prompt.strip():
        return "ERROR: prompt is required and must be non-empty"

    tools = get_settings().tools
    model = tools.image_gen_model
    eff_size = (size or tools.image_gen_default_size).strip() or None
    eff_quality = (quality or tools.image_gen_default_quality).strip() or None
    timeout = tools.image_gen_timeout

    # Only forward the optional kwargs the provider accepts; passing size/quality
    # as None lets a provider that ignores them proceed cleanly.
    kwargs: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "n": 1,
        "response_format": "b64_json",
        "timeout": timeout,
    }
    if eff_size:
        kwargs["size"] = eff_size
    if eff_quality:
        kwargs["quality"] = eff_quality
    if tools.image_gen_api_base.strip():
        kwargs["api_base"] = tools.image_gen_api_base.strip()

    try:
        response = await litellm.aimage_generation(**kwargs)
    except Exception as exc:  # noqa: BLE001 — sanitize every provider failure
        logger.warning("image_generator call failed for model={}: {}", model, type(exc).__name__)
        return f"ERROR: image generation failed ({type(exc).__name__})"

    data = getattr(response, "data", None) or []
    if not data:
        return "ERROR: image generation returned no data"
    first = data[0]

    b64 = getattr(first, "b64_json", None)
    if b64:
        try:
            raw = base64.b64decode(b64, validate=False)
        except (ValueError, base64.binascii.Error) as exc:  # type: ignore[attr-defined]
            logger.warning("image_generator base64 decode failed: {}", type(exc).__name__)
            return "ERROR: image generation returned undecodable image data"
        if not raw:
            return "ERROR: image generation returned empty image data"

        target = normalize(f"{_safe_stem(filename)}{_IMAGE_EXT}", base=results_root())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        return f"Wrote generated image ({len(raw)} bytes) to {target}"

    url = getattr(first, "url", None)
    if url:
        return f"Generated image URL (provider returned no inline image): {url}"

    return "ERROR: image generation returned neither an inline image nor a URL"


TOOL_DEFINITION = {
    "name": "image_generator",
    "handler": image_generator,
    "description": (
        "Generate an image from a text prompt via litellm image generation "
        "(default model gpt-image-1; configurable via IMAGE_GEN_MODEL). "
        "Writes the resulting PNG into the results dir and returns its path "
        "so it can be cited as a deliverable. Use for diagrams, illustrations, "
        "or any visual artifact a goal asks for. Costs ~$0.01/image at low "
        "quality; prefer low/1024x1024 unless higher fidelity is required."
    ),
    "cacheable": False,
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The text-to-image prompt describing the image to generate.",
            },
            "size": {
                "type": "string",
                "description": (
                    "Output size, e.g. '1024x1024', '1024x1536', '1536x1024'. "
                    "Defaults to IMAGE_GEN_DEFAULT_SIZE (1024x1024)."
                ),
            },
            "quality": {
                "type": "string",
                "description": (
                    "Generation quality: 'low', 'medium', or 'high'. "
                    "Defaults to IMAGE_GEN_DEFAULT_QUALITY (low — cheapest)."
                ),
            },
            "filename": {
                "type": "string",
                "description": (
                    "Optional output basename (extension appended automatically). "
                    "A random stem is used when omitted."
                ),
            },
        },
        "required": ["prompt"],
    },
}
