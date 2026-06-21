"""Provider-native prompt caching: Anthropic ``cache_control`` breakpoints.

Anthropic serves prompt caching by marking content blocks with a
``cache_control: {"type": "ephemeral"}`` marker; litellm forwards these markers
to the Anthropic API unmodified (verified in litellm's Anthropic chat
transformations). For an agent that re-sends a large, stable system prompt
across many calls within a run, tagging that system message is the highest-
value, lowest-risk anchor — one breakpoint captures the bulk of the saving.

This module is opt-in and called per-attempt from
``LLMGateway._execute_with_fallback``. When disabled (or the provider is not
Anthropic) the messages list is returned unchanged. Anthropic caps the number
of cache breakpoints, so exactly one is injected (on the first qualifying
system message).
"""

from __future__ import annotations

from typing import Any


def _approx_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token), floored at 1.

    Good enough to decide whether a system prompt is worth a cache breakpoint;
    not used for billing or budget accounting.
    """
    return max(1, len(text) // 4)


def _mark_block(block: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a text content block with an ephemeral cache_control."""
    return {**block, "cache_control": {"type": "ephemeral"}}


def inject_cache_breakpoints(
    messages: list[dict[str, Any]],
    provider: str,
    *,
    enabled: bool,
    min_system_tokens: int = 1024,
) -> list[dict[str, Any]]:
    """Inject an Anthropic ``cache_control`` breakpoint on a long system message.

    Args:
        messages: OpenAI-format chat messages. Never mutated; a new list (and,
            where tagged, new dicts) is returned so the caller's messages stay
            reusable across fallback attempts.
        provider: Provider for the current attempt (e.g. ``"anthropic"``).
        enabled: Master switch (``PromptCacheControlSettings.enabled``).
        min_system_tokens: Minimum estimated system-prompt size before a
            breakpoint is worth the cache write cost.

    Returns:
        A messages list. When the feature is off, the provider is not Anthropic,
        or no qualifying system message exists, the input is returned unchanged
        (same object). Otherwise the first qualifying system message has its
        content replaced by a single text block carrying ``cache_control``.
    """
    if not enabled or provider != "anthropic":
        return messages

    out: list[dict[str, Any]] = []
    marked = False
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "system" or marked:
            out.append(msg)
            continue

        content = msg.get("content")
        if isinstance(content, str):
            if _approx_tokens(content) < min_system_tokens:
                out.append(msg)
                continue
            out.append(
                {
                    **msg,
                    "content": [_mark_block({"type": "text", "text": content})],
                }
            )
            marked = True
            continue

        # Already-structured content list: tag the trailing text block if it is
        # large enough and not already carrying a breakpoint.
        if isinstance(content, list) and content:
            last = content[-1]
            if (
                isinstance(last, dict)
                and last.get("type") == "text"
                and "cache_control" not in last
                and _approx_tokens(str(last.get("text", ""))) >= min_system_tokens
            ):
                tagged_content = [dict(c) for c in content]
                tagged_content[-1] = _mark_block(last)
                out.append({**msg, "content": tagged_content})
                marked = True
                continue

        out.append(msg)

    # No-op: hand back the caller's list so they can cheaply detect unchanged.
    return messages if not marked else out
