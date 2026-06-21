"""Per-tier reasoning/thinking control.

Selectively enables extended thinking/reasoning based on task complexity, so
expensive reasoning modes spend tokens where they help (complex/critical tasks)
and stay off for trivial ones. Opt-in and default-OFF
(``REASONING_CONTROL_ENABLED``); when disabled, no thinking params are emitted
(zero behavior change).

Provider mappings (verified against litellm 1.83.14):
- **Anthropic**: top-level ``thinking={"type":"enabled","budget_tokens":N}``
  (litellm forwards it). Thinking is OFF by default, so no disable is emitted.
  Anthropic also requires ``temperature=1.0`` when thinking is enabled — that
  invariant is baked into the enable params.
- **DeepSeek**: thinking is ON by default; disable via
  ``extra_body={"thinking":{"type":"disabled"}}`` (the native
  ``reasoning_effort="none"`` param is BROKEN in 1.83.14 — it silently drops).
  Enable via ``extra_body={"thinking":{"type":"enabled"}}`` (binary, no budget).
- **Z.AI (GLM)**: ``extra_body={"thinking":{"type":"enabled"}}`` to turn on; off
  is its default, so no disable is emitted for trivial tasks.
- **OpenAI o-series**: top-level ``reasoning_effort`` ("low"/"medium"/"high").

A caller's explicit ``thinking``/``reasoning_effort`` always wins: the gateway
merges this module's output with ``setdefault`` semantics. ``asyncio``-free,
pure, and unit-testable.
"""

from __future__ import annotations

from typing import Any

from src.graph.enums import TaskComplexity


# Complexities that warrant extended thinking when the feature is on.
_HEAVY_COMPLEXITIES: frozenset[TaskComplexity] = frozenset(
    {TaskComplexity.COMPLEX, TaskComplexity.CRITICAL}
)

# Providers whose extended-thinking models are ON by default and therefore get
# an explicit DISABLE when this layer wants thinking OFF for trivial tasks.
# (Anthropic and OpenAI o-series default to off → no disable param is needed.)
_DEFAULT_ON_PROVIDERS: frozenset[str] = frozenset({"deepseek"})


def _is_o_series(model: str) -> bool:
    """Heuristic: does this OpenAI model accept ``reasoning_effort``?"""
    lowered = model.lower().split("/")[-1]
    return lowered.startswith(("o1", "o3", "o4", "o5"))


def _disable_params(provider: str) -> dict[str, Any]:
    """Disable thinking for providers whose default is ON; no-op otherwise."""
    if provider in _DEFAULT_ON_PROVIDERS:
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    return {}


def thinking_params_for(
    complexity: TaskComplexity | None,
    provider: str,
    model: str,
    settings: Any,
) -> dict[str, Any]:
    """Return litellm kwargs enabling/disabling thinking for this tier.

    Returns ``{}`` when the feature is disabled, ``complexity`` is None, the
    resolved effort is ``"none"``, or the provider does not support extended
    thinking. The gateway merges the result with caller-supplied params using
    setdefault semantics, so an explicit caller override (``thinking`` /
    ``reasoning_effort``) always wins. A ``temperature`` key, when present,
    enforces a provider invariant (Anthropic requires 1.0 with thinking) and is
    force-applied.
    """
    if not getattr(settings, "enabled", False) or complexity is None:
        return {}

    heavy = complexity in _HEAVY_COMPLEXITIES
    effort = settings.complex_thinking if heavy else settings.simple_thinking
    if not effort or effort == "none":
        return _disable_params(provider)

    # Thinking ON.
    if provider == "anthropic":
        budget = (
            settings.anthropic_budget_tokens_complex
            if heavy
            else settings.anthropic_budget_tokens_medium
        )
        # Anthropic requires temperature=1.0 when extended thinking is enabled.
        return {
            "thinking": {"type": "enabled", "budget_tokens": int(budget)},
            "temperature": 1.0,
        }
    if provider == "openai" and _is_o_series(model):
        return {"reasoning_effort": effort}
    if provider in ("deepseek", "zai"):
        return {"extra_body": {"thinking": {"type": "enabled"}}}
    return {}
