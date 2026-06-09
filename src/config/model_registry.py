"""
Model registry and fallback chains for the Turing Agent.

Defines ModelTier, ModelSpec, and the canonical MODEL_REGISTRY mapping
every supported model ID to its specification. FALLBACK_CHAINS provide
3-4 fallback models across different providers for resilience.
"""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple


class ModelTier(str, Enum):
    """Cost and performance tiers for model selection."""

    VERY_CHEAP = "very_cheap"  # Free tier, local models, ultra-cheap
    CHEAP = "cheap"  # Budget-conscious for high volume
    MODERATE = "moderate"  # Balanced cost/performance


class ModelSpec(NamedTuple):
    """Specification for an LLM model."""

    model_id: str  # LiteLLM-compatible model ID
    provider: str  # Provider name for API key lookup
    tier: ModelTier  # Cost tier
    max_context: int  # Context window in tokens
    max_output: int  # Maximum output tokens
    supports_tool_calling: bool
    supports_json_mode: bool
    supports_streaming: bool
    supports_images: bool


# ─── Canonical Model Registry ──────────────────────────────────────
# Derived from .claude/rules/llm-model-guardrails.md

MODEL_REGISTRY: dict[str, ModelSpec] = {
    # ── Anthropic ─────────────────────────────────────────────────
    "claude-haiku-4-5-20251001": ModelSpec(
        model_id="claude-haiku-4-5-20251001",
        provider="anthropic",
        tier=ModelTier.CHEAP,
        max_context=200_000,
        max_output=64_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=True,
    ),
    "claude-sonnet-4-6": ModelSpec(
        model_id="claude-sonnet-4-6",
        provider="anthropic",
        tier=ModelTier.MODERATE,
        max_context=1_000_000,
        max_output=128_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=True,
    ),
    # ── DeepSeek ──────────────────────────────────────────────────
    "deepseek-v4-flash": ModelSpec(
        model_id="deepseek/deepseek-v4-flash",
        provider="deepseek",
        tier=ModelTier.CHEAP,
        max_context=128_000,
        max_output=384_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=True,
    ),
    "deepseek-v4-pro": ModelSpec(
        model_id="deepseek/deepseek-v4-pro",
        provider="deepseek",
        tier=ModelTier.MODERATE,
        max_context=128_000,
        max_output=384_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=True,
    ),
    # ── OpenAI ────────────────────────────────────────────────────
    "gpt-4o-mini-2024-07-18": ModelSpec(
        model_id="gpt-4o-mini-2024-07-18",
        provider="openai",
        tier=ModelTier.VERY_CHEAP,
        max_context=128_000,
        max_output=16_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=True,
    ),
    "gpt-4.1-mini-2025-04-14": ModelSpec(
        model_id="gpt-4.1-mini-2025-04-14",
        provider="openai",
        tier=ModelTier.CHEAP,
        max_context=128_000,
        max_output=32_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=True,
    ),
    "gpt-5-nano-2025-08-07": ModelSpec(
        model_id="gpt-5-nano-2025-08-07",
        provider="openai",
        tier=ModelTier.CHEAP,
        max_context=128_000,
        max_output=128_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=True,
    ),
    "gpt-5-nano-2025-08-07": ModelSpec(
        model_id="gpt-5-nano-2025-08-07",
        provider="openai",
        tier=ModelTier.CHEAP,
        max_context=128_000,
        max_output=128_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=True,
    ),
    # ── Z.AI (Zhipu / GLM) ───────────────────────────────────────
    "glm-4.7-flash": ModelSpec(
        model_id="glm-4.7-flash",
        provider="zai",
        tier=ModelTier.VERY_CHEAP,
        max_context=128_000,
        max_output=131_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=False,
    ),
    "glm-4.5-air": ModelSpec(
        model_id="glm-4.5-air",
        provider="zai",
        tier=ModelTier.VERY_CHEAP,
        max_context=128_000,
        max_output=96_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=False,
    ),
    "glm-4.7": ModelSpec(
        model_id="glm-4.7",
        provider="zai",
        tier=ModelTier.MODERATE,
        max_context=128_000,
        max_output=128_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=True,
    ),
    "glm-5-turbo": ModelSpec(
        model_id="glm-5-turbo",
        provider="zai",
        tier=ModelTier.MODERATE,
        max_context=128_000,
        max_output=131_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=True,
    ),
    # ── MiniMax ───────────────────────────────────────────────────
    "minimax-m2.5-highspeed": ModelSpec(
        model_id="minimax-m2.5-highspeed",
        provider="minimax",
        tier=ModelTier.CHEAP,
        max_context=128_000,
        max_output=131_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=True,
    ),
    "minimax-m2.5": ModelSpec(
        model_id="minimax-m2.5",
        provider="minimax",
        tier=ModelTier.MODERATE,
        max_context=128_000,
        max_output=196_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=True,
    ),
    # ── Mistral ───────────────────────────────────────────────────
    "mistral-small-2603": ModelSpec(
        model_id="mistral/mistral-small-2603",
        provider="mistral",
        tier=ModelTier.CHEAP,
        max_context=64_000,
        max_output=256_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=False,
    ),
    "mistral-medium-3-5": ModelSpec(
        model_id="mistral/mistral-medium-3-5",
        provider="mistral",
        tier=ModelTier.MODERATE,
        max_context=128_000,
        max_output=262_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=True,
    ),
    # ── Moonshot ──────────────────────────────────────────────────
    "moonshot-v1-32k": ModelSpec(
        model_id="moonshot-v1-32k",
        provider="moonshot",
        tier=ModelTier.CHEAP,
        max_context=32_000,
        max_output=32_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=False,
    ),
    "kimi-k2.6": ModelSpec(
        model_id="kimi-k2.6",
        provider="moonshot",
        tier=ModelTier.MODERATE,
        max_context=128_000,
        max_output=262_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=True,
    ),
    # ── Qwen (Alibaba DashScope) ─────────────────────────────────
    "qwen3.5-flash": ModelSpec(
        model_id="openai/qwen3.5-flash",
        provider="dashscope",
        tier=ModelTier.VERY_CHEAP,
        max_context=1_000_000,
        max_output=66_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=True,
    ),
    "qwen3.7-plus": ModelSpec(
        model_id="openai/qwen3.7-plus",
        provider="dashscope",
        tier=ModelTier.CHEAP,
        max_context=1_000_000,
        max_output=65_500,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=True,
    ),
    # ── Google ────────────────────────────────────────────────────
    "gemini-2.5-flash-lite": ModelSpec(
        model_id="gemini/gemini-2.5-flash-lite",
        provider="google",
        tier=ModelTier.VERY_CHEAP,
        max_context=1_000_000,
        max_output=65_500,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=True,
    ),
    "gemini-2.5-flash": ModelSpec(
        model_id="gemini/gemini-2.5-flash",
        provider="google",
        tier=ModelTier.CHEAP,
        max_context=1_000_000,
        max_output=65_500,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=True,
    ),
    "gemini-3-flash-preview": ModelSpec(
        model_id="gemini/gemini-3-flash-preview",
        provider="google",
        tier=ModelTier.MODERATE,
        max_context=1_000_000,
        max_output=65_500,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=True,
    ),
    # ── Groq ──────────────────────────────────────────────────────
    "llama-3.1-8b-instant": ModelSpec(
        model_id="groq/llama-3.1-8b-instant",
        provider="groq",
        tier=ModelTier.VERY_CHEAP,
        max_context=128_000,
        max_output=8_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=False,
    ),
    "llama-3.3-70b-versatile": ModelSpec(
        model_id="groq/llama-3.3-70b-versatile",
        provider="groq",
        tier=ModelTier.MODERATE,
        max_context=128_000,
        max_output=32_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=True,
    ),
    # ── OpenRouter (Free Tier) ────────────────────────────────────
    "openrouter/qwen/qwen3-next-80b-a3b-instruct:free": ModelSpec(
        model_id="openrouter/qwen/qwen3-next-80b-a3b-instruct:free",
        provider="openrouter",
        tier=ModelTier.VERY_CHEAP,
        max_context=128_000,
        max_output=262_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=False,
    ),
    # ── Ollama (Local) ────────────────────────────────────────────
    "ollama/qwen3.5:latest": ModelSpec(
        model_id="ollama/qwen3.5:latest",
        provider="ollama",
        tier=ModelTier.VERY_CHEAP,
        max_context=128_000,
        max_output=262_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=False,
    ),
}


# ─── Fallback Chains ───────────────────────────────────────────────
# Each primary model maps to 3-4 fallback models across DIFFERENT providers.
# The gateway walks this chain on failure (rate limit, timeout, server error).

FALLBACK_CHAINS: dict[str, list[str]] = {
    # ── Tier 0 (Very Cheap) fallbacks ─────────────────────────────
    "qwen3.5-flash": [
        "gpt-4o-mini-2024-07-18",
        "glm-4.7-flash",
        "llama-3.1-8b-instant",
        "gemini-2.5-flash-lite",
    ],
    "gpt-4o-mini-2024-07-18": [
        "qwen3.5-flash",
        "glm-4.7-flash",
        "llama-3.1-8b-instant",
    ],
    "glm-4.7-flash": [
        "qwen3.5-flash",
        "gpt-4o-mini-2024-07-18",
        "llama-3.1-8b-instant",
    ],
    "llama-3.1-8b-instant": [
        "qwen3.5-flash",
        "gpt-4o-mini-2024-07-18",
        "glm-4.7-flash",
    ],
    "gemini-2.5-flash-lite": [
        "qwen3.5-flash",
        "gpt-4o-mini-2024-07-18",
        "glm-4.7-flash",
    ],
    # ── Tier 1 (Cheap) fallbacks ──────────────────────────────────
    "claude-haiku-4-5-20251001": [
        "deepseek-v4-flash",
        "qwen3.7-plus",
        "glm-4.5-air",
        "gemini-2.5-flash",
    ],
    "deepseek-v4-flash": [
        "claude-haiku-4-5-20251001",
        "qwen3.7-plus",
        "glm-4.5-air",
        "gpt-4.1-mini-2025-04-14",
    ],
    "qwen3.7-plus": [
        "deepseek-v4-flash",
        "claude-haiku-4-5-20251001",
        "glm-4.5-air",
        "gemini-2.5-flash",
    ],
    "gpt-4.1-mini-2025-04-14": [
        "deepseek-v4-flash",
        "claude-haiku-4-5-20251001",
        "qwen3.7-plus",
        "glm-4.5-air",
    ],
    "gemini-2.5-flash": [
        "claude-haiku-4-5-20251001",
        "deepseek-v4-flash",
        "qwen3.7-plus",
    ],
    # ── Tier 2 (Moderate) fallbacks ───────────────────────────────
    "claude-sonnet-4-6": [
        "deepseek-v4-pro",
        "glm-5-turbo",
        "mistral-medium-3-5",
        "gemini-3-flash-preview",
    ],
    "deepseek-v4-pro": [
        "claude-sonnet-4-6",
        "glm-5-turbo",
        "mistral-medium-3-5",
        "kimi-k2.6",
    ],
    "glm-5-turbo": [
        "claude-sonnet-4-6",
        "deepseek-v4-pro",
        "mistral-medium-3-5",
        "gemini-3-flash-preview",
    ],
    "mistral-medium-3-5": [
        "claude-sonnet-4-6",
        "deepseek-v4-pro",
        "glm-5-turbo",
        "kimi-k2.6",
    ],
    "gemini-3-flash-preview": [
        "claude-sonnet-4-6",
        "deepseek-v4-pro",
        "glm-5-turbo",
    ],
    "kimi-k2.6": [
        "deepseek-v4-pro",
        "claude-sonnet-4-6",
        "glm-5-turbo",
    ],
    "gpt-5-nano-2025-08-07": [
        "deepseek-v4-flash",
        "claude-haiku-4-5-20251001",
        "qwen3.7-plus",
    ],
}


def get_fallback_chain(model_id: str) -> list[str]:
    """Get the fallback chain for a given model.

    Returns the fallback chain if defined, otherwise returns models
    in the same tier from different providers.
    """
    if model_id in FALLBACK_CHAINS:
        return FALLBACK_CHAINS[model_id]

    # Dynamic fallback: find models in the same tier from different providers
    if model_id not in MODEL_REGISTRY:
        return []

    spec = MODEL_REGISTRY[model_id]
    same_tier = [
        mid
        for mid, mspec in MODEL_REGISTRY.items()
        if mspec.tier == spec.tier and mspec.provider != spec.provider
    ]
    return same_tier[:4]


def get_models_by_tier(tier: ModelTier) -> list[ModelSpec]:
    """Get all models in a given tier."""
    return [spec for spec in MODEL_REGISTRY.values() if spec.tier == tier]


def get_model_spec(model_id: str) -> ModelSpec | None:
    """Get the ModelSpec for a model, or None if not found."""
    return MODEL_REGISTRY.get(model_id)
