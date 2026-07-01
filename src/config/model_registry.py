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
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0


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
        input_cost_per_1k=0.001,
        output_cost_per_1k=0.005,
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
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
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
        supports_images=False,  # DeepSeek V4 is text-only (live-probed 2026-06-26: provider drops image_url blocks)
        input_cost_per_1k=0.0001,
        output_cost_per_1k=0.0004,
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
        supports_images=False,  # DeepSeek V4 is text-only (live-probed 2026-06-26: provider drops image_url blocks)
        input_cost_per_1k=0.0005,
        output_cost_per_1k=0.002,
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
        input_cost_per_1k=0.00015,
        output_cost_per_1k=0.0006,
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
        input_cost_per_1k=0.0004,
        output_cost_per_1k=0.0016,
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
        # Live-probed 2026-06-26: image_url block is dropped (in≈25 tokens, no
        # image processing) and the model guesses the color ("pink"/"blue" at
        # temp=0). Effectively text-only for vision INPUT despite doc claims.
        supports_images=False,
        input_cost_per_1k=0.0001,
        output_cost_per_1k=0.0004,
    ),
    # ── Z.AI (Zhipu / GLM) ───────────────────────────────────────
    "glm-4.7-flash": ModelSpec(
        model_id="zai/glm-4.7-flash",
        provider="zai",
        tier=ModelTier.VERY_CHEAP,
        max_context=128_000,
        max_output=131_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=False,
        input_cost_per_1k=0.00005,
        output_cost_per_1k=0.0001,
    ),
    "glm-4.5-air": ModelSpec(
        model_id="zai/glm-4.5-air",
        provider="zai",
        tier=ModelTier.VERY_CHEAP,
        max_context=128_000,
        max_output=96_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=False,
        input_cost_per_1k=0.0001,
        output_cost_per_1k=0.0002,
    ),
    "glm-4.7": ModelSpec(
        model_id="zai/glm-4.7",
        provider="zai",
        tier=ModelTier.MODERATE,
        max_context=128_000,
        max_output=128_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=True,
        input_cost_per_1k=0.0005,
        output_cost_per_1k=0.001,
    ),
    "glm-5-turbo": ModelSpec(
        model_id="zai/glm-5-turbo",
        provider="zai",
        tier=ModelTier.MODERATE,
        max_context=128_000,
        max_output=131_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=True,
        input_cost_per_1k=0.001,
        output_cost_per_1k=0.003,
    ),
    # glm-5.1 / glm-5.2 — Z.AI-native (distinct from the NVIDIA-hosted FREE
    # copy ``nvidia-glm-5-1``). Added 2026-07-01 to back the plan-node routing
    # (glm-5.1 is the strong planner primary; glm-5.2 is a 1M-context peer).
    # Caps per llm-model-guardrails.md:35-36 (both Moderate; glm-5.1 200K/131K,
    # glm-5.2 1M/128K; text INPUT only — no vision INPUT). PRICING IS A
    # CONSERVATIVE MIRROR OF glm-5-turbo (0.001/0.003 per 1K) pending the exact
    # Z.AI per-token rates — the owner should confirm and correct if the ledger
    # must be exact. Both ALLOWED (NOT blocked) per the repo guardrails.
    "glm-5.1": ModelSpec(
        model_id="zai/glm-5.1",
        provider="zai",
        tier=ModelTier.MODERATE,
        max_context=200_000,
        max_output=131_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=False,
        input_cost_per_1k=0.001,
        output_cost_per_1k=0.003,
    ),
    "glm-5.2": ModelSpec(
        model_id="zai/glm-5.2",
        provider="zai",
        tier=ModelTier.MODERATE,
        max_context=1_000_000,
        max_output=128_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=False,
        input_cost_per_1k=0.001,
        output_cost_per_1k=0.003,
    ),
    # ── MiniMax ───────────────────────────────────────────────────
    "minimax-m2.5-highspeed": ModelSpec(
        model_id="minimax/minimax-m2.5-highspeed",
        provider="minimax",
        tier=ModelTier.CHEAP,
        max_context=128_000,
        max_output=131_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        # Live-probed 2026-06-26: "there is no image provided" — MiniMax drops
        # the image_url block (text-only for vision INPUT).
        supports_images=False,
        input_cost_per_1k=0.0001,
        output_cost_per_1k=0.0002,
    ),
    "minimax-m2.5": ModelSpec(
        model_id="minimax/minimax-m2.5",
        provider="minimax",
        tier=ModelTier.MODERATE,
        max_context=128_000,
        max_output=196_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        # Live-probed 2026-06-26: "there is no image provided" — MiniMax drops
        # the image_url block (text-only for vision INPUT).
        supports_images=False,
        input_cost_per_1k=0.0005,
        output_cost_per_1k=0.001,
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
        input_cost_per_1k=0.0002,
        output_cost_per_1k=0.0006,
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
        input_cost_per_1k=0.002,
        output_cost_per_1k=0.006,
    ),
    # ── Moonshot ──────────────────────────────────────────────────
    "moonshot-v1-32k": ModelSpec(
        model_id="moonshot/moonshot-v1-32k",
        provider="moonshot",
        tier=ModelTier.CHEAP,
        max_context=32_000,
        max_output=32_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=False,
        input_cost_per_1k=0.0002,
        output_cost_per_1k=0.0004,
    ),
    "kimi-k2.6": ModelSpec(
        model_id="moonshot/kimi-k2.6",
        provider="moonshot",
        tier=ModelTier.MODERATE,
        max_context=128_000,
        max_output=262_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=True,
        input_cost_per_1k=0.001,
        output_cost_per_1k=0.003,
    ),
    # ── Qwen (Alibaba DashScope) ─────────────────────────────────
    "qwen3.5-flash": ModelSpec(
        model_id="openai/qwen3.5-flash",
        provider="alibaba",
        tier=ModelTier.VERY_CHEAP,
        max_context=1_000_000,
        # Qwen/DashScope hard-rejects max_tokens > 65536 (API error:
        # "Range of max_tokens should be [1, 65536]"). The gateway sends
        # spec.max_output as max_tokens on calls that don't override it, so this
        # MUST stay at/below that API cap. Verified live 2026-06.
        max_output=65_536,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=True,
        input_cost_per_1k=0.00005,
        output_cost_per_1k=0.0001,
    ),
    "qwen3.7-plus": ModelSpec(
        model_id="openai/qwen3.7-plus",
        provider="alibaba",
        tier=ModelTier.CHEAP,
        max_context=1_000_000,
        max_output=65_500,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=True,
        input_cost_per_1k=0.0003,
        output_cost_per_1k=0.0006,
    ),
    # ── Qwen (Alibaba DashScope) — expanded flash/turbo/coder pool ────────
    # Each was live-verified responsive via `python main.py --verify-models`
    # (dashscope-intl OpenAI-compatible endpoint, DASHSCOPE_API_KEY) before
    # landing. Broadening the alibaba pool gives provider-diverse fallback when
    # a single provider caps/burns balance — the recurring blocker that killed
    # battery-04 q08 run2. NOTE: every entry MUST keep max_output <= 65536:
    # DashScope hard-rejects max_tokens above that ("Range of max_tokens should
    # be [1, 65536]"); enforced by tests/test_config/test_alibaba_models.py.
    "qwen3.6-flash": ModelSpec(
        model_id="openai/qwen3.6-flash",
        provider="alibaba",
        tier=ModelTier.VERY_CHEAP,
        max_context=1_000_000,
        max_output=65_536,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=True,
        # flash-tier pricing, consistent with the qwen3.5-flash sibling.
        input_cost_per_1k=0.00005,
        output_cost_per_1k=0.0001,
    ),
    "qwen3.5-flash-2026-02-23": ModelSpec(
        # Pinned snapshot of qwen3.5-flash (same model, frozen to the
        # 2026-02-23 release) — identical spec, distinct model_id for callers
        # that need reproducibility across the rolling alias.
        model_id="openai/qwen3.5-flash-2026-02-23",
        provider="alibaba",
        tier=ModelTier.VERY_CHEAP,
        max_context=1_000_000,
        max_output=65_536,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=True,
        input_cost_per_1k=0.00005,
        output_cost_per_1k=0.0001,
    ),
    "qwen-turbo": ModelSpec(
        model_id="openai/qwen-turbo",
        provider="alibaba",
        tier=ModelTier.CHEAP,
        max_context=1_000_000,
        # Classic qwen-turbo output cap; conservative (undershooting output is
        # safe — overshooting risks a max_tokens-rejection 400).
        max_output=8_192,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        # Live-probed 2026-06-26: classic qwen-turbo drops the image_url block
        # (in≈32 tokens, no image processing) and guesses the color — text-only.
        supports_images=False,
        # Approximate DashScope pricing (cost estimate only).
        input_cost_per_1k=0.0001,
        output_cost_per_1k=0.0002,
    ),
    "qwen3-coder-next": ModelSpec(
        model_id="openai/qwen3-coder-next",
        provider="alibaba",
        tier=ModelTier.CHEAP,
        max_context=1_000_000,
        max_output=65_536,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=False,
        # Approximate DashScope pricing (cost estimate only).
        input_cost_per_1k=0.0001,
        output_cost_per_1k=0.0003,
    ),
    "alibaba-deepseek-v4-flash": ModelSpec(
        # DeepSeek's deepseek-v4-flash HOSTED ON DashScope — distinct from the
        # standalone DeepSeek provider's "deepseek-v4-flash" (different provider
        # + API key + quota). When the standalone DEEPSEEK_API_KEY hits
        # "Insufficient Balance", this DashScope copy can serve the same model
        # — provider-diversity on an identical model. See FALLBACK_CHAINS where
        # the standalone chain reaches this entry first.
        model_id="openai/deepseek-v4-flash",
        provider="alibaba",
        tier=ModelTier.CHEAP,
        max_context=128_000,
        max_output=65_536,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=False,  # same text-only model as standalone deepseek-v4-flash
        # Mirrors the standalone deepseek-v4-flash pricing (same model).
        input_cost_per_1k=0.0001,
        output_cost_per_1k=0.0004,
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
        # Real Google pricing (verified via litellm model_cost): $0.10/$0.40 per 1M.
        input_cost_per_1k=0.0001,
        output_cost_per_1k=0.0004,
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
        input_cost_per_1k=0.00015,
        output_cost_per_1k=0.0006,
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
        input_cost_per_1k=0.00015,
        output_cost_per_1k=0.0006,
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
        # Real Groq pricing (verified via litellm model_cost): $0.05/$0.08 per 1M.
        input_cost_per_1k=0.00005,
        output_cost_per_1k=0.00008,
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
        # Live-probed 2026-06-26: Groq rejects multimodal content outright
        # ("messages[0].content must be a string") — Llama 3.3 70B is text-only.
        supports_images=False,
        # Real Groq pricing (verified via litellm model_cost): $0.59/$0.79 per 1M.
        input_cost_per_1k=0.00059,
        output_cost_per_1k=0.00079,
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
        # OpenRouter free tier — $0.0 (explicit so it is never mispriced as
        # fallback by calculate_cost).
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
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
        # Locally hosted — $0.0 (explicit so it is never mispriced as fallback).
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
    ),
    # ── NVIDIA (Free Tier via build.nvidia.com) ──────────────────
    # All models are free on the NVIDIA API (accessed via NVIDIA_API_KEY), so
    # every entry below carries the default input_cost_per_1k=0.0 /
    # output_cost_per_1k=0.0. CostTracker.calculate_cost prices registered
    # models from their explicit fields, so these cost $0.0 — never the generic
    # fallback rate that previously inflated spend on free-tier calls.
    # litellm model_id format: nvidia/<nvidia-api-model-id>
    "nvidia-nemotron-super-120b": ModelSpec(
        model_id="nvidia/nvidia/nemotron-3-super-120b-a12b",
        provider="nvidia",
        tier=ModelTier.VERY_CHEAP,
        max_context=128_000,
        max_output=65_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=False,
    ),
    "nvidia-nemotron-ultra-550b": ModelSpec(
        model_id="nvidia/nvidia/nemotron-3-ultra-550b-a55b",
        provider="nvidia",
        tier=ModelTier.VERY_CHEAP,
        max_context=128_000,
        max_output=65_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=False,
    ),
    "nvidia-nemotron-super-49b": ModelSpec(
        model_id="nvidia/nvidia/llama-3.3-nemotron-super-49b-v1",
        provider="nvidia",
        tier=ModelTier.VERY_CHEAP,
        max_context=128_000,
        max_output=32_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=False,
    ),
    "nvidia-nemotron-super-49b-v1.5": ModelSpec(
        model_id="nvidia/nvidia/llama-3.3-nemotron-super-49b-v1.5",
        provider="nvidia",
        tier=ModelTier.VERY_CHEAP,
        max_context=128_000,
        max_output=32_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=False,
    ),
    "nvidia-gpt-oss-120b": ModelSpec(
        model_id="nvidia/openai/gpt-oss-120b",
        provider="nvidia",
        tier=ModelTier.VERY_CHEAP,
        max_context=128_000,
        max_output=65_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=False,
    ),
    "nvidia-qwen3-next-80b": ModelSpec(
        model_id="nvidia/qwen/qwen3-next-80b-a3b-instruct",
        provider="nvidia",
        tier=ModelTier.VERY_CHEAP,
        max_context=128_000,
        max_output=65_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=False,
    ),
    "nvidia-qwen3.5-397b": ModelSpec(
        model_id="nvidia/qwen/qwen3.5-397b-a17b",
        provider="nvidia",
        tier=ModelTier.VERY_CHEAP,
        max_context=128_000,
        max_output=65_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=False,
    ),
    "nvidia-qwen3.5-122b": ModelSpec(
        model_id="nvidia/qwen/qwen3.5-122b-a10b",
        provider="nvidia",
        tier=ModelTier.VERY_CHEAP,
        max_context=128_000,
        max_output=65_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=False,
    ),
    # NVIDIA free tier — models from other providers hosted on NVIDIA API
    # EXPENSIVE_MODEL: explicitly requested — free via NVIDIA API, not paid provider
    "nvidia-glm-5-1": ModelSpec(
        model_id="nvidia/z-ai/glm-5.1",
        provider="nvidia",
        tier=ModelTier.VERY_CHEAP,
        max_context=128_000,
        max_output=65_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=False,
    ),
    "nvidia-llama-3.3-70b": ModelSpec(
        model_id="nvidia/meta/llama-3.3-70b-instruct",
        provider="nvidia",
        tier=ModelTier.VERY_CHEAP,
        max_context=128_000,
        max_output=32_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=False,
    ),
    "nvidia-llama-3.2-90b-vision": ModelSpec(
        model_id="nvidia/meta/llama-3.2-90b-vision-instruct",
        provider="nvidia",
        tier=ModelTier.VERY_CHEAP,
        max_context=128_000,
        max_output=32_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=True,
    ),
    # NVIDIA free tier — DeepSeek on NVIDIA API
    "nvidia-deepseek-v4-flash": ModelSpec(
        model_id="nvidia/deepseek-ai/deepseek-v4-flash",
        provider="nvidia",
        tier=ModelTier.VERY_CHEAP,
        max_context=128_000,
        max_output=65_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=False,
    ),
    "nvidia-deepseek-v4-pro": ModelSpec(
        model_id="nvidia/deepseek-ai/deepseek-v4-pro",
        provider="nvidia",
        tier=ModelTier.VERY_CHEAP,
        max_context=128_000,
        max_output=65_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=False,
    ),
    # NVIDIA free tier — MiniMax on NVIDIA API
    # EXPENSIVE_MODEL: explicitly requested — free via NVIDIA API, not paid provider
    "nvidia-minimax-m2-7": ModelSpec(
        model_id="nvidia/minimaxai/minimax-m2.7",
        provider="nvidia",
        tier=ModelTier.VERY_CHEAP,
        max_context=128_000,
        max_output=65_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=False,
    ),
    "nvidia-kimi-k2.6": ModelSpec(
        model_id="nvidia/moonshotai/kimi-k2.6",
        provider="nvidia",
        tier=ModelTier.VERY_CHEAP,
        max_context=128_000,
        max_output=65_000,
        supports_tool_calling=True,
        supports_json_mode=True,
        supports_streaming=True,
        supports_images=False,
    ),
    "nvidia-step-3.5-flash": ModelSpec(
        model_id="nvidia/stepfun-ai/step-3.5-flash",
        provider="nvidia",
        tier=ModelTier.VERY_CHEAP,
        max_context=128_000,
        max_output=65_000,
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
        "qwen3.6-flash",
        "gpt-4o-mini-2024-07-18",
        "glm-4.7-flash",
        "nvidia-qwen3-next-80b",
        "llama-3.1-8b-instant",
    ],
    "qwen3.6-flash": [
        "qwen3.5-flash",
        "gpt-4o-mini-2024-07-18",
        "glm-4.7-flash",
        "llama-3.1-8b-instant",
    ],
    "qwen3.5-flash-2026-02-23": [
        "qwen3.5-flash",
        "qwen3.6-flash",
        "gpt-4o-mini-2024-07-18",
        "glm-4.7-flash",
    ],
    "gpt-4o-mini-2024-07-18": [
        "qwen3.5-flash",
        "nvidia-nemotron-ultra-550b",
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
        # Reach the DashScope-hosted copy FIRST: identical model, independent
        # quota — directly survives a standalone "Insufficient Balance" (the
        # blocker that killed battery-04 q08 run2) by serving the same model
        # via DASHSCOPE_API_KEY instead.
        "alibaba-deepseek-v4-flash",
        "claude-haiku-4-5-20251001",
        "nvidia-deepseek-v4-flash",
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
    "qwen-turbo": [
        "qwen3.5-flash",
        "qwen3.6-flash",
        "deepseek-v4-flash",
        "gpt-4o-mini-2024-07-18",
    ],
    "qwen3-coder-next": [
        "deepseek-v4-flash",
        "qwen3.5-flash",
        "gpt-4.1-mini-2025-04-14",
        "glm-4.5-air",
    ],
    "alibaba-deepseek-v4-flash": [
        # Mirror image of the standalone chain: try the standalone provider
        # first (same model, different quota), then cross-provider peers.
        "deepseek-v4-flash",
        "qwen3.5-flash",
        "claude-haiku-4-5-20251001",
        "glm-4.5-air",
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
    # glm-4.7 is the CRITICAL/reasoning primary (zai) — same Moderate-tier peer
    # set as Sonnet's chain, minus the (quota-blocked) Anthropic entry. Verified
    # live: zai/glm-4.7 returns completions with the funded ZAI_API_KEY.
    "glm-4.7": [
        "deepseek-v4-pro",
        "glm-5-turbo",
        "mistral-medium-3-5",
        "gemini-3-flash-preview",
    ],
    "deepseek-v4-pro": [
        "claude-sonnet-4-6",
        "nvidia-deepseek-v4-pro",
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
    # Plan-node primary (findings-06 retier / phase-2). Order matches the owner
    # spec: claude-sonnet-4-6 first, then glm-4.7. The gateway pre-filters a
    # provider with no API key (gateway.py ``_execute_with_fallback``), so while
    # anthropic is disabled claude-sonnet-4-6 is SKIPPED (no wasted 400) and
    # glm-4.7 — a strong live execute model — is attempted. Once the Anthropic
    # key recovers claude-sonnet-4-6 becomes the first live fallback. deepseek-
    # v4-pro (live reasoning) and glm-5-turbo (same-family peer) round it out.
    "glm-5.1": [
        "claude-sonnet-4-6",
        "glm-4.7",
        "deepseek-v4-pro",
        "glm-5-turbo",
    ],
    # 1M-context planner peer; falls to its same-family successor first, then the
    # same cross-provider net as glm-5.1.
    "glm-5.2": [
        "glm-5.1",
        "claude-sonnet-4-6",
        "glm-4.7",
        "deepseek-v4-pro",
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
    # ── NVIDIA Free Tier fallbacks ────────────────────────────────
    # Each NVIDIA model falls back to other NVIDIA models + non-NVIDIA providers
    "nvidia-nemotron-super-120b": [
        "nvidia-nemotron-ultra-550b",
        "nvidia-gpt-oss-120b",
        "gpt-4o-mini-2024-07-18",
    ],
    "nvidia-nemotron-ultra-550b": [
        "nvidia-nemotron-super-120b",
        "nvidia-gpt-oss-120b",
        "qwen3.5-flash",
    ],
    "nvidia-nemotron-super-49b": [
        "nvidia-nemotron-super-49b-v1.5",
        "nvidia-llama-3.3-70b",
        "gpt-4o-mini-2024-07-18",
    ],
    "nvidia-nemotron-super-49b-v1.5": [
        "nvidia-nemotron-super-49b",
        "nvidia-llama-3.3-70b",
        "qwen3.5-flash",
    ],
    "nvidia-gpt-oss-120b": [
        "nvidia-nemotron-super-120b",
        "nvidia-qwen3.5-397b",
        "gpt-4o-mini-2024-07-18",
    ],
    "nvidia-qwen3-next-80b": [
        "nvidia-qwen3.5-122b",
        "nvidia-llama-3.3-70b",
        "qwen3.5-flash",
    ],
    "nvidia-qwen3.5-397b": [
        "nvidia-qwen3.5-122b",
        "nvidia-gpt-oss-120b",
        "gpt-4o-mini-2024-07-18",
    ],
    "nvidia-qwen3.5-122b": [
        "nvidia-qwen3-next-80b",
        "nvidia-llama-3.3-70b",
        "qwen3.5-flash",
    ],
    "nvidia-glm-5-1": [
        "nvidia-llama-3.3-70b",
        "nvidia-qwen3-next-80b",
        "glm-4.7-flash",
    ],
    "nvidia-llama-3.3-70b": [
        "nvidia-nemotron-super-49b",
        "nvidia-qwen3-next-80b",
        "llama-3.1-8b-instant",
    ],
    "nvidia-llama-3.2-90b-vision": [
        "nvidia-llama-3.3-70b",
        "nvidia-nemotron-super-49b",
        "gemini-2.5-flash-lite",
    ],
    "nvidia-deepseek-v4-flash": [
        "nvidia-deepseek-v4-pro",
        "nvidia-qwen3-next-80b",
        "deepseek-v4-flash",
    ],
    "nvidia-deepseek-v4-pro": [
        "nvidia-deepseek-v4-flash",
        "nvidia-nemotron-super-120b",
        "deepseek-v4-pro",
    ],
    "nvidia-minimax-m2-7": [
        "nvidia-kimi-k2.6",
        "nvidia-qwen3.5-397b",
        "qwen3.5-flash",
    ],
    "nvidia-kimi-k2.6": [
        "nvidia-minimax-m2-7",
        "nvidia-qwen3.5-397b",
        "kimi-k2.6",
    ],
    "nvidia-step-3.5-flash": [
        "nvidia-qwen3-next-80b",
        "nvidia-llama-3.3-70b",
        "qwen3.5-flash",
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
