"""Complexity-based model routing with fallback chains."""

from __future__ import annotations

from loguru import logger

from src.config.model_registry import FALLBACK_CHAINS, MODEL_REGISTRY, ModelTier
from src.config.settings import Settings
from src.graph.enums import TaskComplexity


# Mapping from TaskComplexity to model tier and fallback chain key
COMPLEXITY_TIER_MAP: dict[TaskComplexity, tuple[ModelTier, str]] = {
    TaskComplexity.TRIVIAL: (ModelTier.VERY_CHEAP, "tier_0_micro"),
    TaskComplexity.SIMPLE: (ModelTier.CHEAP, "tier_1_standard"),
    TaskComplexity.COMPLEX: (ModelTier.CHEAP, "tier_1_standard"),
    TaskComplexity.CRITICAL: (ModelTier.MODERATE, "tier_2_reasoning"),
}


class ModelRouter:
    """Routes task complexity to appropriate model with fallback chain support."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._exclude_providers: set[str] = set()

    def route(
        self,
        complexity: TaskComplexity,
        exclude_providers: set[str] | None = None,
    ) -> str:
        """Select the best model for a given complexity level.

        Args:
            complexity: The task complexity classification.
            exclude_providers: Providers to skip (e.g., unhealthy ones).

        Returns:
            A model identifier string (litellm format).
        """
        tier, chain_key = COMPLEXITY_TIER_MAP.get(
            complexity, (ModelTier.CHEAP, "tier_1_standard")
        )
        excluded = (exclude_providers or set()) | self._exclude_providers

        model = self._route_from_chain(chain_key, excluded)
        if model:
            return model

        # Absolute fallback: find any available model in the tier
        for model_id, spec in MODEL_REGISTRY.items():
            if spec.tier == tier and self._extract_provider(model_id) not in excluded:
                logger.warning(f"Using fallback model {model_id} for complexity {complexity}")
                return model_id

        # Last resort: first model in registry
        first = next(iter(MODEL_REGISTRY))
        logger.error(f"All models exhausted for complexity {complexity}, using {first}")
        return first

    def get_fallback_chain(self, model: str) -> list[str]:
        """Get the fallback chain for a specific model.

        Args:
            model: The primary model identifier.

        Returns:
            List of fallback model identifiers.
        """
        return FALLBACK_CHAINS.get(model, [])

    def get_fallback_tier0(self) -> str:
        """Get a cheap fallback model."""
        chain = FALLBACK_CHAINS.get("gpt-4o-mini-2024-07-18", [])
        if chain:
            return chain[0]
        return "qwen3.5-flash"

    def mark_provider_unhealthy(self, provider: str) -> None:
        """Temporarily exclude a provider from routing."""
        self._exclude_providers.add(provider)
        logger.warning(f"Provider {provider} marked unhealthy, excluded from routing")

    def clear_provider_health(self, provider: str) -> None:
        """Re-enable a previously excluded provider."""
        self._exclude_providers.discard(provider)
        logger.info(f"Provider {provider} re-enabled for routing")

    def _route_from_chain(self, chain_key: str, exclude_providers: set[str]) -> str | None:
        """Try models in a fallback chain, skipping excluded providers."""
        chain = FALLBACK_CHAINS.get(chain_key, [])
        if not chain:
            # chain_key might be a model ID — try direct lookup
            chain = FALLBACK_CHAINS.get(chain_key, [])

        for model_id in chain:
            provider = self._extract_provider(model_id)
            if provider not in exclude_providers:
                # Verify we have API key for this provider
                if self._has_provider_key(provider):
                    return model_id
                logger.debug(f"Skipping {model_id}: no API key for {provider}")

        return None

    @staticmethod
    def _extract_provider(model: str) -> str:
        """Extract provider name from a litellm model identifier."""
        # litellm format: "provider/model-name" or "model-name"
        if "/" in model:
            return model.split("/")[0]
        # Known model prefixes
        if model.startswith("gpt-") or model.startswith("text-embedding-"):
            return "openai"
        if model.startswith("claude-"):
            return "anthropic"
        if model.startswith("deepseek-"):
            return "deepseek"
        if model.startswith("gemini-"):
            return "google"
        if model.startswith("mistral-") or model.startswith("ministral-") or model.startswith("open-mistral-"):
            return "mistral"
        if model.startswith("qwen"):
            return "qwen"
        if model.startswith("glm-"):
            return "zai"
        if model.startswith("kimi-") or model.startswith("moonshot-"):
            return "moonshot"
        if model.startswith("minimax-"):
            return "minimax"
        if model.startswith("llama-") or model.startswith("meta-llama/"):
            return "groq"
        return "unknown"

    def _has_provider_key(self, provider: str) -> bool:
        """Check if an API key is available for a provider."""
        try:
            return self._settings.llm.has_provider_key(provider)
        except Exception:
            return False  # Skip provider if settings access fails
