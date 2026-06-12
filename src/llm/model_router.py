"""Complexity-based model routing with fallback chains."""

from __future__ import annotations

from loguru import logger

from src.config.model_registry import FALLBACK_CHAINS, MODEL_REGISTRY, ModelTier
from src.config.settings import Settings
from src.graph.enums import TaskComplexity


# Mapping from TaskComplexity to model tier and fallback chain key
COMPLEXITY_TIER_MAP: dict[TaskComplexity, tuple[ModelTier, str]] = {
    TaskComplexity.TRIVIAL: (ModelTier.VERY_CHEAP, "qwen3.5-flash"),
    TaskComplexity.SIMPLE: (ModelTier.CHEAP, "claude-haiku-4-5-20251001"),
    TaskComplexity.COMPLEX: (ModelTier.CHEAP, "claude-haiku-4-5-20251001"),
    TaskComplexity.CRITICAL: (ModelTier.MODERATE, "claude-sonnet-4-6"),
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
            complexity, (ModelTier.CHEAP, "claude-haiku-4-5-20251001")
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
        chain = FALLBACK_CHAINS.get("qwen3.5-flash", [])
        if chain:
            return chain[0]
        return "qwen3.5-flash"

    def route_reasoning(self) -> str:
        """Select the configured reasoning model.

        Uses the ``reasoning_llm_model`` from settings (e.g. deepseek-v4-pro).
        Falls back to ``route(CRITICAL)`` when the provider has no API key.

        Returns:
            A model identifier string (litellm format).
        """
        model = self._settings.llm.reasoning_llm_model
        provider = self._extract_provider(model)
        if self._has_provider_key(provider):
            return model
        logger.warning(
            f"Reasoning model {model} provider {provider} has no API key, "
            f"falling back to CRITICAL routing"
        )
        return self.route(TaskComplexity.CRITICAL)

    def route_diverse(
        self,
        n: int,
        complexity: TaskComplexity,
        exclude_providers: set[str] | None = None,
    ) -> list[str]:
        """Return *n* models from different providers for a given complexity.

        Used when spawning parallel sub-agents to spread load across
        providers and avoid rate limits.  Falls back to cycling through
        whatever providers are available.

        Args:
            n: Number of distinct models to return.
            complexity: Task complexity level for tier selection.
            exclude_providers: Providers to skip.

        Returns:
            List of *n* model identifiers, one per provider where possible.
        """
        excluded = (exclude_providers or set()) | self._exclude_providers
        tier, chain_key = COMPLEXITY_TIER_MAP.get(
            complexity, (ModelTier.CHEAP, "claude-haiku-4-5-20251001")
        )

        # Collect one model per provider at the target tier
        provider_to_model: dict[str, str] = {}
        for model_id, spec in MODEL_REGISTRY.items():
            if spec.tier != tier:
                continue
            provider = self._extract_provider(model_id)
            if provider in excluded or provider in provider_to_model:
                continue
            if self._has_provider_key(provider):
                provider_to_model[provider] = model_id

        # Supplement from the fallback chain (may cross tiers)
        if len(provider_to_model) < n:
            for model_id in FALLBACK_CHAINS.get(chain_key, []):
                provider = self._extract_provider(model_id)
                if provider in excluded or provider in provider_to_model:
                    continue
                if self._has_provider_key(provider):
                    provider_to_model[provider] = model_id
                if len(provider_to_model) >= n:
                    break

        candidates = list(provider_to_model.values())

        if not candidates:
            # Absolute fallback: just repeat the default route
            return [self.route(complexity, exclude_providers)] * max(1, n)

        # Cycle through candidates to fill n slots
        result: list[str] = []
        for i in range(n):
            result.append(candidates[i % len(candidates)])
        return result

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
        # Registry key prefix for NVIDIA free-tier models
        if model.startswith("nvidia-"):
            return "nvidia"
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
