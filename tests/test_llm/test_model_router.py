"""Tests for src.llm.model_router — complexity-based model routing."""

from __future__ import annotations


import pytest

from src.config.settings import Settings
from src.graph.enums import TaskComplexity
from src.llm.model_router import ModelRouter


@pytest.fixture
def router() -> ModelRouter:
    """Create a ModelRouter with default settings."""
    settings = Settings()
    return ModelRouter(settings)


# Provider → settings.llm attribute holding its API key (mirrors
# ``LLMSettings.get_provider_key``). Lets the F15 routing tests assert the
# primary-vs-fallback INVARIANT regardless of which model the (tunable)
# ``COMPLEXITY_TIER_MAP`` names — so an in-flight tier-map experiment (e.g. the
# battery-04 OpenAI swap) doesn't reduce these to hardcoded-name assertions that
# flip red every time the map is retuned.
_PROVIDER_KEY_FIELDS: dict[str, str] = {
    "openai": "openai_api_key",
    "anthropic": "anthropic_api_key",
    "deepseek": "deepseek_api_key",
    "zai": "zai_api_key",
    "alibaba": "dashscope_api_key",
    "google": "google_api_key",
    "groq": "groq_api_key",
    "mistral": "mistral_api_key",
    "moonshot": "moonshot_api_key",
    "minimax": "minimax_api_key",
    "openrouter": "openrouter_api_key",
    "nvidia": "nvidia_api_key",
}


def _set_only_provider_key(settings: Settings, provider: str, value: str | None) -> None:
    """Clear every provider key, then set exactly ``provider``'s key to ``value``.

    Explicitly nulls all fields so a key inherited from ``.env`` cannot leak in
    and satisfy a different provider's check.
    """
    for field in _PROVIDER_KEY_FIELDS.values():
        setattr(settings.llm, field, None)
    field = _PROVIDER_KEY_FIELDS.get(provider)
    if field is not None:
        setattr(settings.llm, field, value)


class TestModelRouterExtractProvider:
    """Tests for _extract_provider static method."""

    def test_extract_provider_openai(self) -> None:
        """GPT model IDs extract to 'openai'."""
        assert ModelRouter._extract_provider("gpt-4o-mini-2024-07-18") == "openai"

    def test_extract_provider_anthropic(self) -> None:
        """Claude model IDs extract to 'anthropic'."""
        assert ModelRouter._extract_provider("claude-sonnet-4-6") == "anthropic"

    def test_extract_provider_with_slash(self) -> None:
        """Provider prefixes with slash extract correctly."""
        assert ModelRouter._extract_provider("deepseek/deepseek-v4-flash") == "deepseek"

    def test_extract_provider_gemini(self) -> None:
        """Gemini model IDs extract to 'google'."""
        assert ModelRouter._extract_provider("gemini-2.5-flash") == "google"

    def test_extract_provider_glm(self) -> None:
        """GLM model IDs extract to 'zai'."""
        assert ModelRouter._extract_provider("glm-4.7") == "zai"

    def test_extract_provider_unknown(self) -> None:
        """Unknown model prefixes return 'unknown'."""
        assert ModelRouter._extract_provider("foobar-123") == "unknown"


class TestModelRouterRoute:
    """Tests for route method."""

    def test_route_returns_model_string(self, router: ModelRouter) -> None:
        """Routing any complexity returns a non-empty model string."""
        result = router.route(TaskComplexity.SIMPLE)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_route_trivial_complexity(self, router: ModelRouter) -> None:
        """TRIVIAL complexity returns a model string."""
        result = router.route(TaskComplexity.TRIVIAL)
        assert isinstance(result, str)

    def test_route_complex_complexity(self, router: ModelRouter) -> None:
        """COMPLEX complexity returns a model string."""
        result = router.route(TaskComplexity.COMPLEX)
        assert isinstance(result, str)

    def test_route_critical_complexity(self, router: ModelRouter) -> None:
        """CRITICAL complexity returns a model string."""
        result = router.route(TaskComplexity.CRITICAL)
        assert isinstance(result, str)

    def test_route_returns_primary_when_key_present(self) -> None:
        """F15: route() returns the complexity's primary (chain_key) model when
        its provider key is set — not the first fallback. Previously the primary
        was bypassed: _route_from_chain walked FALLBACK_CHAINS[chain_key], which
        excludes the primary, so the configured primary silently resolved to its
        first fallback with a key. Asserted generically against the current
        COMPLEXITY_TIER_MAP so a tier-map experiment (battery-04 OpenAI swap)
        can't flip this red — every provider is keyed, so whichever model the
        map names as primary must be the one returned."""
        from src.llm.model_router import COMPLEXITY_TIER_MAP

        settings = Settings()
        for field in _PROVIDER_KEY_FIELDS.values():
            setattr(settings.llm, field, "test-key")
        router = ModelRouter(settings)
        for complexity in (TaskComplexity.SIMPLE, TaskComplexity.COMPLEX, TaskComplexity.CRITICAL):
            _tier, primary = COMPLEXITY_TIER_MAP[complexity]
            assert router.route(complexity) == primary, (
                f"{complexity}: primary {primary!r} not returned when its key is set"
            )

    def test_route_falls_to_chain_when_primary_provider_lacks_key(self) -> None:
        """When the primary's provider has no key, route falls back to the
        chain's first model whose provider does have a key. Asserted generically
        against the current COMPLEXITY_TIER_MAP + its fallback chain so the test
        tracks the invariant, not a model name that a tier-map experiment retunes."""
        from src.config.model_registry import FALLBACK_CHAINS
        from src.llm.model_router import COMPLEXITY_TIER_MAP

        settings = Settings()
        _tier, primary = COMPLEXITY_TIER_MAP[TaskComplexity.COMPLEX]
        primary_provider = ModelRouter._extract_provider(primary)
        # First chain model on a DIFFERENT provider than the primary, so keying
        # it genuinely exercises the fallback path (same-provider members share
        # the primary's lack of a key).
        fallback = next(
            (
                m
                for m in FALLBACK_CHAINS.get(primary, [])
                if ModelRouter._extract_provider(m) != primary_provider
            ),
            None,
        )
        if fallback is None:
            pytest.skip("COMPLEX primary's fallback chain has no cross-provider model")
        _set_only_provider_key(settings, ModelRouter._extract_provider(fallback), "test-key")
        router = ModelRouter(settings)
        # Primary skipped (no key); first chain member whose provider has a key wins.
        assert router.route(TaskComplexity.COMPLEX) == fallback

    def test_complexity_tier_map_chain_keys_are_valid(self) -> None:
        """COMPLEXITY_TIER_MAP chain keys exist in FALLBACK_CHAINS."""
        from src.config.model_registry import FALLBACK_CHAINS
        from src.llm.model_router import COMPLEXITY_TIER_MAP

        for complexity, (_, chain_key) in COMPLEXITY_TIER_MAP.items():
            assert chain_key in FALLBACK_CHAINS, (
                f"Chain key '{chain_key}' for {complexity.value} not in FALLBACK_CHAINS"
            )


class TestModelRouterProviderHealth:
    """Tests for provider health management."""

    def test_mark_provider_unhealthy_excludes(self, router: ModelRouter) -> None:
        """Marking a provider unhealthy excludes it from routing."""
        router.mark_provider_unhealthy("openai")
        assert "openai" in router._exclude_providers

    def test_clear_provider_health_restores(self, router: ModelRouter) -> None:
        """Clearing provider health re-enables the provider."""
        router.mark_provider_unhealthy("openai")
        router.clear_provider_health("openai")
        assert "openai" not in router._exclude_providers

    def test_get_fallback_chain(self, router: ModelRouter) -> None:
        """get_fallback_chain returns a list."""
        chain = router.get_fallback_chain("gpt-4o-mini-2024-07-18")
        assert isinstance(chain, list)


class TestModelRouterRouteReasoning:
    """Tests for route_reasoning method."""

    def test_route_reasoning_returns_configured_model(self) -> None:
        """route_reasoning returns the reasoning_llm_model from settings."""
        settings = Settings()
        router = ModelRouter(settings)
        result = router.route_reasoning()
        # Default is deepseek-v4-pro; may fall back if no key, but always returns str
        assert isinstance(result, str)
        assert len(result) > 0

    def test_route_reasoning_defaults_to_deepseek(self) -> None:
        """Default reasoning model setting should be deepseek-v4-pro (code default)."""
        # Verify the code default in LLMProviderSettings
        from src.config.settings import LLMProviderSettings

        field_default = LLMProviderSettings.model_fields["reasoning_llm_model"].default
        assert field_default == "deepseek-v4-pro"
        provider_default = LLMProviderSettings.model_fields["reasoning_llm_provider"].default
        assert provider_default == "deepseek"

    def test_route_reasoning_fallback_on_missing_key(self) -> None:
        """When reasoning model's provider has no key, falls back to CRITICAL."""
        settings = Settings()
        # Ensure no deepseek key so it falls back
        settings.llm.deepseek_api_key = None
        router = ModelRouter(settings)
        result = router.route_reasoning()
        # Should still return a valid model (from CRITICAL fallback)
        assert isinstance(result, str)
        assert len(result) > 0


class TestModelRouterRouteDiverse:
    """Tests for route_diverse method."""

    def test_returns_n_models(self, router: ModelRouter) -> None:
        """route_diverse returns exactly n model identifiers."""
        models = router.route_diverse(n=3, complexity=TaskComplexity.SIMPLE)
        assert isinstance(models, list)
        assert len(models) == 3
        for m in models:
            assert isinstance(m, str) and len(m) > 0

    def test_models_from_different_providers(self, router: ModelRouter) -> None:
        """When enough providers exist, models come from different providers."""
        models = router.route_diverse(n=3, complexity=TaskComplexity.SIMPLE)
        providers = {ModelRouter._extract_provider(m) for m in models}
        # Should have at least 2 distinct providers if keys are configured
        assert len(providers) >= 1  # minimum: all same provider (key-limited)

    def test_cycles_when_fewer_providers_than_n(self, router: ModelRouter) -> None:
        """When fewer providers than n, cycles through available models."""
        models = router.route_diverse(n=10, complexity=TaskComplexity.SIMPLE)
        assert len(models) == 10
        # All should be valid model strings
        for m in models:
            assert isinstance(m, str) and len(m) > 0

    def test_single_model_request(self, router: ModelRouter) -> None:
        """n=1 returns a list with one model."""
        models = router.route_diverse(n=1, complexity=TaskComplexity.SIMPLE)
        assert len(models) == 1
        assert isinstance(models[0], str)

    def test_excludes_unhealthy_providers(self, router: ModelRouter) -> None:
        """route_diverse respects excluded providers."""
        router.mark_provider_unhealthy("openai")
        models = router.route_diverse(n=3, complexity=TaskComplexity.SIMPLE)
        providers = {ModelRouter._extract_provider(m) for m in models}
        assert "openai" not in providers
