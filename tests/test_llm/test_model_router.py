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

    def test_complexity_tier_map_chain_keys_are_valid(self) -> None:
        """COMPLEXITY_TIER_MAP chain keys exist in FALLBACK_CHAINS."""
        from src.config.model_registry import FALLBACK_CHAINS
        from src.llm.model_router import COMPLEXITY_TIER_MAP

        for complexity, (tier, chain_key) in COMPLEXITY_TIER_MAP.items():
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
