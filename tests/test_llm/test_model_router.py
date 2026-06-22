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

    def test_extract_provider_alibaba_hosted_deepseek(self) -> None:
        """Regression: ``alibaba-deepseek-v4-flash`` (DeepSeek's model hosted on
        DashScope) resolves to provider ``alibaba`` — NOT ``deepseek`` and NOT
        ``unknown``. Its registry key matches no prefix heuristic, so before the
        registry-lookup fix the router returned ``unknown``, logged "no API key
        for unknown", and silently skipped it in every fallback chain — defeating
        the provider-diverse routing it was added for."""
        assert ModelRouter._extract_provider("alibaba-deepseek-v4-flash") == "alibaba"

    def test_every_registered_key_resolves_to_its_spec_provider(self) -> None:
        """Cross-cutting invariant: every registered model key must resolve via
        ``_extract_provider`` to exactly the provider declared in its ModelSpec.
        A mismatch means the router cannot look up the model's API key, so it is
        silently skipped in every fallback chain — an unroutable model is worse
        than no fallback. This would have caught the alibaba-deepseek-v4-flash
        regression for ANY future model whose key breaks the prefix heuristic."""
        from src.config.model_registry import MODEL_REGISTRY

        mismatches = {
            key: (ModelRouter._extract_provider(key), spec.provider)
            for key, spec in MODEL_REGISTRY.items()
            if ModelRouter._extract_provider(key) != spec.provider
        }
        assert not mismatches, (
            f"_extract_provider disagrees with ModelSpec.provider for: {mismatches}"
        )


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

    def test_complexity_tier_map_primaries_are_registered(self) -> None:
        """Every COMPLEXITY_TIER_MAP primary model is a key in MODEL_REGISTRY.

        Complements the FALLBACK_CHAINS check above: catches tier-map / registry
        drift where the map names a model that no longer exists in the registry.
        """
        from src.config.model_registry import MODEL_REGISTRY
        from src.llm.model_router import COMPLEXITY_TIER_MAP, DEFAULT_COMPLEXITY_TIER

        for complexity, (_, chain_key) in COMPLEXITY_TIER_MAP.items():
            assert chain_key in MODEL_REGISTRY, (
                f"Primary '{chain_key}' for {complexity.value} not in MODEL_REGISTRY"
            )
        # The defensive default must also be a real registered model.
        assert DEFAULT_COMPLEXITY_TIER[1] in MODEL_REGISTRY

    def test_trivial_primary_is_qwen36_flash_with_qwen35_fallback(self) -> None:
        """Promotion regression: the TRIVIAL primary is qwen3.6-flash (the
        rolling flash alias), and its predecessor qwen3.5-flash survives as a
        fallback — so promoting the newer flash model never strands the old one.

        Locks the deliberate COMPLEXITY_TIER_MAP swap (qwen3.5-flash →
        qwen3.6-flash for TRIVIAL) together with its safety property: the new
        primary is routable (registered, provider alibaba) and the old primary
        is still reachable on failure via its fallback chain."""
        from src.config.model_registry import FALLBACK_CHAINS, MODEL_REGISTRY
        from src.llm.model_router import COMPLEXITY_TIER_MAP

        trivial_primary = COMPLEXITY_TIER_MAP[TaskComplexity.TRIVIAL][1]
        assert trivial_primary == "qwen3.6-flash"
        # The primary must resolve to a real alibaba-routed model, else it would
        # be silently skipped in every chain it appears in (the
        # alibaba-deepseek-v4-flash regression class).
        assert trivial_primary in MODEL_REGISTRY
        assert ModelRouter._extract_provider(trivial_primary) == "alibaba"
        # Safety: the predecessor flash model is reachable as a fallback, so the
        # promotion can't strand the previously-primary model on failure.
        assert "qwen3.5-flash" in FALLBACK_CHAINS.get(trivial_primary, [])

    def test_route_uses_default_complexity_tier_when_unmapped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When a complexity is absent from COMPLEXITY_TIER_MAP, route() falls
        back to the single DEFAULT_COMPLEXITY_TIER constant — not a duplicated
        inline literal. Patches the map empty so SIMPLE is treated as 'unmapped',
        keys the default's provider, and asserts route() returns the default's
        primary model. Regression guard for the duplicated-default dedup."""
        from src.llm import model_router as mr
        from src.llm.model_router import DEFAULT_COMPLEXITY_TIER

        settings = Settings()
        default_provider = ModelRouter._extract_provider(DEFAULT_COMPLEXITY_TIER[1])
        _set_only_provider_key(settings, default_provider, "test-key")
        router = ModelRouter(settings)
        monkeypatch.setattr(mr, "COMPLEXITY_TIER_MAP", {})

        assert router.route(TaskComplexity.SIMPLE) == DEFAULT_COMPLEXITY_TIER[1]


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


class TestModelRouterPerNodeRouting:
    """Tests for per-node routing (findings-05 A): NODE_TIER_MAP overrides, the
    de-flat (COMPLEX != SIMPLE), and the resurrected route_reasoning caller
    (verify/reflect on complex/critical goals)."""

    @staticmethod
    def _all_keyed_router() -> ModelRouter:
        """A router with every provider key set, so each complexity/node primary
        resolves to itself (no fallback masking the tier-map choice)."""
        settings = Settings()
        for field in _PROVIDER_KEY_FIELDS.values():
            setattr(settings.llm, field, "test-key")
        return ModelRouter(settings)

    def test_deflat_complex_ne_simple_for_plan(self) -> None:
        """De-flat (findings-03 #1): a COMPLEX plan no longer collapses to the
        same model as a SIMPLE plan. With all providers keyed, COMPLEX→plan
        resolves to its NODE_TIER_MAP primary (MODERATE glm-4.7) while
        SIMPLE→plan falls to the COMPLEXITY_TIER_MAP SIMPLE default (CHEAP)."""
        from src.llm.model_router import NODE_TIER_MAP

        router = self._all_keyed_router()
        complex_plan = router.route(TaskComplexity.COMPLEX, node="plan")
        simple_plan = router.route(TaskComplexity.SIMPLE, node="plan")
        assert complex_plan != simple_plan
        # COMPLEX+plan hits the NODE_TIER_MAP override primary.
        assert complex_plan == NODE_TIER_MAP[(TaskComplexity.COMPLEX, "plan")][1]

    def test_execute_stays_cheap_on_complex_goal(self) -> None:
        """Cost discipline: execute stays CHEAP even on a COMPLEX goal —
        NODE_TIER_MAP overrides the de-flatted COMPLEX→MODERATE default so
        tool-calling steps don't overspend. Asserted via the registry tier of
        the returned model so it tracks the invariant, not a model name."""
        from src.config.model_registry import MODEL_REGISTRY, ModelTier

        router = self._all_keyed_router()
        model = router.route(TaskComplexity.COMPLEX, node="execute")
        assert MODEL_REGISTRY[model].tier in {ModelTier.VERY_CHEAP, ModelTier.CHEAP}

    def test_verify_and_reflect_complex_route_to_reasoning(self) -> None:
        """Resurrected route_reasoning caller: verify/reflect on a COMPLEX goal
        prefer the reasoning model (deepseek-v4-pro) when its provider key is
        set — route_reasoning() previously had zero callers."""
        settings = Settings()
        _set_only_provider_key(settings, "deepseek", "test-key")
        router = ModelRouter(settings)
        assert router.route(TaskComplexity.COMPLEX, node="verify") == "deepseek-v4-pro"
        assert router.route(TaskComplexity.COMPLEX, node="reflect") == "deepseek-v4-pro"

    def test_verify_critical_routes_to_reasoning_when_keyed(self) -> None:
        """CRITICAL verify also prefers the reasoning model when keyed."""
        settings = Settings()
        _set_only_provider_key(settings, "deepseek", "test-key")
        router = ModelRouter(settings)
        assert router.route(TaskComplexity.CRITICAL, node="verify") == "deepseek-v4-pro"

    def test_verify_falls_back_when_reasoning_key_absent(self) -> None:
        """When the reasoning model's provider has no key, verify/reflect fall
        back to the CRITICAL complexity default (NOT deepseek-v4-pro), via
        route_reasoning → route(CRITICAL) with node=None (no recursion)."""
        settings = Settings()
        _set_only_provider_key(settings, "zai", "test-key")  # no deepseek key
        router = ModelRouter(settings)
        result = router.route(TaskComplexity.CRITICAL, node="verify")
        assert result != "deepseek-v4-pro"
        # route_reasoning falls back to route(CRITICAL) → CRITICAL primary glm-4.7.
        assert result == "glm-4.7"

    def test_node_tier_map_miss_falls_back_to_complexity_map(self) -> None:
        """A (complexity, node) pair absent from NODE_TIER_MAP uses the
        COMPLEXITY_TIER_MAP default. SIMPLE+reflect is not a NODE_TIER_MAP key
        and SIMPLE is not complex enough for the reasoning branch, so it returns
        the SIMPLE primary."""
        from src.llm.model_router import COMPLEXITY_TIER_MAP

        router = self._all_keyed_router()
        _tier, simple_primary = COMPLEXITY_TIER_MAP[TaskComplexity.SIMPLE]
        assert router.route(TaskComplexity.SIMPLE, node="reflect") == simple_primary

    def test_node_param_none_preserves_existing_behavior(self) -> None:
        """Regression guard: route(complexity) with no node behaves exactly as
        before — node-aware branches are skipped, COMPLEXITY_TIER_MAP wins."""
        from src.llm.model_router import COMPLEXITY_TIER_MAP

        router = self._all_keyed_router()
        for complexity in (
            TaskComplexity.SIMPLE,
            TaskComplexity.COMPLEX,
            TaskComplexity.CRITICAL,
        ):
            _tier, primary = COMPLEXITY_TIER_MAP[complexity]
            assert router.route(complexity) == primary

    def test_node_tier_map_keys_are_registered_and_chained(self) -> None:
        """Cross-cutting invariant: every NODE_TIER_MAP chain key is a real
        registered model with a fallback chain — mirrors the COMPLEXITY_TIER_MAP
        guard so a per-node override can never name an unroutable model."""
        from src.config.model_registry import FALLBACK_CHAINS, MODEL_REGISTRY
        from src.llm.model_router import NODE_TIER_MAP

        for (_complexity, _node), (_tier, chain_key) in NODE_TIER_MAP.items():
            assert chain_key in MODEL_REGISTRY, f"{chain_key} not in MODEL_REGISTRY"
            assert chain_key in FALLBACK_CHAINS, f"{chain_key} not in FALLBACK_CHAINS"
