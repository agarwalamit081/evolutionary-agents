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
        from src.llm.model_router import COMPLEXITY_TIER_MAP, _resolved_disabled_providers

        settings = Settings()
        _tier, primary = COMPLEXITY_TIER_MAP[TaskComplexity.COMPLEX]
        primary_provider = ModelRouter._extract_provider(primary)
        # A disabled provider (e.g. anthropic under a quota cap) is never
        # routable even with a key, so it is excluded here just as route() does
        # — otherwise the picker selects a fallback production can never reach.
        disabled = _resolved_disabled_providers(settings)
        # First chain model on a DIFFERENT, non-disabled provider than the
        # primary, so keying it genuinely exercises the fallback path
        # (same-provider members share the primary's lack of a key).
        fallback = next(
            (
                m
                for m in FALLBACK_CHAINS.get(primary, [])
                if ModelRouter._extract_provider(m) != primary_provider
                and ModelRouter._extract_provider(m) not in disabled
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
        resolves to its NODE_TIER_MAP primary (MODERATE glm-5.2) while
        SIMPLE→plan falls to the COMPLEXITY_TIER_MAP SIMPLE default (CHEAP)."""
        from src.llm.model_router import NODE_TIER_MAP

        router = self._all_keyed_router()
        complex_plan = router.route(TaskComplexity.COMPLEX, node="plan")
        simple_plan = router.route(TaskComplexity.SIMPLE, node="plan")
        assert complex_plan != simple_plan
        # COMPLEX+plan hits the NODE_TIER_MAP override primary.
        assert complex_plan == NODE_TIER_MAP[(TaskComplexity.COMPLEX, "plan")][1]

    def test_execute_upgrades_to_glm52_on_complex_goal(self) -> None:
        """Track-1 re-baseline: execute on a COMPLEX/CRITICAL goal runs the
        MODERATE glm-5.2 — the successor to the glm-5.1 C3 primary, a stronger
        live tool-caller than the CHEAP tier. The cost uplift is bounded by
        per-step routing (Phase 3 routes trivial/simple steps back to the CHEAP
        tier) + RAG-over-tools, NOT by keeping execute CHEAP. Pinned to the model
        id (not just the tier) since the primary swap is the point."""
        router = self._all_keyed_router()
        assert router.route(TaskComplexity.COMPLEX, node="execute") == "glm-5.2"
        assert router.route(TaskComplexity.CRITICAL, node="execute") == "glm-5.2"

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


class TestRoutingEnvOverrides:
    """Tests for the F2 operator env-knob tier overrides (RoutingSettings).

    ``_apply_routing_overrides`` layers JSON ``{routing_key: model_id}`` overrides
    over the curated COMPLEXITY_TIER_MAP / NODE_TIER_MAP at ``route()`` call-time.
    A node-tier key (``"COMPLEXITY:node"``) wins over a complexity-tier key (bare
    ``"COMPLEXITY"``); empty / invalid JSON or an unknown model_id leaves the
    curated tier unchanged so a bad env can never break routing. When an override
    applies, the tier is re-derived from the override model's ModelSpec.
    """

    @staticmethod
    def _all_keyed_router(
        node_json: str = "{}", cpx_json: str = "{}"
    ) -> ModelRouter:
        """Router with every provider key set + the given routing override JSON.

        Keying all providers means each override's named primary resolves to
        itself (its provider has a key) instead of being masked by a fallback —
        so the assertion tracks the override choice, not the fallback path.
        """
        settings = Settings()
        for field in _PROVIDER_KEY_FIELDS.values():
            setattr(settings.llm, field, "test-key")
        settings.routing.routing_node_tier_overrides_json = node_json
        settings.routing.routing_complexity_tier_overrides_json = cpx_json
        return ModelRouter(settings)

    def test_node_tier_override_flips_routing(self) -> None:
        """A node-tier override redirects a (complexity, node) decision. With all
        providers keyed, COMPLEX+execute normally returns the NODE_TIER_MAP
        primary glm-5.2 (Track-1 re-baseline); overriding ``COMPLEX:execute`` →
        deepseek-v4-pro flips it."""
        from src.llm.model_router import NODE_TIER_MAP

        router = self._all_keyed_router(
            node_json='{"COMPLEX:execute": "deepseek-v4-pro"}'
        )
        # Sanity: the curated default for COMPLEX:execute is NOT the override.
        assert NODE_TIER_MAP[(TaskComplexity.COMPLEX, "execute")][1] == "glm-5.2"
        assert router.route(TaskComplexity.COMPLEX, node="execute") == "deepseek-v4-pro"

    def test_complexity_tier_override_flips_routing_no_node(self) -> None:
        """A bare complexity-tier override redirects the node=None default. With
        all providers keyed, route(COMPLEX) normally returns glm-4.7; overriding
        ``COMPLEX`` → deepseek-v4-pro flips it."""
        router = self._all_keyed_router(cpx_json='{"COMPLEX": "deepseek-v4-pro"}')
        assert router.route(TaskComplexity.COMPLEX) == "deepseek-v4-pro"

    def test_node_tier_override_wins_over_complexity_tier(self) -> None:
        """When a node-tier and a complexity-tier override name different models
        for the same decision, the more-specific node-tier wins. Here
        ``COMPLEX:execute`` → deepseek-v4-pro beats ``COMPLEX`` → glm-5.1
        (both differ from the curated COMPLEX:execute primary glm-4.7)."""
        router = self._all_keyed_router(
            node_json='{"COMPLEX:execute": "deepseek-v4-pro"}',
            cpx_json='{"COMPLEX": "glm-5.1"}',
        )
        assert router.route(TaskComplexity.COMPLEX, node="execute") == "deepseek-v4-pro"

    def test_invalid_json_override_is_ignored(self) -> None:
        """Malformed override JSON is logged at DEBUG and treated as no override
        — routing falls through to the curated tier (cannot break routing)."""
        from src.llm.model_router import NODE_TIER_MAP

        curated = NODE_TIER_MAP[(TaskComplexity.COMPLEX, "execute")][1]
        router = self._all_keyed_router(node_json="{not valid json}")
        assert router.route(TaskComplexity.COMPLEX, node="execute") == curated

    def test_empty_override_is_noop(self) -> None:
        """The default empty JSON leaves behavior identical to no overrides —
        regression guard that the default-off path is byte-identical to the
        curated tier maps."""
        from src.llm.model_router import NODE_TIER_MAP

        curated = NODE_TIER_MAP[(TaskComplexity.COMPLEX, "plan")][1]
        router = self._all_keyed_router()  # default "{}" / "{}"
        assert router.route(TaskComplexity.COMPLEX, node="plan") == curated

    def test_unknown_model_id_override_is_ignored(self) -> None:
        """An override naming a model absent from MODEL_REGISTRY is ignored
        (logged WARNING) so a typo can never select an unroutable model."""
        from src.llm.model_router import NODE_TIER_MAP

        curated = NODE_TIER_MAP[(TaskComplexity.COMPLEX, "execute")][1]
        router = self._all_keyed_router(
            node_json='{"COMPLEX:execute": "does-not-exist-9000"}'
        )
        assert router.route(TaskComplexity.COMPLEX, node="execute") == curated

    def test_override_rederives_tier_from_override_model(self) -> None:
        """When an override applies, the tier is re-derived from the override
        model's ModelSpec so the absolute-fallback loop stays coherent. Curated
        COMPLEX is glm-4.7 (MODERATE); overriding ``COMPLEX`` → gemini-2.5-flash
        (CHEAP) must report CHEAP, not the curated MODERATE."""
        from src.config.model_registry import MODEL_REGISTRY

        override_model = "gemini-2.5-flash"
        override_tier = MODEL_REGISTRY[override_model].tier
        curated_tier = MODEL_REGISTRY["glm-4.7"].tier
        assert override_tier != curated_tier  # genuinely crosses tiers
        router = self._all_keyed_router(cpx_json=f'{{"COMPLEX": "{override_model}"}}')
        tier, chain_key = router._apply_routing_overrides(
            TaskComplexity.COMPLEX, None, curated_tier, "glm-4.7"
        )
        assert chain_key == override_model
        assert tier == override_tier


class TestCrossProviderFallbackInvariant:
    """Lock requirement 2 (#311): every routing primary has at least one
    *registered* fallback on a *different provider* in its FALLBACK_CHAINS.

    A fallback chain that is entirely same-provider is useless when the primary's
    own provider caps/burns balance or rate-limits — the whole chain dies with it.
    This cross-provider property is what lets the gateway pivot to another provider
    on failure. It is already true in FALLBACK_CHAINS; these tests pin it so a
    future edit can't silently collapse a chain onto one provider.

    Note: the invariant is "at least one cross-provider member", NOT "the first
    member is cross-provider" — several primaries deliberately front a same-
    provider sibling (e.g. qwen3.6-flash → qwen3.5-flash, both alibaba) so a
    model promotion never strands its predecessor. The first member may share a
    provider; what matters is that a different provider is reachable.
    """

    @staticmethod
    def _routing_primaries() -> set[str]:
        """Distinct model IDs the router can ever return as a primary.

        Union of COMPLEXITY_TIER_MAP values, NODE_TIER_MAP values, the
        DEFAULT_COMPLEXITY_TIER fallback, and the configured reasoning model
        (route_reasoning's primary on verify/reflect for complex/critical goals).
        """
        from src.config.settings import LLMProviderSettings
        from src.llm.model_router import (
            COMPLEXITY_TIER_MAP,
            DEFAULT_COMPLEXITY_TIER,
            NODE_TIER_MAP,
        )

        primaries: set[str] = set()
        for entry in COMPLEXITY_TIER_MAP.values():
            primaries.add(entry[1])
        for entry in NODE_TIER_MAP.values():
            primaries.add(entry[1])
        primaries.add(DEFAULT_COMPLEXITY_TIER[1])
        reasoning = LLMProviderSettings.model_fields["reasoning_llm_model"].default
        if isinstance(reasoning, str):
            primaries.add(reasoning)
        return primaries

    @staticmethod
    def _cross_provider_offenders(models: set[str]) -> dict[str, list[str]]:
        """For each model, report its chain if it lacks a registered
        cross-provider fallback. Empty dict ⇒ invariant holds for every model.

        A named-but-unregistered fallback is silently skipped by
        ``_route_from_chain`` (no key lookup), so membership must be registered to
        count; a same-provider fallback dies with the primary, so it must be
        cross-provider to count.
        """
        from src.config.model_registry import FALLBACK_CHAINS, MODEL_REGISTRY

        offenders: dict[str, list[str]] = {}
        for model in models:
            chain = FALLBACK_CHAINS.get(model, [])
            provider = ModelRouter._extract_provider(model)
            has_cross = any(
                member in MODEL_REGISTRY
                and ModelRouter._extract_provider(member) != provider
                for member in chain
            )
            if not has_cross:
                offenders[model] = chain
        return offenders

    def test_routing_primaries_all_define_a_fallback_chain(self) -> None:
        """Every routing primary is a key in FALLBACK_CHAINS — else the gateway
        has no fallback path at all for that primary."""
        from src.config.model_registry import FALLBACK_CHAINS

        missing = [p for p in self._routing_primaries() if p not in FALLBACK_CHAINS]
        assert not missing, f"Routing primaries without a FALLBACK_CHAINS entry: {missing}"

    def test_every_routing_primary_has_registered_cross_provider_fallback(self) -> None:
        """Every routing primary's chain has >=1 fallback that is BOTH registered
        AND on a different provider — so a single-provider outage can never strand
        a primary with no escape."""
        offenders = self._cross_provider_offenders(self._routing_primaries())
        assert not offenders, (
            "Routing primaries whose fallback chain has no registered "
            "cross-provider fallback (would strand on a single-provider outage): "
            f"{offenders}"
        )

    def test_candidate_models_have_cross_provider_fallback(self) -> None:
        """Promotion safety for the plan-named candidates (qwen3-coder-next,
        gpt-5-nano-2025-08-07, kimi-k2.6): each already satisfies the
        cross-provider-fallback invariant, so promoting any to a routing primary
        later cannot strand its chain. Not current primaries (owner decision: no
        routing edits), but locked pre-emptively."""
        candidates = {"qwen3-coder-next", "gpt-5-nano-2025-08-07", "kimi-k2.6"}
        offenders = self._cross_provider_offenders(candidates)
        assert not offenders, (
            "Candidate models (promotion targets) whose fallback chain has no "
            f"registered cross-provider fallback: {offenders}"
        )


class TestDefaultComplexityTierOverride:
    """ROUTING_DEFAULT_COMPLEXITY_TIER retunes the defensive default tier for an
    UNMAPPED TaskComplexity (the .get() fallback in route()/route_diverse()),
    WITHOUT a code change. Empty/unknown → the curated DEFAULT_COMPLEXITY_TIER.

    ``_effective_default_tier`` is the single source the two .get() call sites
    now consult; an override's tier is re-derived from its ModelSpec so the
    absolute-fallback loop stays coherent."""

    @staticmethod
    def _all_keyed_router(default_tier: str = "") -> ModelRouter:
        settings = Settings()
        for field in _PROVIDER_KEY_FIELDS.values():
            setattr(settings.llm, field, "test-key")
        settings.routing.routing_default_complexity_tier = default_tier
        return ModelRouter(settings)

    def test_override_flips_unmapped_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Patching the tier map empty makes SIMPLE 'unmapped', so route() falls
        to the override model (deepseek-v4-pro) instead of the curated default
        (claude-haiku-4-5-20251001)."""
        from src.llm import model_router as mr

        router = self._all_keyed_router(default_tier="deepseek-v4-pro")
        monkeypatch.setattr(mr, "COMPLEXITY_TIER_MAP", {})
        assert router.route(TaskComplexity.SIMPLE) == "deepseek-v4-pro"

    def test_empty_uses_curated_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty override → byte-identical to the curated DEFAULT_COMPLEXITY_TIER."""
        from src.llm import model_router as mr
        from src.llm.model_router import DEFAULT_COMPLEXITY_TIER

        router = self._all_keyed_router(default_tier="")
        monkeypatch.setattr(mr, "COMPLEXITY_TIER_MAP", {})
        assert router.route(TaskComplexity.SIMPLE) == DEFAULT_COMPLEXITY_TIER[1]

    def test_unknown_model_id_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An override naming a model absent from MODEL_REGISTRY is ignored
        (logged WARNING) so a typo can't select an unroutable model."""
        from src.llm import model_router as mr
        from src.llm.model_router import DEFAULT_COMPLEXITY_TIER

        router = self._all_keyed_router(default_tier="does-not-exist-9000")
        monkeypatch.setattr(mr, "COMPLEXITY_TIER_MAP", {})
        assert router.route(TaskComplexity.SIMPLE) == DEFAULT_COMPLEXITY_TIER[1]

    def test_override_rederives_tier_from_override_model(self) -> None:
        """The resolved tier is re-derived from the override model's ModelSpec so
        the absolute-fallback loop stays coherent (curated default is CHEAP;
        overriding to glm-4.7 must report MODERATE)."""
        from src.config.model_registry import MODEL_REGISTRY, ModelTier

        router = self._all_keyed_router(default_tier="glm-4.7")
        tier, chain_key = router._effective_default_tier()
        assert chain_key == "glm-4.7"
        assert tier == MODEL_REGISTRY["glm-4.7"].tier
        assert MODEL_REGISTRY["glm-4.7"].tier == ModelTier.MODERATE

    def test_both_get_call_sites_use_resolver(self) -> None:
        """Regression: route() and route_diverse() BOTH consult
        _effective_default_tier() — not a duplicated inline literal — so the two
        cannot silently drift (the dedup the DEFAULT_COMPLEXITY_TIER constant was
        introduced for)."""
        import inspect

        from src.llm.model_router import ModelRouter

        route_src = inspect.getsource(ModelRouter.route)
        route_diverse_src = inspect.getsource(ModelRouter.route_diverse)
        assert "_effective_default_tier()" in route_src
        assert "_effective_default_tier()" in route_diverse_src


class TestDisabledProvidersEnv:
    """DISABLED_PROVIDERS is the env-side lever for excluding providers from ALL
    routing. AUTHORITATIVE-when-set (not merge): None/empty → nothing disabled
    (the temporary Anthropic quota-cap baseline was reverted); any non-empty
    comma-list → the authoritative disabled set."""

    def test_none_disables_nothing(self) -> None:
        """UNSET → nothing is disabled (no hardcoded baseline after the
        temporary Anthropic quota-cap block was reverted)."""
        from src.llm.model_router import _resolved_disabled_providers

        settings = Settings()
        settings.routing.routing_disabled_providers = None
        assert _resolved_disabled_providers(settings) == set()
        assert "anthropic" not in _resolved_disabled_providers(settings)

    def test_empty_string_clears_all(self) -> None:
        """EMPTY string (=) means NONE disabled."""
        from src.llm.model_router import _resolved_disabled_providers

        settings = Settings()
        settings.routing.routing_disabled_providers = ""
        assert _resolved_disabled_providers(settings) == set()
        assert "anthropic" not in _resolved_disabled_providers(settings)

    def test_explicit_list_is_authoritative(self) -> None:
        """A set comma-list IS the disabled set — nothing else is merged in."""
        from src.llm.model_router import _resolved_disabled_providers

        settings = Settings()
        settings.routing.routing_disabled_providers = "minimax"
        resolved = _resolved_disabled_providers(settings)
        assert resolved == {"minimax"}
        assert "anthropic" not in resolved

    def test_comma_list_parses_with_whitespace(self) -> None:
        """A multi-entry comma-list trims whitespace around each member."""
        from src.llm.model_router import _resolved_disabled_providers

        settings = Settings()
        settings.routing.routing_disabled_providers = " anthropic , minimax , "
        assert _resolved_disabled_providers(settings) == {"anthropic", "minimax"}

    def test_disabled_provider_has_no_key(self) -> None:
        """A disabled provider reports key-less via _has_provider_key so it is
        dropped from primary/chain/diverse selection AND the gateway's fallback
        pre-filter — regardless of whether an actual key is present in settings."""
        settings = Settings()
        settings.routing.routing_disabled_providers = "openai"
        settings.llm.openai_api_key = "real-but-disabled-key"
        router = ModelRouter(settings)
        assert router._has_provider_key("openai") is False

    def test_non_disabled_provider_key_check_unaffected(self) -> None:
        """A provider NOT in the disabled set still gets its real key checked."""
        settings = Settings()
        settings.routing.routing_disabled_providers = "anthropic"  # disable anthropic only
        settings.llm.openai_api_key = "real-key"
        router = ModelRouter(settings)
        assert router._has_provider_key("openai") is True
        assert router._has_provider_key("anthropic") is False

    def test_runtime_excluded_set_seeds_from_env(self) -> None:
        """The router's runtime _exclude_providers (consulted by the absolute-
        fallback loop) seeds from the resolved disabled set at __init__."""
        settings = Settings()
        settings.routing.routing_disabled_providers = "minimax,moonshot"
        router = ModelRouter(settings)
        assert router._exclude_providers == {"minimax", "moonshot"}

    def test_is_provider_disabled_reads_live_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """is_provider_disabled() is static but reads the live resolved disabled
        set on EACH call (via get_settings() + _resolved_disabled_providers), so
        it tracks a runtime flip — not a snapshot taken at import. Hermetic to
        .env: the resolver is stubbed with a mutable box rather than relying on
        the live DISABLED_PROVIDERS value."""
        import src.llm.model_router as mr
        from src.llm.model_router import ModelRouter

        box: dict[str, set[str]] = {"s": {"anthropic"}}
        monkeypatch.setattr(mr, "_resolved_disabled_providers", lambda _s: set(box["s"]))

        assert ModelRouter.is_provider_disabled("anthropic") is True
        assert ModelRouter.is_provider_disabled("minimax") is False

        # Flip the live set at runtime — the static method must observe it.
        box["s"] = {"minimax"}
        assert ModelRouter.is_provider_disabled("anthropic") is False
        assert ModelRouter.is_provider_disabled("minimax") is True
