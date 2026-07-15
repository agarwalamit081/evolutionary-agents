"""Gap-focused depth tests for the LLM routing / cache / cost subsystem.

Targets the *behavior gaps* not already covered by the sibling files in
``tests/test_llm/``:

* ``model_router.COMPLEXITY_TIER_MAP`` — each ``TaskComplexity`` maps to a
  registered model whose tier is non-decreasing with complexity (the de-flat
  invariant: COMPLEX must NOT collapse onto the same Cheap model as SIMPLE).
* ``NODE_TIER_MAP`` — per-node overrides actually change the routed model for
  ``plan``/``execute`` vs the complexity default, and ``verify``/``reflect`` on
  hard goals route to the reasoning model.
* ``FALLBACK_CHAINS`` — every primary chain contains at least one
  cross-provider entry (a dead single-provider chain would defeat resilience).
* ``cache.PromptCache`` — deterministic key for identical inputs (independent of
  kwarg order), distinct keys for distinct inputs, TTL forwarded to Redis on
  ``set``, and a miss falls through (returns ``None``) so the gateway calls the
  provider.
* ``cost_tracker.CostTracker`` — per-``run_id`` token + cost aggregation, and
  the resilience pattern: a failed ledger write rolls back, is logged, and never
  re-raises (a poisoned session recovers on the next call).

Order-safety: every test builds its OWN ``Settings()`` / mock session. The
``get_settings()`` singleton is never mutated, so pytest-randomly order cannot
poison a later test.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config.model_registry import (
    FALLBACK_CHAINS,
    MODEL_REGISTRY,
    ModelSpec,
    ModelTier,
)
from src.config.settings import Settings
from src.graph.enums import TaskComplexity
from src.llm.cache import PromptCache
from src.llm.cost_tracker import CostTracker
from src.llm.model_router import (
    COMPLEXITY_TIER_MAP,
    DEFAULT_COMPLEXITY_TIER,
    NODE_TIER_MAP,
    ModelRouter,
)
from src.llm.models import LLMResponse


# ─── Helpers ─────────────────────────────────────────────────────────


# Mirrors LLMProviderSettings.get_provider_key. Lets a test arm exactly one
# provider's key on a fresh Settings without touching the get_settings()
# singleton (order-safe under pytest-randomly).
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


def _fresh_settings() -> Settings:
    """A brand-new Settings (NOT the get_settings() singleton)."""
    return Settings()


def _arm_only(settings: Settings, provider: str) -> None:
    """Clear every provider key, then set exactly ``provider`` to a non-empty value."""
    for field in _PROVIDER_KEY_FIELDS.values():
        setattr(settings.llm, field, None)
    field = _PROVIDER_KEY_FIELDS.get(provider)
    if field is not None:
        setattr(settings.llm, field, "fake-key-for-tests")


def _make_response(**overrides: Any) -> LLMResponse:
    defaults = {
        "content": "cached answer",
        "model": "gpt-4o-mini-2024-07-18",
        "provider": "openai",
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
        "cost_usd": 0.0001,
        "finish_reason": "stop",
    }
    defaults.update(overrides)
    return LLMResponse(**defaults)


def _dict_redis() -> tuple[MagicMock, dict[str, str]]:
    """Mock aioredis client backed by a plain dict; captures set() kwargs."""
    store: dict[str, str] = {}
    set_calls: list[dict[str, Any]] = []

    redis_mock = MagicMock()

    async def _get(key: str) -> Any:
        return store.get(key)

    async def _set(key: str, value: str, **kwargs: Any) -> None:
        store[key] = value
        set_calls.append({"key": key, "value": value, **kwargs})

    redis_mock.get = AsyncMock(side_effect=_get)
    redis_mock.set = AsyncMock(side_effect=_set)
    redis_mock.scan_iter = MagicMock(return_value=iter([]))
    redis_mock._store = store  # type: ignore[attr-defined]
    redis_mock._set_calls = set_calls  # type: ignore[attr-defined]
    return redis_mock, store


# ════════════════════════════════════════════════════════════════════
# 1. COMPLEXITY_TIER_MAP — complexity → correct tier / model invariants
# ════════════════════════════════════════════════════════════════════


class TestComplexityTierMap:
    """The curated complexity→(tier,model) map must be internally coherent."""

    def test_every_complexity_is_mapped(self) -> None:
        """All four TaskComplexity members have a tier-map entry."""
        for cpx in TaskComplexity:
            assert cpx in COMPLEXITY_TIER_MAP, f"{cpx.name} missing from COMPLEXITY_TIER_MAP"

    def test_each_mapped_model_is_registered(self) -> None:
        """Every model the map names exists in MODEL_REGISTRY (no dead routing)."""
        for _tier, model in COMPLEXITY_TIER_MAP.values():
            assert model in MODEL_REGISTRY, f"COMPLEXITY_TIER_MAP names unknown model {model!r}"

    def test_complex_is_stronger_or_equal_to_simple(self) -> None:
        """De-flat invariant: COMPLEX tier is >= SIMPLE tier (never collapses to Cheap)."""
        # Tier rank: VERY_CHEAP < CHEAP < MODERATE.
        rank = {ModelTier.VERY_CHEAP: 0, ModelTier.CHEAP: 1, ModelTier.MODERATE: 2}
        simple_tier = COMPLEXITY_TIER_MAP[TaskComplexity.SIMPLE][0]
        complex_tier = COMPLEXITY_TIER_MAP[TaskComplexity.COMPLEX][0]
        assert rank[complex_tier] >= rank[simple_tier]

    def test_trivial_is_weakest_or_equal_to_simple(self) -> None:
        """TRIVIAL never routes to a stronger tier than SIMPLE (cost discipline)."""
        rank = {ModelTier.VERY_CHEAP: 0, ModelTier.CHEAP: 1, ModelTier.MODERATE: 2}
        trivial_tier = COMPLEXITY_TIER_MAP[TaskComplexity.TRIVIAL][0]
        simple_tier = COMPLEXITY_TIER_MAP[TaskComplexity.SIMPLE][0]
        assert rank[trivial_tier] <= rank[simple_tier]

    def test_default_complexity_tier_is_registered(self) -> None:
        """The defensive fallback (unmapped complexity) names a registered model."""
        _tier, model = DEFAULT_COMPLEXITY_TIER
        assert model in MODEL_REGISTRY

    def test_complex_and_simple_do_not_both_collapse_to_same_cheap_model(self) -> None:
        """The pre-de-flat bug: SIMPLE==COMPLEX==deepseek-v4-flash. Lock the fix.

        A non-trivial goal (COMPLEX) must not silently land on the SAME model as
        a SIMPLE goal — it should route to a distinct (stronger) model.
        """
        simple_model = COMPLEXITY_TIER_MAP[TaskComplexity.SIMPLE][1]
        complex_model = COMPLEXITY_TIER_MAP[TaskComplexity.COMPLEX][1]
        assert simple_model != complex_model, (
            "SIMPLE and COMPLEX both route to the same model — the flat-routing bug regressed"
        )


# ════════════════════════════════════════════════════════════════════
# 2. NODE_TIER_MAP — per-node overrides
# ════════════════════════════════════════════════════════════════════


class TestNodeTierMapOverrides:
    """Per-node overrides change routing vs the complexity default."""

    def test_plan_node_keys_into_registered_models(self) -> None:
        for (cpx, node), (_tier, model) in NODE_TIER_MAP.items():
            assert node is not None
            assert model in MODEL_REGISTRY, f"{cpx.name}:{node} -> unknown {model!r}"

    def test_execute_override_upgrades_complex_to_glm52(self) -> None:
        """Track-1 re-baseline: execute on a COMPLEX goal moves to MODERATE glm-5.2
        (the successor to the C3 glm-5.1 primary — 1M ctx, 128K out, stronger
        cost-effective tool-caller). The execute NODE_TIER_MAP override pins it
        explicitly instead of keeping execution CHEAP — the cost uplift is bounded
        by per-step routing (Phase 3, trivial steps back to CHEAP) + RAG-over-tools,
        not by a CHEAP execute tier. glm-5.1 stays its first FALLBACK_CHAINS entry.
        """
        key = (TaskComplexity.COMPLEX, "execute")
        assert key in NODE_TIER_MAP
        tier, model = NODE_TIER_MAP[key]
        assert tier == ModelTier.MODERATE
        assert model == "glm-5.2"
        assert MODEL_REGISTRY[model].tier == ModelTier.MODERATE

    def test_execute_override_explicitly_pins_glm52(self) -> None:
        """The execute override is an explicit glm-5.2 pin. Post Track-1 re-baseline
        it coincides with the COMPLEX complexity default (also glm-5.2), but the
        override entry is KEPT so execute won't silently drift if the complexity
        default ever changes — the model id is locked here, not implied."""
        assert NODE_TIER_MAP[(TaskComplexity.COMPLEX, "execute")][1] == "glm-5.2"

    def test_plan_tier_differs_from_or_refines_default(self) -> None:
        """The plan override for COMPLEX targets the reasoning tier (MODERATE)."""
        plan_tier, _plan_model = NODE_TIER_MAP[(TaskComplexity.COMPLEX, "plan")]
        assert plan_tier == ModelTier.MODERATE

    def test_route_uses_execute_override_when_provider_armed(self) -> None:
        """route(COMPLEX, 'execute') returns the execute-override model when its
        provider is armed. Post Phase-2 retier the execute primary is glm-4.7
        (provider ``zai``), so arming zai makes it resolve directly."""
        settings = _fresh_settings()
        _arm_only(settings, "zai")  # glm-4.7 provider (Phase-2 execute primary)
        router = ModelRouter(settings)
        expected = NODE_TIER_MAP[(TaskComplexity.COMPLEX, "execute")][1]
        assert router.route(TaskComplexity.COMPLEX, node="execute") == expected

    def test_route_plan_uses_override_model(self) -> None:
        """route(COMPLEX, 'plan') returns the plan-override model when its key is armed."""
        settings = _fresh_settings()
        _arm_only(settings, "zai")  # glm-4.7 provider
        router = ModelRouter(settings)
        expected = NODE_TIER_MAP[(TaskComplexity.COMPLEX, "plan")][1]
        assert router.route(TaskComplexity.COMPLEX, node="plan") == expected

    def test_route_unknown_node_falls_back_to_complexity_default(self) -> None:
        """A node with no override uses the plain complexity default."""
        settings = _fresh_settings()
        _arm_only(settings, "zai")  # glm-4.7 = COMPLEX default primary
        router = ModelRouter(settings)
        default_model = COMPLEXITY_TIER_MAP[TaskComplexity.COMPLEX][1]
        assert router.route(TaskComplexity.COMPLEX, node="no_such_node") == default_model

    def test_verify_on_complex_routes_to_reasoning_model(self) -> None:
        """verify on a COMPLEX goal prefers the configured reasoning model.

        Invariant to WHICH model ``reasoning_llm_model`` resolves to: we arm that
        model's own provider, so ``route_reasoning`` returns it directly (instead
        of falling back to CRITICAL routing) and the routed model must equal it.
        """
        settings = _fresh_settings()
        reasoning_model = settings.llm.reasoning_llm_model
        provider = ModelRouter._extract_provider(reasoning_model)
        _arm_only(settings, provider)
        router = ModelRouter(settings)
        assert router.route(TaskComplexity.COMPLEX, node="verify") == reasoning_model

    def test_reflect_on_critical_routes_to_reasoning_model(self) -> None:
        """reflect on a CRITICAL goal also routes to the reasoning model."""
        settings = _fresh_settings()
        reasoning_model = settings.llm.reasoning_llm_model
        provider = ModelRouter._extract_provider(reasoning_model)
        _arm_only(settings, provider)
        router = ModelRouter(settings)
        assert router.route(TaskComplexity.CRITICAL, node="reflect") == reasoning_model

    def test_verify_on_simple_does_not_route_to_reasoning(self) -> None:
        """The reasoning shortcut only fires for COMPLEX/CRITICAL, not SIMPLE."""
        settings = _fresh_settings()
        _arm_only(settings, "deepseek")
        router = ModelRouter(settings)
        # SIMPLE verify should NOT return the reasoning model.
        result = router.route(TaskComplexity.SIMPLE, node="verify")
        assert result != settings.llm.reasoning_llm_model


# ════════════════════════════════════════════════════════════════════
# 3. FALLBACK_CHAINS — cross-provider coverage invariant
# ════════════════════════════════════════════════════════════════════


class TestFallbackChainCrossProvider:
    """Every primary's chain must reach at least one DIFFERENT provider."""

    @staticmethod
    def _providers(model_id: str) -> str:
        spec = MODEL_REGISTRY.get(model_id)
        if spec is not None:
            return spec.provider
        return ModelRouter._extract_provider(model_id)

    def test_every_chain_has_cross_provider_entry(self) -> None:
        """No chain is single-provider — a dead provider would otherwise kill routing."""
        bad: list[str] = []
        for primary, chain in FALLBACK_CHAINS.items():
            if not chain:
                bad.append(f"{primary!r}: empty chain")
                continue
            primary_provider = self._providers(primary)
            cross = [m for m in chain if self._providers(m) != primary_provider]
            if not cross:
                bad.append(f"{primary!r}: no cross-provider fallback in {chain}")
        assert not bad, "Chains lacking cross-provider fallback: " + "; ".join(bad)

    def test_every_chain_member_is_registered(self) -> None:
        """Chain entries must be registered models (the gateway needs a ModelSpec)."""
        for primary, chain in FALLBACK_CHAINS.items():
            for member in chain:
                assert member in MODEL_REGISTRY, (
                    f"chain for {primary!r} references unregistered model {member!r}"
                )

    def test_chain_for_each_complexity_primary_exists(self) -> None:
        """Every model the complexity map names as a primary has a fallback chain."""
        for _tier, model in COMPLEXITY_TIER_MAP.values():
            assert model in FALLBACK_CHAINS, f"complexity primary {model!r} has no fallback chain"
            assert len(FALLBACK_CHAINS[model]) >= 1

    def test_glm47_chain_reaches_free_tier_cross_provider(self) -> None:
        """The CRITICAL primary glm-4.7 reaches a non-zai provider (resilience)."""
        chain = FALLBACK_CHAINS["glm-4.7"]
        providers = {self._providers(m) for m in chain}
        assert "zai" not in providers or len(providers) > 1
        assert len(providers) >= 2, "glm-4.7 chain must span >=2 providers"


# ════════════════════════════════════════════════════════════════════
# 4. PromptCache — stable key / TTL / miss-through
# ════════════════════════════════════════════════════════════════════


class TestPromptCacheKeyStability:
    """The cache key is a deterministic hash of the full request signature."""

    def test_identical_inputs_same_key(self) -> None:
        settings = _fresh_settings()
        redis, _ = _dict_redis()
        cache = PromptCache(redis, settings)
        msgs = [{"role": "user", "content": "hello"}]
        k1 = cache._make_cache_key(msgs, "gpt-4o-mini-2024-07-18", 0.5, 100)
        k2 = cache._make_cache_key(msgs, "gpt-4o-mini-2024-07-18", 0.5, 100)
        assert k1 == k2

    def test_key_independent_of_message_list_order_of_construction(self) -> None:
        """Two lists built differently but with identical content hash the same."""
        settings = _fresh_settings()
        redis, _ = _dict_redis()
        cache = PromptCache(redis, settings)
        base = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        rebuilt = [{"role": role, "content": content} for role, content in [("system", "s"), ("user", "u")]]
        k1 = cache._make_cache_key(base, "m", 0.0, None)
        k2 = cache._make_cache_key(rebuilt, "m", 0.0, None)
        assert k1 == k2

    def test_different_messages_different_key(self) -> None:
        settings = _fresh_settings()
        redis, _ = _dict_redis()
        cache = PromptCache(redis, settings)
        k1 = cache._make_cache_key([{"role": "user", "content": "a"}], "m", 0.5, 10)
        k2 = cache._make_cache_key([{"role": "user", "content": "b"}], "m", 0.5, 10)
        assert k1 != k2

    def test_different_model_different_key(self) -> None:
        settings = _fresh_settings()
        redis, _ = _dict_redis()
        cache = PromptCache(redis, settings)
        msgs = [{"role": "user", "content": "x"}]
        assert cache._make_cache_key(msgs, "model-a", 0.5, 10) != cache._make_cache_key(
            msgs, "model-b", 0.5, 10
        )

    def test_different_temperature_different_key(self) -> None:
        """Temperature is part of the signature — a non-zero temp must not be cached as greedy."""
        settings = _fresh_settings()
        redis, _ = _dict_redis()
        cache = PromptCache(redis, settings)
        msgs = [{"role": "user", "content": "x"}]
        assert cache._make_cache_key(msgs, "m", 0.0, 10) != cache._make_cache_key(msgs, "m", 0.7, 10)

    def test_different_max_tokens_different_key(self) -> None:
        """max_tokens is part of the signature — a truncated response must not masquerade as full."""
        settings = _fresh_settings()
        redis, _ = _dict_redis()
        cache = PromptCache(redis, settings)
        msgs = [{"role": "user", "content": "x"}]
        assert cache._make_cache_key(msgs, "m", 0.5, 100) != cache._make_cache_key(msgs, "m", 0.5, 200)

    def test_key_has_cache_prefix(self) -> None:
        """The key carries the namespaced Redis prefix (no key collisions with other users)."""
        settings = _fresh_settings()
        redis, _ = _dict_redis()
        cache = PromptCache(redis, settings)
        key = cache._make_cache_key([{"role": "user", "content": "x"}], "gpt-4o-mini-2024-07-18", 0.5)
        assert key.startswith("turing:llm_cache:")


class TestPromptCacheGetSet:
    """get/set round-trip, miss fall-through, and TTL forwarding."""

    async def test_miss_returns_none_and_records_miss(self) -> None:
        """A cache miss returns None so the gateway calls the provider."""
        settings = _fresh_settings()
        redis, _ = _dict_redis()
        cache = PromptCache(redis, settings)
        result = await cache.get([{"role": "user", "content": "q"}], "m", 0.5, 10)
        assert result is None
        stats = await cache.stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0

    async def test_set_then_get_round_trips_response(self) -> None:
        """A stored response is returned verbatim on the matching signature."""
        settings = _fresh_settings()
        redis, _ = _dict_redis()
        cache = PromptCache(redis, settings)
        msgs = [{"role": "user", "content": "q"}]
        resp = _make_response()
        await cache.set(resp, msgs, "gpt-4o-mini-2024-07-18", 0.5, 100)
        got = await cache.get(msgs, "gpt-4o-mini-2024-07-18", 0.5, 100)
        assert got is not None
        assert got.cached is True
        assert got.content == resp.content

    async def test_set_forwards_ttl_to_redis(self) -> None:
        """The configured TTL is passed as ``ex=`` on the Redis SET."""
        settings = _fresh_settings()
        settings.redis.cache_ttl_seconds = 1234
        redis, _ = _dict_redis()
        cache = PromptCache(redis, settings)
        await cache.set(_make_response(), [{"role": "user", "content": "q"}], "m", 0.5, 10)
        # The captured set() kwargs must include ex=<ttl>.
        assert redis._set_calls, "Redis SET was not called"  # type: ignore[attr-defined]
        assert redis._set_calls[-1].get("ex") == 1234  # type: ignore[attr-defined]

    async def test_set_respects_configured_ttl_default(self) -> None:
        """Without override, the TTL equals settings.redis.cache_ttl_seconds."""
        settings = _fresh_settings()
        default_ttl = settings.redis.cache_ttl_seconds
        redis, _ = _dict_redis()
        cache = PromptCache(redis, settings)
        await cache.set(_make_response(), [{"role": "user", "content": "q"}], "m", 0.5)
        assert redis._set_calls[-1].get("ex") == default_ttl  # type: ignore[attr-defined]

    async def test_hit_increments_hit_counter(self) -> None:
        settings = _fresh_settings()
        redis, _ = _dict_redis()
        cache = PromptCache(redis, settings)
        msgs = [{"role": "user", "content": "q"}]
        await cache.set(_make_response(), msgs, "m", 0.5, 10)
        await cache.get(msgs, "m", 0.5, 10)
        stats = await cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0

    async def test_get_redis_error_falls_through_to_miss(self) -> None:
        """A Redis failure during get returns None (never raises) — a transparent miss."""
        settings = _fresh_settings()
        redis = MagicMock()
        redis.get = AsyncMock(side_effect=RuntimeError("redis down"))
        redis.scan_iter = MagicMock(return_value=iter([]))
        cache = PromptCache(redis, settings)
        result = await cache.get([{"role": "user", "content": "q"}], "m", 0.5, 10)
        assert result is None

    async def test_set_redis_error_does_not_raise(self) -> None:
        """A Redis failure during set is swallowed — caching is best-effort."""
        settings = _fresh_settings()
        redis = MagicMock()
        redis.set = AsyncMock(side_effect=RuntimeError("redis down"))
        cache = PromptCache(redis, settings)
        # Must not raise.
        await cache.set(_make_response(), [{"role": "user", "content": "q"}], "m", 0.5, 10)

    async def test_stored_payload_excludes_cached_flag(self) -> None:
        """The serialized payload omits ``cached`` (re-added as True on read)."""
        settings = _fresh_settings()
        redis, store = _dict_redis()
        cache = PromptCache(redis, settings)
        await cache.set(_make_response(), [{"role": "user", "content": "q"}], "m", 0.5, 10)
        raw = next(iter(store.values()))
        data = json.loads(raw)
        assert "cached" not in data
        # And it round-trips back as cached=True.
        got = await cache.get([{"role": "user", "content": "q"}], "m", 0.5, 10)
        assert got is not None and got.cached is True


# ════════════════════════════════════════════════════════════════════
# 5. CostTracker — per-run aggregation + resilience + free-tier pricing
# ════════════════════════════════════════════════════════════════════


def _mock_session() -> MagicMock:
    """An AsyncSession mock whose execute() returns a scripted scalar."""
    session = MagicMock()
    result = MagicMock()
    result.scalar_one = MagicMock(return_value=0)
    result.one = MagicMock(return_value=(0, 0, 0))
    result.all = MagicMock(return_value=[])
    session.execute = AsyncMock(return_value=result)
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


class TestCostTrackerAggregation:
    """Per-run_id token + cost aggregate queries read back the summed ledger."""

    async def test_get_run_spend_returns_summed_cost(self) -> None:
        session = _mock_session()
        session.execute.return_value.scalar_one.return_value = 1.25
        tracker = CostTracker(session, _fresh_settings())
        spend = await tracker.get_run_spend("cli-q05")
        assert spend == pytest.approx(1.25)

    async def test_get_run_token_usage_returns_summed_tokens(self) -> None:
        session = _mock_session()
        session.execute.return_value.scalar_one.return_value = 4321
        tracker = CostTracker(session, _fresh_settings())
        tokens = await tracker.get_run_token_usage("cli-q05")
        assert tokens == 4321

    async def test_get_run_spend_zero_when_no_rows(self) -> None:
        """coalesce(sum, 0) → 0.0 for an unattributed run."""
        session = _mock_session()
        session.execute.return_value.scalar_one.return_value = 0.0
        tracker = CostTracker(session, _fresh_settings())
        assert await tracker.get_run_spend("never-ran") == 0.0

    async def test_get_run_token_usage_zero_when_no_rows(self) -> None:
        session = _mock_session()
        session.execute.return_value.scalar_one.return_value = 0
        tracker = CostTracker(session, _fresh_settings())
        assert await tracker.get_run_token_usage("never-ran") == 0

    async def test_get_runs_summary_shapes_rows(self) -> None:
        """get_runs_summary maps grouped rows into typed summary dicts."""
        session = _mock_session()
        row = MagicMock()
        row.run_id = "cli-q05"
        row.cost_usd = 2.5
        row.calls = 3
        row.input_tokens = 100
        row.output_tokens = 200
        row.total_tokens = 300
        row.last_call = None
        session.execute.return_value.all.return_value = [row]
        tracker = CostTracker(session, _fresh_settings())
        summary = await tracker.get_runs_summary()
        assert len(summary) == 1
        entry = summary[0]
        assert entry["run_id"] == "cli-q05"
        assert entry["calls"] == 3
        assert entry["cost_usd"] == pytest.approx(2.5)
        assert entry["total_tokens"] == 300


class TestCostTrackerResilience:
    """A failed ledger write rolls back, logs, and never re-raises."""

    async def test_record_usage_returns_cost_on_commit_failure(self) -> None:
        """A commit error is swallowed; the calculated cost is still returned."""
        session = _mock_session()
        session.commit = AsyncMock(side_effect=RuntimeError("connection lost"))
        tracker = CostTracker(session, _fresh_settings())
        # Must NOT raise — returns the calculated cost instead.
        cost = await tracker.record_usage(
            "deepseek-v4-flash", "deepseek", input_tokens=100, output_tokens=50
        )
        assert cost > 0

    async def test_record_usage_rolls_back_on_failure(self) -> None:
        """The poisoned session is rolled back so the next call can recover."""
        session = _mock_session()
        session.commit = AsyncMock(side_effect=RuntimeError("poison"))
        tracker = CostTracker(session, _fresh_settings())
        await tracker.record_usage("deepseek-v4-flash", "deepseek", 10, 5)
        session.rollback.assert_awaited_once()

    async def test_record_usage_succeeds_after_a_prior_failure(self) -> None:
        """The session recovers: a second record_usage commits normally."""
        session = _mock_session()
        tracker = CostTracker(session, _fresh_settings())
        # Force the first commit to fail, then succeed on retry.
        session.commit.side_effect = [RuntimeError("transient"), None]
        cost1 = await tracker.record_usage("deepseek-v4-flash", "deepseek", 10, 5)
        cost2 = await tracker.record_usage("deepseek-v4-flash", "deepseek", 10, 5)
        assert cost1 > 0 and cost2 > 0
        session.rollback.assert_awaited_once()  # only the failed call rolled back

    async def test_record_usage_commits_on_success(self) -> None:
        """Happy path: add + commit, no rollback."""
        session = _mock_session()
        tracker = CostTracker(session, _fresh_settings())
        await tracker.record_usage("deepseek-v4-flash", "deepseek", 10, 5, run_id="cli-q01")
        session.add.assert_called_once()
        session.commit.assert_awaited_once()
        session.rollback.assert_not_awaited()

    async def test_record_usage_writes_run_id(self) -> None:
        """The run_id correlation key is attached to the ledger row."""
        session = _mock_session()
        tracker = CostTracker(session, _fresh_settings())
        await tracker.record_usage("deepseek-v4-flash", "deepseek", 10, 5, run_id="cli-q09")
        added = session.add.call_args[0][0]
        assert added.run_id == "cli-q09"
        assert added.total_tokens == 15


class TestCostTrackerPricing:
    """calculate_cost: registered models priced from their spec; free-tier = $0."""

    def test_free_tier_model_costs_zero(self) -> None:
        """A registered free-tier (NVIDIA / OpenRouter-free) model bills $0.0."""
        # nvidia-qwen3-next-80b has 0.0 input/output cost in the registry.
        cost = CostTracker.calculate_cost("nvidia-qwen3-next-80b", 1000, 500)
        assert cost == 0.0

    def test_registered_model_uses_spec_pricing(self) -> None:
        """A registered paid model is priced from its explicit per-1k rates."""
        spec: ModelSpec = MODEL_REGISTRY["deepseek-v4-flash"]
        cost = CostTracker.calculate_cost("deepseek-v4-flash", 1000, 1000)
        expected = (1000 * spec.input_cost_per_1k / 1000) + (1000 * spec.output_cost_per_1k / 1000)
        assert cost == pytest.approx(expected)

    def test_unknown_model_uses_conservative_fallback(self) -> None:
        """An unregistered model is over-estimated (safe direction for the budget gate)."""
        cost = CostTracker.calculate_cost("totally-unknown-model", 1000, 1000)
        # Fallback rate is 0.005/1k in + 0.015/1k out.
        assert cost == pytest.approx(0.005 + 0.015)
