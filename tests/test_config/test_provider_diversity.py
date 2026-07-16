"""Provider-diversity fallback chains (battery-04 q08 storm-resilience).

A single-provider (ZAI) quota/rate-limit outage killed battery-04 q08: the
glm-5.2 chain burned straight to the expensive Anthropic hop once zai/glm-5.2
rate-limited. The fix is same-model cross-provider copies — register the SAME
model under OpenRouter (paid, independent quota/key) and NVIDIA (free), and
chain to them FIRST so an outage degrades to an *equivalent* model instead of a
more expensive one. Slugs + the funded OPENROUTER_API_KEY verified live
2026-07-16; `nvidia/z-ai/glm-5.2` verified live via the app shim path.
"""
from __future__ import annotations

from src.config.model_registry import (
    FALLBACK_CHAINS,
    MODEL_REGISTRY,
    ModelTier,
)

# The 7 paid OpenRouter same-model copies: registry key → litellm slug.
OPENROUTER_SPECS: dict[str, str] = {
    "openrouter-glm-5-2": "openrouter/z-ai/glm-5.2",
    "openrouter-glm-5-1": "openrouter/z-ai/glm-5.1",
    "openrouter-glm-4-7": "openrouter/z-ai/glm-4.7",
    "openrouter-claude-sonnet-4-6": "openrouter/anthropic/claude-sonnet-4.6",
    "openrouter-claude-haiku-4-5": "openrouter/anthropic/claude-haiku-4.5",
    "openrouter-deepseek-v4-flash": "openrouter/deepseek/deepseek-v4-flash",
    "openrouter-deepseek-v4-pro": "openrouter/deepseek/deepseek-v4-pro",
}

# Each OpenRouter copy's own chain must reach the NATIVE same model first — an
# OpenRouter outage is independent of the native provider's quota, so the copy
# falls back to the native equivalent rather than a different family.
OPENROUTER_NATIVE_FIRST: dict[str, str] = {
    "openrouter-glm-5-2": "glm-5.2",
    "openrouter-glm-5-1": "glm-5.1",
    "openrouter-glm-4-7": "glm-4.7",
    "openrouter-claude-sonnet-4-6": "claude-sonnet-4-6",
    "openrouter-claude-haiku-4-5": "claude-haiku-4-5-20251001",
    "openrouter-deepseek-v4-flash": "deepseek-v4-flash",
    "openrouter-deepseek-v4-pro": "deepseek-v4-pro",
}


class TestOpenRouterSpecs:
    """The 7 paid OpenRouter same-model copies are registered correctly."""

    def test_all_registered_with_openrouter_provider(self) -> None:
        for key, slug in OPENROUTER_SPECS.items():
            assert key in MODEL_REGISTRY, f"{key!r} missing from MODEL_REGISTRY"
            spec = MODEL_REGISTRY[key]
            assert spec.provider == "openrouter", f"{key}: provider {spec.provider!r}"
            assert spec.model_id == slug, f"{key}: model_id {spec.model_id!r} != {slug!r}"
            # OpenRouter copies carry the full agent capability surface so they
            # can stand in for the native primary at any node.
            assert spec.supports_tool_calling, f"{key}: no tool calling"
            assert spec.supports_json_mode, f"{key}: no JSON mode"
            assert spec.supports_streaming, f"{key}: no streaming"

    def test_openrouter_copies_are_paid_not_free(self) -> None:
        """OpenRouter charges ~the upstream rate for these (small markup). Cost
        must be mirrored and non-zero so the cost_ledger never silently
        under-prices a real OpenRouter fallback call."""
        for key in OPENROUTER_SPECS:
            spec = MODEL_REGISTRY[key]
            assert spec.input_cost_per_1k > 0.0, f"{key}: free-tier input pricing on a paid model"
            assert spec.output_cost_per_1k > 0.0, f"{key}: free-tier output pricing on a paid model"


class TestNvidiaGlm52Spec:
    """The free NVIDIA-hosted glm-5.2 copy is registered for the shim path."""

    def test_registered_free_tier(self) -> None:
        spec = MODEL_REGISTRY["nvidia-glm-5-2"]
        assert spec.provider == "nvidia"
        assert spec.model_id == "nvidia/z-ai/glm-5.2"
        # FREE on the NVIDIA API → VERY_CHEAP tier, explicit $0.0 cost (the
        # ModelSpec cost default) so it is the cheapest same-model hop.
        assert spec.tier == ModelTier.VERY_CHEAP
        assert spec.input_cost_per_1k == 0.0
        assert spec.output_cost_per_1k == 0.0
        assert spec.supports_tool_calling


class TestGlm52StormResilienceChain:
    """THE core invariant: the glm-5.2 (COMPLEX/CRITICAL routing primary) chain
    reaches the SAME model on two independent providers BEFORE degrading to a
    different family. A future chain edit that drops the cross-provider hops
    would silently re-introduce the single-provider storm fragility."""

    def test_same_model_cross_provider_first(self) -> None:
        chain = FALLBACK_CHAINS["glm-5.2"]
        # Hop 1: same model on OpenRouter (paid, independent key/quota).
        assert chain[0] == "openrouter-glm-5-2", (
            f"glm-5.2 must reach openrouter-glm-5-2 first; got {chain}"
        )
        # Hop 2: same model on NVIDIA (free).
        assert chain[1] == "nvidia-glm-5-2", (
            f"glm-5.2 must reach nvidia-glm-5-2 second; got {chain}"
        )
        # Then the same-family successor + a cross-family peer (NOT before the
        # same-model hops).
        assert "glm-5.1" in chain
        assert "claude-sonnet-4-6" in chain or "deepseek-v4-pro" in chain


class TestNewModelsHaveFallbackChains:
    """Every new provider-diversity model declares a chain whose entries all
    resolve to a registered ModelSpec (no dangling refs that burn retry slots)."""

    NEW_KEYS: tuple[str, ...] = tuple(OPENROUTER_SPECS) + ("nvidia-glm-5-2",)

    def test_each_new_model_has_chain(self) -> None:
        for key in self.NEW_KEYS:
            chain = FALLBACK_CHAINS.get(key)
            assert chain is not None, f"{key!r} missing a fallback chain"
            assert len(chain) >= 2, f"{key!r}: chain shorter than 2 hops: {chain}"

    def test_openrouter_copy_chains_reach_native_first(self) -> None:
        for or_key, native_key in OPENROUTER_NATIVE_FIRST.items():
            chain = FALLBACK_CHAINS[or_key]
            assert chain[0] == native_key, (
                f"{or_key}: first hop must be native {native_key!r}; got {chain}"
            )

    def test_no_dangling_refs_in_new_chains(self) -> None:
        for key in self.NEW_KEYS:
            for fb in FALLBACK_CHAINS[key]:
                assert fb in MODEL_REGISTRY, f"{key}: dangling fallback ref {fb!r}"
