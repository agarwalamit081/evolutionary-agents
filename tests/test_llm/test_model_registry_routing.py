"""Registry model_ids must be litellm-routable.

Regression for the bare-prefix bug: zai models were registered with a bare
``model_id`` (e.g. ``glm-4.7-flash``) that litellm could not route
(``BadRequestError: LLM Provider NOT provided``) until prefixed ``zai/``.
This asserts every registry ``model_id`` resolves via ``litellm.get_llm_provider``
(no API call) so the same class of bug is caught deterministically for every
provider — not just the ones exercised by the live e2e matrix.

NVIDIA NIM models (``nvidia/<org>/<model>``) are OpenAI-compatible, so the
gateway rewrites them to ``openai/<org>/<model>`` against the pinned NIM
``api_base`` at call time (see ``src/llm/nvidia_shim.py``). This test applies
that shim before the routability assertion, so the 16 registered nvidia models
are covered deterministically (they were previously skipped because the bare
``nvidia/`` prefix is rejected by litellm in this build).
"""

from __future__ import annotations

import pytest

import litellm

from src.config.model_registry import MODEL_REGISTRY
from src.llm.nvidia_shim import NVIDIA_API_BASE, nvidia_shim_model_id

# Providers reached via an OpenAI-compatible gateway with a custom api_base
# (set in gateway._build_litellm_kwargs), so their model_id legitimately uses
# the ``openai/`` prefix and litellm reports provider="openai", not the spec's
# provider. We assert routability only — not provider identity — for these.
_OPENAI_COMPATIBLE_PROVIDERS: set[str] = {"alibaba"}

# litellm uses internal provider names that differ from our registry's provider
# field for the same upstream service. Google's Gemini models are registered
# under provider="google" but litellm reports provider="gemini" (the ``gemini/``
# prefix on model_id deliberately selects the Gemini API path, NOT vertex_ai,
# which would require GCP service-account auth). Map each registry provider to
# the set of litellm provider names that count as a correct match.
_LITELLM_PROVIDER_ALIASES: dict[str, set[str]] = {"google": {"gemini"}}


@pytest.mark.parametrize("key", list(MODEL_REGISTRY.keys()))
def test_model_id_is_litellm_routable(key: str) -> None:
    spec = MODEL_REGISTRY[key]
    # NVIDIA NIM is OpenAI-compatible but litellm in this build rejects the bare
    # ``nvidia/`` prefix, so the gateway rewrites it to ``openai/<id>`` against
    # the pinned NIM base (src/llm/nvidia_shim.py). Assert the POST-shim id is
    # routable rather than skipping — the contract the gateway actually relies on.
    if spec.provider == "nvidia":
        effective_model, shim_kwargs = nvidia_shim_model_id(
            spec.provider, spec.model_id
        )
        assert shim_kwargs.get("api_base") == NVIDIA_API_BASE, (
            f"{key}: nvidia shim must pin the NIM api_base"
        )
        assert effective_model.startswith("openai/"), (
            f"{key}: nvidia shim must rewrite to openai/<id>, got {effective_model!r}"
        )
        try:
            _, provider, _, _ = litellm.get_llm_provider(effective_model)
        except Exception as exc:  # noqa: BLE001 — litellm raises various types
            pytest.fail(
                f"{key}: shimmed model_id {effective_model!r} not routable ({exc})"
            )
        assert provider == "openai", (
            f"{key}: shimmed model_id {effective_model!r} routes to "
            f"provider={provider!r}, expected 'openai'"
        )
        return
    try:
        _, provider, _, _ = litellm.get_llm_provider(spec.model_id)
    except Exception as exc:  # noqa: BLE001 — litellm raises various types
        pytest.fail(
            f"{key}: model_id={spec.model_id!r} is not routable by litellm ({exc})"
        )
    if spec.provider not in _OPENAI_COMPATIBLE_PROVIDERS:
        acceptable = {spec.provider} | _LITELLM_PROVIDER_ALIASES.get(spec.provider, set())
        assert provider in acceptable, (
            f"{key}: model_id={spec.model_id!r} routes to provider={provider!r}, "
            f"expected one of {acceptable}"
        )


def test_zai_models_carry_zai_prefix() -> None:
    """Direct regression for the fixed bare-prefix bug on the glm family."""
    zai_models = [
        k for k, s in MODEL_REGISTRY.items() if s.provider == "zai"
    ]
    assert zai_models, "precondition: registry has zai models"
    for key in zai_models:
        assert MODEL_REGISTRY[key].model_id.startswith("zai/"), (
            f"{key}: zai model_id must carry the 'zai/' litellm prefix, "
            f"got {MODEL_REGISTRY[key].model_id!r}"
        )


# DeepSeek V4 is text-only for vision INPUT. Live-probed 2026-06-26: both
# deepseek-v4-flash and deepseek-v4-pro return "there is no image provided"
# (in=24 tokens — the image_url block is dropped) when an image is fed via
# chat completions. The registry previously over-declared supports_images=True,
# which let the D2 vision path (require_vision fallback filter) attempt a
# text-only model and silently lose the image. Locks the correction so a
# maintainer does not "restore" the flag from a doc that implies vision.
_DEEPSEEK_TEXT_ONLY_KEYS = (
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "alibaba-deepseek-v4-flash",  # same model, DashScope-hosted copy
)


@pytest.mark.parametrize("key", _DEEPSEEK_TEXT_ONLY_KEYS)
def test_deepseek_v4_is_text_only(key: str) -> None:
    """DeepSeek V4 (both variants) is text-only — must not declare vision INPUT."""
    assert key in MODEL_REGISTRY, f"precondition: {key} in registry"
    assert MODEL_REGISTRY[key].supports_images is False, (
        f"{key}: DeepSeek V4 is text-only (live-probed 2026-06-26: drops image_url "
        f"blocks); supports_images must be False or the vision path silently fails"
    )


# Live-probed 2026-06-26 batch (32x32 solid-red PNG sent as a data URI, temp=0).
# Each of these over-declared supports_images=True, yet none actually processes an
# image_url block over chat completions: OpenAI gpt-5-nano and classic qwen-turbo
# drop the block (in≈25-32 tokens) and guess the color; MiniMax m2.5 / m2.5-highspeed
# answer "there is no image provided"; Groq's llama-3.3-70b rejects multimodal
# content outright ("messages[0].content must be a string"). The D2 vision path
# (require_vision fallback filter) would otherwise hand a real image to one of
# these and silently lose it, so the flag must stay False. Locks the prune so a
# doc that implies vision does not re-introduce the silent-failure vector.
_PRUNED_VISION_OVER_DECLARATION_KEYS = (
    "gpt-5-nano-2025-08-07",
    "minimax-m2.5",
    "minimax-m2.5-highspeed",
    "qwen-turbo",
    "llama-3.3-70b-versatile",
)


@pytest.mark.parametrize("key", _PRUNED_VISION_OVER_DECLARATION_KEYS)
def test_pruned_models_do_not_declare_vision_input(key: str) -> None:
    """Live-probed text-only models must not declare vision INPUT support."""
    assert key in MODEL_REGISTRY, f"precondition: {key} in registry"
    assert MODEL_REGISTRY[key].supports_images is False, (
        f"{key}: live-probed text-only (2026-06-26 vision batch — drops/rejects "
        f"image_url); supports_images must be False or the vision path silently fails"
    )


# ─── glm-5.1 / glm-5.2 (Z.AI-native) registration regression ────────────────


def test_glm51_glm52_registered_and_chained() -> None:
    """glm-5.1 / glm-5.2 are the Z.AI-native planner models (distinct from the
    NVIDIA-hosted FREE copy nvidia-glm-5-1). Both must be registered with the
    zai/ litellm prefix, text-only INPUT, and have a fallback chain so the
    gateway can fail over. Caps per .claude/rules/llm-model-guardrails.md:35-36.
    """
    from src.config.model_registry import FALLBACK_CHAINS

    for key, ctx, out in (("glm-5.1", 200_000, 131_000), ("glm-5.2", 1_000_000, 128_000)):
        assert key in MODEL_REGISTRY, f"{key} must be registered"
        spec = MODEL_REGISTRY[key]
        assert spec.model_id == f"zai/{key}", f"{key}: model_id must be zai/-prefixed"
        assert spec.provider == "zai"
        assert spec.max_context == ctx, f"{key}: context cap {spec.max_context} != {ctx}"
        assert spec.max_output == out, f"{key}: output cap {spec.max_output} != {out}"
        assert spec.supports_images is False, f"{key}: text-only INPUT per guardrails"
        assert key in FALLBACK_CHAINS, f"{key}: must have a fallback chain"
        assert FALLBACK_CHAINS[key], f"{key}: fallback chain must be non-empty"


# ─── effective_fallback_chains (FALLBACK_CHAINS_JSON overlay) ───────────────


def test_effective_fallback_chains_empty_overlay_returns_curated() -> None:
    """No overlay → the curated FALLBACK_CHAINS object is returned unchanged."""
    from src.config.model_registry import FALLBACK_CHAINS, effective_fallback_chains

    assert effective_fallback_chains(None) is FALLBACK_CHAINS
    assert effective_fallback_chains("") is FALLBACK_CHAINS
    assert effective_fallback_chains("   ") is FALLBACK_CHAINS


def test_effective_fallback_chains_overlay_replaces_and_adds() -> None:
    """A valid overlay REPLACES an existing model's chain and ADDS new keys;
    untouched curated chains are preserved (merge, not full replace)."""
    import json

    from src.config.model_registry import FALLBACK_CHAINS, effective_fallback_chains

    overlay = json.dumps(
        {
            "glm-4.7": ["glm-5.1", "deepseek-v4-pro"],  # replace existing
            "brand-new-fake-model": ["gpt-4o-mini-2024-07-18"],  # add new key
        }
    )
    merged = effective_fallback_chains(overlay)

    assert merged["glm-4.7"] == ["glm-5.1", "deepseek-v4-pro"]  # overwritten
    assert merged["brand-new-fake-model"] == ["gpt-4o-mini-2024-07-18"]  # added
    # Untouched curated chain preserved (merge, not replace):
    assert merged["deepseek-v4-flash"] == FALLBACK_CHAINS["deepseek-v4-flash"]
    # Original curated dict NOT mutated (effective returns a fresh dict on overlay):
    assert FALLBACK_CHAINS["glm-4.7"] != ["glm-5.1", "deepseek-v4-pro"]


def test_effective_fallback_chains_invalid_overlay_is_ignored() -> None:
    """A bad overlay (unparseable / non-dict / non-list values) must NEVER break
    routing — the curated chains are returned unchanged."""
    from src.config.model_registry import FALLBACK_CHAINS, effective_fallback_chains

    assert effective_fallback_chains("not json {") is FALLBACK_CHAINS
    assert effective_fallback_chains("[1, 2, 3]") is FALLBACK_CHAINS  # not a dict
    # Non-list values are skipped; valid sibling entries still merge:
    import json

    merged = effective_fallback_chains(
        json.dumps({"glm-4.7": "not-a-list", "qwen3.5-flash": ["gpt-4o-mini-2024-07-18"]})
    )
    assert merged["glm-4.7"] == FALLBACK_CHAINS["glm-4.7"]  # bad entry ignored
    assert merged["qwen3.5-flash"] == ["gpt-4o-mini-2024-07-18"]  # good entry merged


# ─── Phase 3.5 Cluster B: latency-aware FALLBACK_CHAINS rebalance ──────────
# cost_ledger.latency_ms (12,596 rows) showed two latency bombs sitting in the
# hot-path fallback chains: mistral-medium-3-5 (946s mean / 1358s max) was a
# fallback entry in every MODERATE chain, and nvidia-deepseek-v4-pro (164s) was
# a fallback in the deepseek-v4-pro reasoning chain. NVIDIA free-tier models
# succeed so they never trip the breaker/timeout logs — they just burn
# wall-clock. These locks keep the bombs out of the hot path so a maintainer
# does not "restore" them from a doc that implies depth.


def test_mistral_medium_latency_bomb_removed_from_all_chains() -> None:
    """mistral-medium-3-5 (946s mean per cost_ledger.latency_ms) must not be a
    fallback entry in ANY chain — it is a hard latency bomb. Its own key is
    retained (never a routing primary; reached only by the exhaustion
    tier-scan), but it never lists itself, so the assertion is simply that the
    model_id appears in no fallback list at all."""
    from src.config.model_registry import FALLBACK_CHAINS

    offenders = [
        prim for prim, chain in FALLBACK_CHAINS.items() if "mistral-medium-3-5" in chain
    ]
    assert not offenders, (
        "mistral-medium-3-5 (946s mean latency bomb) must not appear as a "
        f"fallback in any chain; found in: {offenders}"
    )


def test_nvidia_deepseek_pro_removed_from_reasoning_chain() -> None:
    """nvidia-deepseek-v4-pro (164s mean) must not sit in the deepseek-v4-pro
    reasoning chain. It MAY remain in NVIDIA-primary chains (free-tier
    affinity) — only the *paid-reasoning* hot path is the concern."""
    from src.config.model_registry import FALLBACK_CHAINS

    chain = FALLBACK_CHAINS["deepseek-v4-pro"]
    assert "nvidia-deepseek-v4-pro" not in chain, (
        "nvidia-deepseek-v4-pro (164s mean) must not be a fallback in the "
        f"deepseek-v4-pro reasoning chain; chain={chain}"
    )


@pytest.mark.parametrize(
    "primary",
    ["qwen3.5-flash", "gpt-4o-mini-2024-07-18"],
)
def test_slow_nvidia_demoted_to_last_or_absent_in_cheap_chains(primary: str) -> None:
    """In the TRIVIAL/CHEAP primary chains, any NVIDIA free-tier entry must be
    LAST or absent — fast paid tiers (qwen/gpt/glm, ~5-14s) must precede the
    silently-slow NVIDIA tail (nvidia-qwen3-next-80b 47s/719s-max,
    nvidia-nemotron-ultra-550b 120s)."""
    from src.config.model_registry import FALLBACK_CHAINS

    chain = FALLBACK_CHAINS[primary]
    nvidia_idx = [i for i, m in enumerate(chain) if m.startswith("nvidia-")]
    for idx in nvidia_idx:
        assert idx == len(chain) - 1, (
            f"{primary}: NVIDIA entry {chain[idx]!r} at index {idx} must be "
            f"last (chain len {len(chain)}); demote fast tiers ahead of the "
            f"silently-slow NVIDIA tail. chain={chain}"
        )


def test_groq_fast_fallback_present_in_cheap_chains() -> None:
    """llama-3.1-8b-instant (Groq, VERY_CHEAP, ~1s, now keyed) must be reachable
    as a fallback in at least one TRIVIAL/CHEAP primary chain so the fast
    Groq tier is actually used (it had ZERO prior calls — buried at the tail)."""
    from src.config.model_registry import FALLBACK_CHAINS, MODEL_REGISTRY, ModelTier

    cheap_primaries = [
        prim
        for prim, chain in FALLBACK_CHAINS.items()
        if prim in MODEL_REGISTRY
        and MODEL_REGISTRY[prim].tier in {ModelTier.VERY_CHEAP, ModelTier.CHEAP}
        and "llama-3.1-8b-instant" in chain
    ]
    assert cheap_primaries, (
        "llama-3.1-8b-instant (Groq fast tier) must be a fallback in at least "
        "one VERY_CHEAP/CHEAP primary chain"
    )


def test_every_fallback_chain_entry_is_registered() -> None:
    """Rebalance invariant: every primary key and every fallback entry resolves
    to a MODEL_REGISTRY spec (no dangling refs introduced by hand-editing the
    chains; model_registry is a LEAF and cannot synthesize keys)."""
    from src.config.model_registry import FALLBACK_CHAINS, MODEL_REGISTRY

    for primary, chain in FALLBACK_CHAINS.items():
        assert primary in MODEL_REGISTRY, (
            f"chain primary {primary!r} missing from MODEL_REGISTRY"
        )
        for entry in chain:
            assert entry in MODEL_REGISTRY, (
                f"chain[{primary!r}] entry {entry!r} missing from MODEL_REGISTRY"
            )


def test_route_resolves_every_complexity() -> None:
    """Rebalance must not break routing: route() returns a registered model for
    each TaskComplexity (deterministic — asserts the returned id is in the
    registry, independent of which provider keys are present in .env)."""
    from src.config.model_registry import MODEL_REGISTRY
    from src.config.settings import Settings
    from src.graph.enums import TaskComplexity
    from src.llm.model_router import ModelRouter

    router = ModelRouter(Settings())
    for complexity in TaskComplexity:
        resolved = router.route(complexity)
        assert resolved in MODEL_REGISTRY, (
            f"route({complexity.name})={resolved!r} is not in MODEL_REGISTRY "
            "— rebalance broke routing"
        )


def test_router_caches_overlay_in_init() -> None:
    """ModelRouter resolves the overlay ONCE in __init__ (not per call), so the
    gateway hot path does not re-parse FALLBACK_CHAINS_JSON."""
    import json

    from src.config.settings import Settings
    from src.llm.model_router import ModelRouter

    settings = Settings()
    settings.routing.routing_fallback_chains_json = json.dumps(
        {"glm-4.7": ["glm-5.1", "deepseek-v4-pro"]}
    )
    router = ModelRouter(settings)
    assert router.get_fallback_chain("glm-4.7") == ["glm-5.1", "deepseek-v4-pro"]
    # Untouched model still uses curated chain via the same cache:
    assert router.get_fallback_chain("qwen3.5-flash")  # non-empty curated chain
