"""Registry coverage for the Alibaba (DashScope) model additions.

The user requested these DashScope-hosted models be registered to broaden the
fallback pool (provider exhaustion has been the recurring blocker killing live
runs). Per project discipline, each was live-verified responsive via
``python main.py --verify-models`` BEFORE landing here; these deterministic
unit tests assert the registration is correct and stays correct (no live calls,
CI-runnable). The companion ``tests/test_e2e/test_alibaba_models_e2e.py`` makes
the real billed smoke call.

New keys (all ``provider="alibaba"``, routed via the DashScope OpenAI-compatible
endpoint → ``model_id`` ``openai/<name>``):

* ``qwen3.6-flash`` — newer flash variant (VERY_CHEAP).
* ``qwen3.5-flash-2026-02-23`` — pinned snapshot of qwen3.5-flash (VERY_CHEAP).
* ``qwen-turbo`` — classic turbo (CHEAP).
* ``qwen3-coder-next`` — coder-focused (CHEAP).
* ``alibaba-deepseek-v4-flash`` — DeepSeek's model **hosted on DashScope**,
  distinct from the standalone DeepSeek provider's ``deepseek-v4-flash``
  (different provider + API key → independent quota, the whole point of adding it).
"""

from __future__ import annotations

import pytest

from src.config.model_registry import (
    FALLBACK_CHAINS,
    MODEL_REGISTRY,
    ModelSpec,
    ModelTier,
)

NEW_KEYS: tuple[str, ...] = (
    "qwen3.6-flash",
    "qwen3.5-flash-2026-02-23",
    "qwen-turbo",
    "qwen3-coder-next",
    "alibaba-deepseek-v4-flash",
)

# DashScope hard-rejects max_tokens > 65536 ("Range of max_tokens should be
# [1, 65536]"). The gateway sends spec.max_output as max_tokens by default, so
# EVERY alibaba model must stay at/below this cap or its first call 400s.
# Verified live 2026-06. See the qwen3.5-flash comment in model_registry.py.
_DASHSCOPE_MAX_OUTPUT_CAP = 65_536


class TestNewAlibabaModelsRegistered:
    """The five new DashScope-hosted models are present and correctly spec'd."""

    @pytest.mark.parametrize("key", NEW_KEYS)
    def test_each_new_model_registered(self, key: str) -> None:
        assert key in MODEL_REGISTRY, f"{key!r} missing from MODEL_REGISTRY"

    @pytest.mark.parametrize("key", NEW_KEYS)
    def test_each_uses_alibaba_provider_and_openai_prefix(self, key: str) -> None:
        """provider=alibaba + openai/<name> prefix = the DashScope routing path."""
        spec = MODEL_REGISTRY[key]
        assert spec.provider == "alibaba"
        assert spec.model_id.startswith("openai/"), (
            f"{key}: model_id {spec.model_id!r} must use the openai/ prefix "
            f"for DashScope OpenAI-compatible routing"
        )

    @pytest.mark.parametrize("key", NEW_KEYS)
    def test_each_supports_agent_capabilities(self, key: str) -> None:
        """The agent needs tool-calling + JSON mode + streaming from any model
        it routes to; a model lacking these silently degrades nodes."""
        spec = MODEL_REGISTRY[key]
        assert spec.supports_tool_calling
        assert spec.supports_json_mode
        assert spec.supports_streaming

    @pytest.mark.parametrize("key", NEW_KEYS)
    def test_each_has_nontrivial_context_window(self, key: str) -> None:
        spec = MODEL_REGISTRY[key]
        assert spec.max_context >= 32_000


class TestAlibabaMaxOutputCap:
    """DashScope rejects max_tokens > 65536; enforce it for EVERY alibaba model
    so adding a future model with an over-large max_output fails CI here, not on
    the first live 400."""

    def test_all_alibaba_models_within_dashscope_cap(self) -> None:
        offenders = [
            key for key, spec in MODEL_REGISTRY.items()
            if spec.provider == "alibaba" and spec.max_output > _DASHSCOPE_MAX_OUTPUT_CAP
        ]
        assert not offenders, (
            f"alibaba models exceeding the DashScope max_tokens cap "
            f"({_DASHSCOPE_MAX_OUTPUT_CAP}): {offenders}"
        )


class TestAlibabaDeepseekDistinctFromStandalone:
    """The Alibaba-hosted deepseek-v4-flash is a SEPARATE registry entry from the
    standalone DeepSeek one — different provider + key = independent quota, which
    is the entire reason to register it (provider-diversity on the same model)."""

    def test_both_entries_exist_with_different_providers(self) -> None:
        alibaba = MODEL_REGISTRY.get("alibaba-deepseek-v4-flash")
        standalone = MODEL_REGISTRY.get("deepseek-v4-flash")
        assert alibaba is not None, "alibaba-deepseek-v4-flash not registered"
        assert standalone is not None, "standalone deepseek-v4-flash removed"

        assert alibaba.provider == "alibaba"
        assert standalone.provider == "deepseek"
        # Different litellm routing (openai/ shim vs deepseek/) → different quota.
        assert alibaba.model_id != standalone.model_id

    def test_alibaba_copy_serves_same_model_family(self) -> None:
        """Both expose deepseek-v4-flash (one via DashScope, one standalone)."""
        alibaba = MODEL_REGISTRY["alibaba-deepseek-v4-flash"]
        assert "deepseek-v4-flash" in alibaba.model_id


class TestNewModelsHaveFallbackChains:
    """A registered model with no fallback chain cannot survive its own provider
    rate-limiting — exactly the failure mode that prompted adding these models.
    Each new model must declare a chain, and every id it names must resolve."""

    @pytest.mark.parametrize("key", NEW_KEYS)
    def test_each_new_model_has_fallback_chain(self, key: str) -> None:
        chain = FALLBACK_CHAINS.get(key)
        assert chain, f"{key!r} has no FALLBACK_CHAINS entry"
        assert len(chain) >= 2, f"{key!r} fallback chain too short: {chain}"


class TestFallbackChainsReferenceRegisteredModels:
    """Cross-cutting typo guard: every model id referenced in ANY fallback chain
    must be a registered key. A dangling reference silently burns a retry slot on
    every failed run that reaches it (the gateway resolves it to a None spec and
    errors). Catches the exact class of bug that an unverified model_id would add."""

    def test_no_dangling_fallback_references(self) -> None:
        dangling: dict[str, list[str]] = {}
        for primary, chain in FALLBACK_CHAINS.items():
            missing = [fb for fb in chain if fb not in MODEL_REGISTRY]
            if missing:
                dangling[primary] = missing
        assert not dangling, (
            f"FALLBACK_CHAINS reference unregistered models: {dangling}"
        )

    def test_new_models_are_referenced_as_fallbacks(self) -> None:
        """The new cheap alibaba models broaden OTHER models' fallback pools —
        at least the qwen3.7-plus / deepseek chains should now reach an alibaba
        peer so a single-provider quota cap no longer starves the chain."""
        referenced: set[str] = set()
        for chain in FALLBACK_CHAINS.values():
            referenced.update(chain)
        # The flash models are the most broadly useful fallback additions.
        assert "qwen3.6-flash" in referenced or "qwen3.5-flash" in referenced, (
            "no cheap alibaba flash model wired into any fallback chain"
        )


class TestTierSanity:
    """Tiers drive cost-aware routing; a mis-tiered model misroutes silently."""

    @pytest.mark.parametrize("key,expected", [
        ("qwen3.6-flash", ModelTier.VERY_CHEAP),
        ("qwen3.5-flash-2026-02-23", ModelTier.VERY_CHEAP),
        ("qwen-turbo", ModelTier.CHEAP),
        ("qwen3-coder-next", ModelTier.CHEAP),
        ("alibaba-deepseek-v4-flash", ModelTier.CHEAP),
    ])
    def test_tier(self, key: str, expected: ModelTier) -> None:
        spec: ModelSpec = MODEL_REGISTRY[key]
        assert spec.tier == expected
