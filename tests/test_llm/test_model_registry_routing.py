"""Registry model_ids must be litellm-routable.

Regression for the bare-prefix bug: zai models were registered with a bare
``model_id`` (e.g. ``glm-4.7-flash``) that litellm could not route
(``BadRequestError: LLM Provider NOT provided``) until prefixed ``zai/``.
This asserts every registry ``model_id`` resolves via ``litellm.get_llm_provider``
(no API call) so the same class of bug is caught deterministically for every
provider — not just the ones exercised by the live e2e matrix.
"""

from __future__ import annotations

import pytest

import litellm

from src.config.model_registry import MODEL_REGISTRY

# Providers whose litellm prefix routing is known-broken in the installed
# litellm version (every model_id format unmapped via get_llm_provider). These
# are fallback-only; tracked for a litellm-upgrade / endpoint fix separately.
# NOTE: ``nvidia`` is *callable* despite being natively unroutable — the gateway
# (``LLMGateway._build_kwargs``) rewrites the ``nvidia/<id>`` model_id to the
# OpenAI-compatible ``openai/<id>`` shim against the NIM api_base at call time.
# That shim is covered by tests/test_llm/test_gateway.py::TestBuildKwargs; this
# test only asserts native litellm routability, which the bare prefix still lacks.
_KNOWN_UNROUTABLE_PROVIDERS: set[str] = {"nvidia"}

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
    if spec.provider in _KNOWN_UNROUTABLE_PROVIDERS:
        pytest.skip(f"{key}: {spec.provider} routing broken in installed litellm")
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
