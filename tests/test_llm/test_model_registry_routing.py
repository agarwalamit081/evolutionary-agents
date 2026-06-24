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
