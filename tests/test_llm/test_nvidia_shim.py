"""Deterministic unit tests for the NVIDIA NIM OpenAI-compatible shim.

The shim rewrites a registered ``nvidia/<id>`` model_id to ``openai/<id>`` and
pins the NIM api_base so litellm (which rejects the bare ``nvidia/`` prefix in
this build) can route the call. Pure — no litellm call, no API key, no network.
Covers the rewrite for flat + nested ids, the api_base pin, and the no-op for
non-nvidia providers. Gateway-level wiring (that ``_build_kwargs`` applies it)
is covered separately by test_gateway.py::TestBuildKwargs.
"""

from __future__ import annotations

from src.llm.nvidia_shim import NVIDIA_API_BASE, nvidia_shim_model_id


class TestNvidiaShim:
    def test_flat_nvidia_id_rewrites_to_openai(self) -> None:
        effective, extra = nvidia_shim_model_id(
            "nvidia", "nvidia/nemotron-3-super-120b-a12b"
        )
        assert effective == "openai/nemotron-3-super-120b-a12b"
        assert extra == {"api_base": NVIDIA_API_BASE, "model": effective}

    def test_nested_nvidia_id_preserves_org_segment(self) -> None:
        # ``nvidia/qwen/qwen3-next-80b-a3b-instruct`` -> ``openai/qwen/qwen3-...``
        effective, extra = nvidia_shim_model_id(
            "nvidia", "nvidia/qwen/qwen3-next-80b-a3b-instruct"
        )
        assert effective == "openai/qwen/qwen3-next-80b-a3b-instruct"
        assert extra["model"] == effective
        assert extra["api_base"] == NVIDIA_API_BASE

    def test_openai_org_nvidia_id_rewrites_to_nested_openai(self) -> None:
        # ``nvidia/openai/gpt-oss-120b`` -> ``openai/openai/gpt-oss-120b``
        effective, extra = nvidia_shim_model_id("nvidia", "nvidia/openai/gpt-oss-120b")
        assert effective == "openai/openai/gpt-oss-120b"
        assert extra["api_base"] == NVIDIA_API_BASE
        assert extra["model"] == effective

    def test_api_base_always_pinned_for_nvidia(self) -> None:
        # A bare nvidia id (none registered today, no ``nvidia/`` prefix) is NOT
        # rewritten but the NIM base is still pinned so the call lands on NIM.
        effective, extra = nvidia_shim_model_id("nvidia", "some-bare-model")
        assert effective == "some-bare-model"
        assert extra == {"api_base": NVIDIA_API_BASE}

    def test_non_nvidia_provider_is_a_noop(self) -> None:
        for provider in ("openai", "anthropic", "alibaba", "zai", "google", ""):
            effective, extra = nvidia_shim_model_id(
                provider, "openai/gpt-4o-mini-2024-07-18"
            )
            assert effective == "openai/gpt-4o-mini-2024-07-18"
            assert extra == {}

    def test_returned_keys_are_litellm_kwargs(self) -> None:
        # The merged dict must only contain valid litellm request fields.
        _, extra = nvidia_shim_model_id("nvidia", "nvidia/x")
        assert set(extra.keys()) <= {"api_base", "model"}


class TestNvidiaApiBaseOverride:
    """An explicit ``api_base`` (sourced from ``settings.llm.nvidia_api_base`` by
    the gateway) overrides the curated public NIM endpoint — for pointing at a
    private/regional NIM instance. ``None``/empty falls back to the curated base."""

    def test_explicit_api_base_overrides_curated_base(self) -> None:
        private = "https://nim.internal.corp/v1"
        effective, extra = nvidia_shim_model_id(
            "nvidia", "nvidia/nemotron-3-super-120b-a12b", api_base=private
        )
        assert effective == "openai/nemotron-3-super-120b-a12b"
        assert extra["api_base"] == private
        assert extra["model"] == effective

    def test_none_api_base_falls_back_to_curated(self) -> None:
        _, extra = nvidia_shim_model_id(
            "nvidia", "nvidia/x", api_base=None
        )
        assert extra["api_base"] == NVIDIA_API_BASE

    def test_empty_api_base_falls_back_to_curated(self) -> None:
        # An empty-string override (e.g. NVIDIA_API_BASE= in .env) must NOT wipe
        # the base — it falls back to the curated public NIM endpoint.
        _, extra = nvidia_shim_model_id(
            "nvidia", "nvidia/x", api_base=""
        )
        assert extra["api_base"] == NVIDIA_API_BASE

    def test_override_does_not_affect_non_nvidia_provider(self) -> None:
        effective, extra = nvidia_shim_model_id(
            "openai", "openai/gpt-4o-mini-2024-07-18", api_base="https://x/v1"
        )
        assert effective == "openai/gpt-4o-mini-2024-07-18"
        assert extra == {}


class TestEveryRegisteredNvidiaModelIsShimmed:
    """Regression: every registered NVIDIA model_id rewrites to openai/<id> with
    the NIM base pinned — the contract the gateway relies on to actually route
    these models. Pulls from the live registry so a new nvidia entry is covered
    automatically."""

    def test_all_nvidia_models_rewrite_and_pin_base(self) -> None:
        from src.config.model_registry import MODEL_REGISTRY

        nvidia = {
            k: s for k, s in MODEL_REGISTRY.items() if s.provider == "nvidia"
        }
        assert nvidia, "precondition: registry has nvidia models"
        for key, spec in nvidia.items():
            assert spec.model_id.startswith("nvidia/"), (
                f"{key}: nvidia model_id must carry the 'nvidia/' prefix, "
                f"got {spec.model_id!r}"
            )
            effective, extra = nvidia_shim_model_id(spec.provider, spec.model_id)
            assert effective.startswith("openai/"), (
                f"{key}: shim must rewrite to openai/<id>, got {effective!r}"
            )
            assert effective == "openai/" + spec.model_id[len("nvidia/") :], (
                f"{key}: shim dropped a segment rewriting {spec.model_id!r} -> {effective!r}"
            )
            assert extra["api_base"] == NVIDIA_API_BASE, f"{key}: base not pinned"
            assert extra["model"] == effective, f"{key}: model not echoed in kwargs"
