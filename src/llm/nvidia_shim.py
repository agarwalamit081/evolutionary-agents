"""NVIDIA NIM OpenAI-compatible routing shim.

litellm in this build rejects the bare ``nvidia/`` provider prefix
(``BadRequestError: LLM Provider NOT provided``). The NVIDIA NIM endpoint is
OpenAI-compatible, so a registered nvidia model_id (``nvidia/<org>/<model>``)
is rewritten to the ``openai/<org>/<model>`` shim against the pinned NIM
``api_base`` at call time. Verified live for all 16 registered NVIDIA models.

Pure — no litellm call, no API key, deterministic. Unit-covered in
tests/test_llm/test_nvidia_shim.py; the gateway-level wiring (that
``LLMGateway._build_kwargs`` applies it) is covered by
tests/test_llm/test_gateway.py::TestBuildKwargs.
"""

from __future__ import annotations

# The NVIDIA NIM endpoint is OpenAI-compatible; this base is what the ``openai/``
# shim routes against when litellm's ``nvidia/`` prefix is unsupported in build.
NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"


def nvidia_shim_model_id(
    provider: str, litellm_model: str, api_base: str | None = None
) -> tuple[str, dict[str, str]]:
    """Rewrite a registered NVIDIA model_id for the OpenAI-compatible NIM shim.

    Returns ``(effective_model, kwargs)`` where ``kwargs`` holds the call-time
    fields to merge into the litellm request (``api_base`` always for nvidia;
    ``model`` when the id was rewritten). A no-op — ``(litellm_model, {})`` —
    for any non-nvidia provider, so the gateway can call it unconditionally.

    Args:
        provider: The registry provider (``spec.provider``); only ``"nvidia"``
            triggers the rewrite / base pin.
        litellm_model: The resolved litellm model_id (``spec.model_id``).
        api_base: Optional explicit NIM endpoint (e.g. a private/regional NIM
            instance) sourced from ``settings.llm.nvidia_api_base`` by the
            gateway. ``None`` → the curated public NIM base ``NVIDIA_API_BASE``.

    Returns:
        The effective model_id to send and the kwargs to merge into the request.
    """
    if provider != "nvidia":
        return litellm_model, {}
    extra: dict[str, str] = {"api_base": api_base or NVIDIA_API_BASE}
    effective = litellm_model
    # Only a ``nvidia/``-prefixed id needs the rewrite to ``openai/``; a bare id
    # (none registered today) still gets the base pinned so the call lands on NIM.
    if litellm_model.startswith("nvidia/"):
        effective = "openai/" + litellm_model[len("nvidia/") :]
        extra["model"] = effective
    return effective, extra
