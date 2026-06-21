"""E2E live smoke for the Alibaba (DashScope) model additions.

The deterministic unit tests in ``tests/test_config/test_alibaba_models.py``
assert the registration is correct; THIS module proves each model actually
answers through the real gateway (the same path the live battery uses, including
DashScope routing + the tenacity retry/fallback resilience). Marked ``e2e``:
these make REAL billed calls and are excluded from the default
``-k "not e2e"`` run. Each case skips (not fails) when ``DASHSCOPE_API_KEY`` is
absent, so the suite degrades gracefully on a machine without the key.

Run with::

    python -m pytest tests/test_e2e/test_alibaba_models_e2e.py -v -m e2e
"""

from __future__ import annotations

import pytest

# Explicitly load .env into os.environ BEFORE settings is constructed, mirroring
# how the app resolves keys at runtime via pydantic-settings.
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is a hard dependency, defensive only
    pass

from src.config import get_settings
from src.config.model_registry import FALLBACK_CHAINS, MODEL_REGISTRY
from src.llm.gateway import LLMGateway

pytestmark = pytest.mark.e2e

# The newly-registered DashScope-hosted models. ``deepseek-v4-flash`` here is the
# Alibaba-hosted copy (key ``alibaba-deepseek-v4-flash``), not the standalone.
_TARGETS: list[str] = [
    "qwen3.6-flash",
    "qwen3.5-flash-2026-02-23",
    "qwen-turbo",
    "qwen3-coder-next",
    "alibaba-deepseek-v4-flash",
]


@pytest.fixture(scope="module")
def gateway() -> LLMGateway:
    return LLMGateway(get_settings())


def _has_dashscope_key(gateway: LLMGateway) -> bool:
    spec = MODEL_REGISTRY["qwen3.5-flash"]  # canonical alibaba model
    return bool(gateway._get_api_key(spec.provider))  # noqa: SLF001 — app's own resolver


@pytest.mark.parametrize("model_key", _TARGETS)
@pytest.mark.asyncio
async def test_alibaba_model_responds(gateway: LLMGateway, model_key: str) -> None:
    """Each newly-registered DashScope model answers a tiny prompt via litellm."""
    if not _has_dashscope_key(gateway):
        pytest.skip("DASHSCOPE_API_KEY not set — alibaba E2E smoke cannot run")

    spec = MODEL_REGISTRY[model_key]
    # Reasoning-capable models (deepseek-v4-flash) spend output tokens on a
    # hidden reasoning phase; 128 leaves room to clear it (16 truncates empty).
    response = await gateway.acompletion(
        messages=[{"role": "user", "content": "Reply with the single word: OK"}],
        model=model_key,
        max_tokens=128,
        temperature=0.0,
    )

    assert response.content, f"{model_key}: empty response content"
    assert response.total_tokens > 0, f"{model_key}: zero tokens consumed"
    # Served by the requested provider OR a legitimate fallback from its chain.
    # The DashScope ``openai/`` shim reports provider="alibaba" (the spec's
    # provider, set in gateway._parse_response). A provider OUTSIDE this set
    # would indicate misrouting — the real bug this guards against.
    acceptable: set[str] = {spec.provider}
    for fb in FALLBACK_CHAINS.get(model_key, []):
        fb_spec = MODEL_REGISTRY.get(fb)
        if fb_spec:
            acceptable.add(fb_spec.provider)
    assert response.provider in acceptable, (
        f"{model_key}: served by provider={response.provider}, "
        f"expected one of {acceptable} (primary or a legit fallback)"
    )
