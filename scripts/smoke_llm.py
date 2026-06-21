"""Live provider smoke test — confirms an LLM call resolves and returns 200.

Usage::

    python scripts/smoke_llm.py                 # pings the Qwen/DashScope models
    python scripts/smoke_llm.py qwen3.7-plus    # ping one registered model

Why this exists: the registry maps ``qwen3.5-flash`` -> model_id
``openai/qwen3.5-flash`` with ``provider="alibaba"``. The unit tests prove
``_build_kwargs`` pins ``api_base`` to the DashScope OpenAI-compatible
endpoint and attaches ``DASHSCOPE_API_KEY``; this script proves DashScope
actually accepts the call end-to-end. Prints NO secrets — only booleans,
public URLs, token counts, and (truncated) error class/message.
"""

from __future__ import annotations

import asyncio
import sys

from src.config.settings import get_settings
from src.llm.gateway import LLMGateway

_DEFAULT_MODELS: tuple[str, ...] = ("qwen3.5-flash", "qwen3.7-plus")
_PROMPT = "Reply with exactly one word: pong"


def _redacted_key_diag(key: str) -> str:
    """Non-secret key health: length, prefix family, whitespace/quote leakage."""
    return (
        f"len={len(key)} starts_sk={key.startswith('sk-')} "
        f"has_leading_ws={key != key.lstrip()} "
        f"has_trailing_ws={key != key.rstrip()} "
        f"has_newline={chr(10) in key} has_quote={chr(34) in key or chr(39) in key}"
    )


async def _ping(gateway: LLMGateway, model: str) -> int:
    key = gateway._settings.llm.dashscope_api_key or ""  # noqa: SLF001
    # Mirror the gateway's actual default (international DashScope), not a stale
    # China fallback — otherwise this diagnostic prints an endpoint the call never
    # used. See gateway._build_kwargs for the authoritative pin.
    api_base = gateway._settings.llm.alibaba_api_base or (  # noqa: SLF001
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )
    print(f"[setup] key set: {bool(key)} | {_redacted_key_diag(key)} | api_base: {api_base}")
    try:
        resp = await gateway.acompletion(
            messages=[{"role": "user", "content": _PROMPT}],
            model=model,
            temperature=0.0,
            max_tokens=16,
        )
    except Exception as exc:  # noqa: BLE001 — smoke test surfaces any failure
        print(f"[fail] {model}: {type(exc).__name__}: {str(exc)[:200]}")
        return 1
    print(
        f"[ok] {model}: provider={resp.provider} model={resp.model} "
        f"tokens(in={resp.input_tokens},out={resp.output_tokens}) "
        f"cost=${resp.cost_usd:.6f} content={resp.content!r}"
    )
    return 0


async def _main(models: tuple[str, ...]) -> int:
    gateway = LLMGateway(get_settings())
    rc = 0
    for model in models:
        rc |= await _ping(gateway, model)
    return rc


if __name__ == "__main__":
    models = tuple(sys.argv[1:]) or _DEFAULT_MODELS
    sys.exit(asyncio.run(_main(models)))
