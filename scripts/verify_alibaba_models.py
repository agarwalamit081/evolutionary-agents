"""Live verification probe for Alibaba (DashScope) models.

Grounds model registration in reality: an unverified ``model_id`` in a
``FALLBACK_CHAIN`` is a liability — a 404 burns a retry slot on every failed
run. This calls each candidate via the same routing the gateway uses
(``provider="alibaba"`` → dashscope-intl OpenAI-compatible endpoint,
``DASHSCOPE_API_KEY``, ``model_id`` ``openai/<name>``) and reports only
pass/fail + a sanitized error category.

Reads ``DASHSCOPE_API_KEY`` / ``ALIBABA_API_BASE`` from ``.env`` via the app's
own config (never printed). No other API key required.

Usage::

    python scripts/verify_alibaba_models.py                       # default set
    python scripts/verify_alibaba_models.py qwen3.6-flash qwen-turbo   # explicit
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Make ``src`` importable when run directly as ``python scripts/verify_alibaba_models.py``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import litellm  # noqa: E402

from src.config import get_settings  # noqa: E402

# The Alibaba-hosted model set (all routed via DashScope, not the standalone
# provider). ``deepseek-v4-flash`` here is the Alibaba-hosted copy, distinct
# from the standalone DeepSeek provider's ``deepseek/deepseek-v4-flash``.
DEFAULT_CANDIDATES: tuple[str, ...] = (
    "qwen3.6-flash",
    "qwen3.5-flash",
    "qwen3.5-flash-2026-02-23",
    "qwen-turbo",
    "deepseek-v4-flash",
    "qwen3-coder-next",
)


async def _probe(name: str, api_base: str, api_key: str) -> tuple[bool, str]:
    """One trivial completion call against ``openai/<name>`` via DashScope."""
    model = f"openai/{name}"
    try:
        resp = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word: OK"}],
            api_base=api_base,
            api_key=api_key,
            max_tokens=8,
            timeout=30,
        )
        # litellm ModelResponse supports dict-like access; avoids the
        # CustomStreamWrapper union-attr diagnostic on attribute access.
        txt = str(resp["choices"][0]["message"]["content"] or "").strip()  # type: ignore[index]
        return True, f"ok -> {txt.replace(chr(10), ' ')[:40]!r}"
    except Exception as e:  # noqa: BLE001 — probe must report all error categories
        etype = type(e).__name__
        msg = str(e).replace("\n", " ").strip()
        if api_key and api_key in msg:  # never echo the key
            msg = msg.replace(api_key, "<redacted>")
        return False, f"{etype}: {msg[:160]}"


async def _run(candidates: list[str]) -> int:
    s = get_settings().llm
    api_base = s.alibaba_api_base or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    api_key = s.dashscope_api_key or ""
    if not api_key:
        print("NO DASHSCOPE_API_KEY configured — cannot probe alibaba routing")
        return 2

    print(f"api_base = {api_base}")
    print(f"api_key  = <set, len={len(api_key)}>\n")

    failures = 0
    for name in candidates:
        ok, detail = await _probe(name, api_base, api_key)
        if not ok:
            failures += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:28s} {detail}")
    print(f"\n{len(candidates) - failures}/{len(candidates)} healthy")
    return 0 if failures == 0 else 1


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Live-verify Alibaba (DashScope) models respond before/after registry edits.",
    )
    ap.add_argument(
        "models", nargs="*", default=list(DEFAULT_CANDIDATES),
        help="bare model names to probe (default: the known Alibaba set)",
    )
    args = ap.parse_args()
    raise SystemExit(asyncio.run(_run(args.models)))


if __name__ == "__main__":
    main()
