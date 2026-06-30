"""Probe raw provider behavior under a FORCED ``tool_choice`` (#1a).

The execute-node write-nudge forces ``tool_choice`` to a specific function
(``file_writer``) so the model emits the deliverable file instead of narrating.
Production wraps every call in two safety nets:

1. ``LLMGateway._configure_litellm`` sets ``litellm.drop_params = True``
   (gateway.py:1269) — litellm silently DROPS any param a provider rejects.
2. ``_execute_with_fallback`` recovers a tool_choice/thinking conflict reactively
   (gateway.py:856-931): DeepSeek gets ``extra_body={"thinking":{"type":...
   "disabled"}}``; everything else falls back to dropping tool_choice.

Both nets hide the RAW provider signal we need to design the #1b mitigations
(``tool_choice="required"`` intermediate fallback; generalize the DeepSeek
extra_body path; stricter write-nudge directive). This probe BYPASSES both nets:
it reuses ``gateway._build_kwargs`` for the correct per-provider config
(registry-key→model_id resolution, alibaba DashScope api_base, nvidia shim,
provider api_key, max_tokens cap) but then calls ``litellm.acompletion``
DIRECTLY with ``drop_params=False`` — no reactive recovery, no silent drop — so
a provider that rejects forced tool_choice surfaces as a raw 400 / narration.

Three variants per model, each directly informing one #1b mitigation:
  forced            — the prod write-nudge dict {type:function,function:{name:
                      file_writer}}. Baseline raw behavior.
  required          — OpenAI ``"required"`` (call ANY tool). Mitigation (1): is
                      an intermediate "required" fallback viable before the drop?
  forced_no_thinking — forced dict + extra_body={"thinking":{"type":"disabled"}}.
                      Mitigation (2): does the DeepSeek thinking-disable path
                      generalize to non-DeepSeek (gpt-4o-mini, glm-4.7), or must
                      it stay provider-scoped? (drop_params=False ⇒ a provider
                      that rejects the unknown extra_body surfaces as a 400.)

Models default to every write-nudge model (the standalone + alibaba + nvidia
DeepSeek-V4-flash hostings, plus gpt-4o-mini and glm-4.7). Per-model provider
key presence is printed as a BOOLEAN only — never the key itself. Results are
written to ``logs/tool_choice_probe_<ts>.json`` and a summary table is printed.

Needs a funded provider key for each model probed. Confirm with the owner which
model/provider to probe + that the key is present before running.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import litellm

from src.config.model_registry import MODEL_REGISTRY
from src.config.settings import get_settings
from src.llm.gateway import LLMGateway

# Every model the execute-node write-nudge may force file_writer on (the
# SIMPLE+COMPLEX routing primaries + their cross-provider DeepSeek hostings).
_DEFAULT_MODELS: list[str] = [
    "deepseek-v4-flash",          # standalone DeepSeek provider
    "alibaba-deepseek-v4-flash",  # DashScope-hosted copy (openai/ model_id)
    "nvidia-deepseek-v4-flash",   # NVIDIA NIM-hosted copy (nvidia/ model_id)
    "gpt-4o-mini-2024-07-18",     # OpenAI
    "glm-4.7",                    # Z.AI
]

# Minimal but valid OpenAI-format file_writer tool — exactly what the write-nudge
# forces. The narration-tempting user message makes "model just explains" the
# natural failure mode, so tool_called=False is a meaningful signal.
_FILE_WRITER_TOOL: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "file_writer",
        "description": "Write text content to a file at the given path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Destination file path."},
                "content": {"type": "string", "description": "Text to write."},
            },
            "required": ["path", "content"],
        },
    },
}

_USER_MSG: str = (
    "I need to save the report 'Quarterly revenue was $4.2M, up 12% YoY.' "
    "to results/report.txt. Please handle writing this file now."
)

_FORCED_CHOICE: dict[str, object] = {
    "type": "function",
    "function": {"name": "file_writer"},
}

# variant name → (tool_choice value, extra_body or None). Order = run order.
_VARIANTS: dict[str, tuple[object, dict[str, object] | None]] = {
    "forced": (_FORCED_CHOICE, None),
    "required": ("required", None),
    "forced_no_thinking": (
        _FORCED_CHOICE,
        {"thinking": {"type": "disabled"}},
    ),
}

_MAX_SNIPPET = 120


def _truncate(text: str, limit: int = _MAX_SNIPPET) -> str:
    """Collapse whitespace and cap length so the summary stays readable."""
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"


async def _probe_variant(
    gateway: LLMGateway,
    model_key: str,
    variant: str,
    tool_choice: object,
    extra_body: dict[str, object] | None,
    max_tokens: int,
) -> dict[str, object]:
    """Issue ONE forced-tool_choice completion against litellm directly.

    Returns a result dict. status is "OK" (litellm returned a choice), "ERROR"
    (provider/litellm raised — the raw 400 signal we are hunting), with
    tool_called/content captured for the OK case.
    """
    # Reuse the gateway's per-provider kwargs (model_id, api_key, api_base for
    # alibaba, nvidia shim, max_tokens cap) — but call litellm.acompletion
    # directly so drop_params=False is honored and the reactive handler never
    # runs. This isolates the probe from both prod safety nets.
    kwargs = gateway._build_kwargs(  # noqa: SLF001 — diagnostic probe reuses prod config
        model_key, 0.0, max_tokens, metadata=None,
    )
    kwargs["messages"] = [{"role": "user", "content": _USER_MSG}]
    kwargs["tools"] = [_FILE_WRITER_TOOL]
    kwargs["tool_choice"] = tool_choice
    if extra_body:
        kwargs["extra_body"] = extra_body
    try:
        # litellm's declared return is a ModelResponse|stream union; we never
        # stream here, so cast to Any and access .choices like the gateway's
        # own _parse_response (gateway.py:1027-1029) — matches that pattern.
        resp = cast(Any, await litellm.acompletion(**kwargs, drop_params=False))
    except Exception as exc:  # noqa: BLE001 — probe surfaces ANY failure as signal
        return {
            "variant": variant,
            "status": "ERROR",
            "error_type": type(exc).__name__,
            "error": _truncate(str(exc)),
            "tool_called": False,
            "content": "",
        }

    msg = resp.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None) or []
    tool_called = any(
        getattr(tc, "function", None) is not None
        and getattr(tc.function, "name", None) == "file_writer"
        for tc in tool_calls
    )
    content = (getattr(msg, "content", None) or "").strip()
    return {
        "variant": variant,
        "status": "OK",
        "error_type": "",
        "error": "",
        "tool_called": tool_called,
        "content": _truncate(content),
    }


async def _probe_model(
    gateway: LLMGateway,
    model_key: str,
    variants: list[str],
    max_tokens: int,
) -> dict[str, object]:
    """Probe every variant for one model. Skips cleanly if the key is absent."""
    spec = MODEL_REGISTRY[model_key]
    # Provider key presence — BOOLEAN only (never the key value).
    key_present = bool(gateway._get_api_key(spec.provider))  # noqa: SLF001
    results: list[dict[str, object]] = []
    if not key_present:
        return {
            "model": model_key,
            "provider": spec.provider,
            "litellm_model_id": spec.model_id,
            "key_present": key_present,
            "results": results,
        }

    for variant in variants:
        tool_choice, extra_body = _VARIANTS[variant]
        result = await _probe_variant(
            gateway, model_key, variant, tool_choice, extra_body, max_tokens,
        )
        results.append(result)
        mark = "✓" if result["tool_called"] else "✗"
        print(
            f"  [{mark}] {variant:<20} status={result['status']:<6} "
            f"tool_called={result['tool_called']!s:<5} "
            f"{result['error'] or result['content']}"
        )
    return {
        "model": model_key,
        "provider": spec.provider,
        "litellm_model_id": spec.model_id,
        "key_present": key_present,
        "results": results,
    }


async def _main(args: argparse.Namespace) -> int:
    settings = get_settings()
    # Construct the gateway so litellm is fully configured (provider api_bases /
    # keys from settings), then flip drop_params OFF so per-call drop_params=False
    # is the authoritative behavior and nothing is silently dropped.
    gateway = LLMGateway(settings)
    litellm.drop_params = False  # type: ignore[attr-defined]

    models = args.models.split(",") if args.models else list(_DEFAULT_MODELS)
    variants = args.variants.split(",") if args.variants else list(_VARIANTS)
    unknown_models = [m for m in models if m not in MODEL_REGISTRY]
    if unknown_models:
        print(f"[abort] unknown model key(s): {unknown_models}")
        return 2
    unknown_variants = [v for v in variants if v not in _VARIANTS]
    if unknown_variants:
        print(f"[abort] unknown variant(s): {unknown_variants}; choose from {list(_VARIANTS)}")
        return 2

    print(
        f"[setup] probing {len(models)} model(s) × {len(variants)} variant(s) "
        f"with drop_params=False (raw provider behavior, no reactive recovery)"
    )
    for model in models:
        spec = MODEL_REGISTRY[model]
        present = bool(gateway._get_api_key(spec.provider))  # noqa: SLF001
        print(f"[key]  {model:<28} provider={spec.provider:<8} key_present={present}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
        "variants": variants,
        "max_tokens": args.max_tokens,
    }
    all_rows: list[dict[str, object]] = []

    for model in models:
        spec = MODEL_REGISTRY[model]
        present = bool(gateway._get_api_key(spec.provider))  # noqa: SLF001
        print(f"\n=== {model} ({spec.provider}, key_present={present}) ===")
        if not present:
            print("  [skip] no provider key present")
        row = await _probe_model(gateway, model, variants, args.max_tokens)
        all_rows.append(row)
    report["results"] = all_rows

    out_path = out_dir / f"tool_choice_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Summary: count tool_called across all probed (key-bearing) variants.
    # Iterate the typed all_rows list; cast each row's "results" to a list so
    # pyright can follow the dict[str, object] value type.
    total_ok = total_called = total_error = 0
    for row in all_rows:
        if not row.get("key_present"):
            continue
        for res in cast(list[dict[str, object]], row["results"]):
            if res.get("status") == "OK":
                total_ok += 1
            if res.get("tool_called"):
                total_called += 1
            if res.get("status") == "ERROR":
                total_error += 1
    print("\n=== SUMMARY ===")
    print(f"OK={total_ok}  tool_called={total_called}  ERROR(400/raw)={total_error}")
    print(f"full JSON → {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe raw provider behavior under a forced tool_choice (#1a).",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated MODEL_REGISTRY keys (default: the 5 write-nudge models).",
    )
    parser.add_argument(
        "--variants",
        default="",
        help=f"Comma-separated variants (default: all). Choose from {list(_VARIANTS)}.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="max_tokens per probe call (default: 512).",
    )
    parser.add_argument(
        "--out",
        default="logs",
        help="Output directory for the JSON report (default: logs).",
    )
    args = parser.parse_args()
    return asyncio.run(_main(args))


if __name__ == "__main__":
    sys.exit(main())
