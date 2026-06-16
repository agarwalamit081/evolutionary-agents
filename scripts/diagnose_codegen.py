"""One-off diagnostic: prove JSON mode fixes tool code-gen truncation.

Hypothesis (from logs/n5.log): the dynamic tool generator requests FREE-TEXT
output, so a multi-line Python handler embedded as a JSON string value breaks
parsing on an unescaped quote/newline; ``json_repair`` then salvages a
TRUNCATED ``handler_code`` (observed: 156-char handlers, "Syntax error at line
6: '(' was never closed") that fails the AST safety gate and never registers.

This script calls the exact code-gen prompt against ``deepseek-v4-pro`` both
WITHOUT ``response_format`` (reproduce the bug) and WITH JSON mode
(``{"type": "json_object"}``, the fix), printing the raw response length,
whether the raw response is directly-valid JSON (no repair), and the extracted
``handler_code`` length for each. If free-text truncates and JSON mode does
not, the hypothesis is confirmed.

The gateway is constructed WITHOUT a Redis prompt cache, so every call is an
independent real sample (no cache hits).

Usage::

    python scripts/diagnose_codegen.py            # 3 rounds, each mode
    python scripts/diagnose_codegen.py --rounds 1  # cheapest (2 calls total)

Cost: ~$0.01 per round (2 deepseek-v4-pro calls). Requires DEEPSEEK_API_KEY.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.config import get_settings
from src.graph.prompts import TOOL_GENERATE_SYSTEM, TOOL_GENERATE_USER
from src.llm.gateway import LLMGateway
from src.llm.structured_output import StructuredOutputManager
from src.tools.dynamic.generator import GeneratedTool

# The N5 goal — the exact capability gap that produced the 156-char handlers.
_GAP = (
    "A tool called duplicate_finder that takes a glob pattern and returns the "
    "duplicate non-empty lines found across all matched files."
)
_CONTEXT: dict[str, Any] = {
    "goal_text": (
        "Create a short tool called duplicate_finder that takes a glob and "
        "returns duplicate non-empty lines across matched files."
    ),
    "failed_tools": "none",
    "error_details": "",
    "existing_tools": [],
}
_MODEL = "deepseek-v4-pro"


def _build_messages() -> list[dict[str, str]]:
    """Mirror ToolGenerator._build_messages exactly."""
    user = TOOL_GENERATE_USER.format(
        gap_description=_GAP,
        goal_text=_CONTEXT["goal_text"],
        failed_tools=_CONTEXT["failed_tools"],
        error_details=_CONTEXT["error_details"],
        existing_tools=", ".join(_CONTEXT["existing_tools"]) or "none",
    )
    return [
        {"role": "system", "content": str(TOOL_GENERATE_SYSTEM)},
        {"role": "user", "content": user},
    ]


def _handler_tail(handler: str | None) -> str:
    if not handler:
        return "(no handler)"
    last = handler.strip().splitlines()[-1] if handler.strip() else "(empty)"
    return last[:80]


async def _run_mode(
    gateway: LLMGateway,
    messages: list[dict[str, str]],
    json_mode: bool,
    round_idx: int,
) -> dict[str, Any]:
    label = "JSON mode" if json_mode else "free-text"
    kwargs: dict[str, Any] = {"messages": messages, "model": _MODEL}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        resp = await gateway.acompletion(**kwargs)
    except Exception as exc:  # noqa: BLE001 — diagnostic, surface any failure
        return {"label": label, "round": round_idx, "error": f"{type(exc).__name__}: {exc}"}

    raw = resp.content or ""
    direct_json_ok = _is_direct_json(raw)
    handler_len: int | None = None
    tail = "(no handler)"
    parse_error = ""
    try:
        tool = await StructuredOutputManager().extract(raw, GeneratedTool)
        if tool is not None:
            handler_len = len(tool.handler_code)
            tail = _handler_tail(tool.handler_code)
        else:
            parse_error = "StructuredOutputManager returned None"
    except Exception as exc:  # noqa: BLE001
        parse_error = f"{type(exc).__name__}: {exc}"

    return {
        "label": label,
        "round": round_idx,
        "raw_len": len(raw),
        "direct_json_ok": direct_json_ok,
        "handler_len": handler_len,
        "handler_tail": tail,
        "parse_error": parse_error,
    }


def _is_direct_json(text: str) -> bool:
    """True if the raw response parses as JSON without json_repair."""
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


async def main(rounds: int) -> int:
    # Silence gateway chatter; the script prints its own structured output.
    logger.remove()
    settings = get_settings()
    gateway = LLMGateway(settings)  # no set_cache() → no Redis cache → real calls
    messages = _build_messages()

    print(f"Model: {_MODEL} | rounds: {rounds} | gap: duplicate_finder\n")
    print(f"{'mode':10} {'rnd':>3} {'raw':>6} {'json?':>6} {'handler':>8}  tail / note")
    print("-" * 88)
    rows: list[dict[str, Any]] = []
    for r in range(1, rounds + 1):
        for json_mode in (False, True):
            row = await _run_mode(gateway, messages, json_mode, r)
            rows.append(row)
            if "error" in row:
                print(f"{row['label']:10} {r:>3}  ERROR: {row['error']}")
                continue
            print(
                f"{row['label']:10} {r:>3} {row['raw_len']:>6} "
                f"{str(row['direct_json_ok']):>6} {str(row['handler_len']):>8}  "
                f"{row['handler_tail']}"
            )
            if row["parse_error"]:
                print(f"{'':>30} parse: {row['parse_error']}")

    # Aggregate
    print("\n=== Summary ===")
    for label in ("free-text", "JSON mode"):
        subset = [x for x in rows if x.get("label") == label and "handler_len" in x]
        if not subset:
            print(f"{label}: no successful extractions")
            continue
        lens = [x["handler_len"] for x in subset]  # type: ignore[index]
        direct = sum(1 for x in subset if x["direct_json_ok"])
        print(
            f"{label}: {len(subset)} samples | handler chars "
            f"min={min(lens)} max={max(lens)} avg={sum(lens)//len(lens)} | "
            f"direct-valid-JSON {direct}/{len(subset)}"
        )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3, help="rounds per mode (each = 2 calls)")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.rounds)))
