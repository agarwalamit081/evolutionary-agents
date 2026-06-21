"""One-off verification: does the current code register duplicate_finder?

Runs the REAL ToolGenerator.generate() + validate_and_register() path (the
exact code tool_create_node uses) for the N5 duplicate_finder capability gap,
against the configured tool_generation_model (default deepseek-v4-pro).

Prints: resolved model, generated handler length, safety-validation verdict,
and whether the tool landed in the registry. Decisive for "does Fix 1 actually
register the tool end-to-end" — the question logs/n5.log left ambiguous.

The gateway is built WITHOUT a Redis prompt cache so the call is a real sample.

Usage::

    python scripts/verify_tool_registers.py

Cost: ~1 deepseek-v4-pro call. Requires DEEPSEEK_API_KEY.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.config import get_settings
from src.llm.gateway import LLMGateway
from src.safety.pipeline import SafetyPipeline
from src.tools.dynamic.generator import ToolGenerator
from src.tools.registry import ToolRegistry

_GAP = (
    "A tool called duplicate_finder that takes a glob pattern and returns the "
    "duplicate non-empty lines found across all matched files."
)
_CONTEXT = {
    "goal_text": (
        "Create a short tool called duplicate_finder that takes a glob and "
        "returns duplicate non-empty lines across matched files."
    ),
    "failed_tools": "none",
    "error_details": "",
    "existing_tools": [],
}


async def main() -> int:
    logger.remove()
    settings = get_settings()
    model = settings.agent.tool_generation_model or "(legacy complexity=SIMPLE)"
    gateway = LLMGateway(settings)  # no set_cache() → real, uncached call
    gen = ToolGenerator(gateway=gateway, safety_pipeline=SafetyPipeline())
    registry = ToolRegistry()

    print(f"tool_generation_model = {model}\n")
    tool = await gen.generate(_GAP, _CONTEXT)
    if tool is None:
        print("RESULT: generate() returned None (generation/extraction failed)")
        return 1

    print(f"generated handler: {len(tool.handler_code)} chars")
    print(f"handler first line: {tool.handler_code.strip().splitlines()[0][:80]}")
    print(f"handler last line:  {tool.handler_code.strip().splitlines()[-1][:80]}\n")

    result = await gen.validate_and_register(tool, registry)
    print(f"validate_and_register success = {result.get('success')}")
    if not result.get("success"):
        print(f"reason: {result.get('reason')}")
        return 1

    handler = registry.get_handler(tool.tool_name)
    print(f"registry contains '{tool.tool_name}': {handler is not None}")
    if handler is not None:
        out = await handler("results/*.md")
        preview = (out or "")[:120].replace("\n", " \\n ")
        print(f"sample invoke '{tool.tool_name}(results/*.md)' -> {preview}")
    print("\nRESULT: tool REGISTERED ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
