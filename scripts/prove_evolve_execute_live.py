"""Live proof that a deployed TOOL mutation fires the evolve→execute edge.

Deterministically demonstrates Phase-4 change E end-to-end using REAL LLMs: a
deployed TOOL mutation is live-registered in the ToolRegistry and
``route_after_evolve`` routes to ``execute`` for exactly ONE re-execution pass,
then ``store_memory`` (it cannot loop).

Why a focused script and not a blind CLI run
--------------------------------------------
The evolution engine selects ``opportunities[0]`` (engine.py:897), and
``failure_patterns`` is built from ``reflection.lessons_learned`` +
``reflection.errors`` (engine.py:872-878). So ANY reflect-produced lesson creates
a PROMPT opportunity that lands at index 0 and beats TOOL — which is exactly why
the prior live smoke (the analytics goal) emitted a PROMPT mutation, not a TOOL
one. A blind CLI run keeps losing that coin-flip (several $3 runs with no
guarantee). This script forces the selection deterministically — a clean
``ReflectionResult`` with empty ``lessons_learned``/no errors so TOOL is
``opportunities[0]`` — but every mechanism downstream of the selection is the
REAL production code with REAL LLMs:

  1. ``evolve_node`` builds ``SelfEvolutionEngine`` + ``SafetyPipeline`` + the
     sandbox (from settings — or ``None`` if Docker is down, same as a CLI run)
     + ``GitTracker``, exactly as in ``main.py``.
  2. ``analyze()`` picks the TOOL opportunity (no failure_patterns).
  3. ``_llm_generate()`` calls the real gateway (the codegen model) → a ``.py``
     tool handler (the codegen constraint forces ``target_path`` to end in .py).
  4. ``validate()`` SafetyPipeline layers 1-5 + ``sandbox_test()`` + ``ab_test()``
     + ``deploy()`` → shadow-repo commit (with no sandbox both gates return pass).
  5. ``_try_register_deployed_tool()`` live-registers the handler via the real
     ``ToolGenerator.validate_and_register`` + ``SafetyPipeline``.
  6. ``evolve_node`` returns ``evolve_reexecute_offered=True`` +
     ``evolve_reexecute_done=True``.
  7. ``route_after_evolve(state)`` == ``"execute"``   ← THE EDGE FIRES.
  8. ``route_after_evolve({offered:False, done:True})`` == ``"store_memory"``
     ← once-bound (the run re-executes at most once).

Only the state-seeding is deterministic; generation/deploy/registration/routing
are the live path. Cost: ~1-3 gateway calls. Bounded by ``MAX_COST_USD``.

Side effects to clean up afterward (deferred to the owner, same pattern as the
prior smoke): one TOOL mutation written to the shadow repo
(``.turing/evolution-repo``, local/gitignored) + a ``mutations`` table row + a
``tool_registrations`` row if live-persistence succeeded. The script prints the
registered tool name so the cleanup scope is known.

Usage::

    python scripts/prove_evolve_execute_live.py
    python scripts/prove_evolve_execute_live.py --verbose   # DEBUG logging

No secrets are read or printed; connects via the app's settings.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

# Force the Phase-4-E flag on + a tight cost cap BEFORE any settings access so
# pydantic-settings picks them up at first instantiation (``get_settings``
# caches). ``EVOLUTION_REEXECUTE_TOOL`` gates the evolve→execute edge; without
# it the evolve node never offers re-execution and the proof cannot pass.
os.environ.setdefault("EVOLUTION_REEXECUTE_TOOL", "true")
os.environ.setdefault("MAX_COST_USD", "1.0")

# ``python scripts/x.py`` puts ``scripts/`` first on sys.path, so the project
# root (which owns the ``src`` package) must be inserted before ``src`` is
# imported — otherwise ``from src.config import ...`` raises ModuleNotFoundError.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _configure_logging(verbose: bool) -> None:
    """Sink logs to ./logs/ (per project rules) + stderr; keep stdout for the verdict."""
    from loguru import logger

    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logger.add(log_dir / "prove_evolve_execute_live.log", level=level, rotation="5 MB")
    logger.add(sys.stderr, level="WARNING")  # warnings/errors visible; INFO→file


async def _build_deps() -> tuple[Any, Any, Any]:
    """Build the real gateway + governed ToolRegistry, mirroring main.py."""
    from src.config import get_settings
    from src.llm.gateway import LLMGateway
    from src.tools import create_default_registry
    from src.tools.dynamic.persister import ToolPersister

    settings = get_settings()
    gateway = LLMGateway(settings)
    tools = create_default_registry()
    # Load previously-created dynamic tools with the governance passes (semantic
    # dedup, cumulative cap, redundancy retirement), exactly like a CLI run, so
    # _try_register_deployed_tool registers against a realistic registry.
    try:
        persister = ToolPersister()
        await persister.load_active_tools(tools, settings=settings.agent)
    except Exception as e:  # noqa: BLE001 — non-fatal: governance load is best-effort
        print(f"  (governance load skipped: {e})")
    return settings, gateway, tools


def _seed_state() -> dict[str, Any]:
    """AgentState seeded so the engine selects a TOOL opportunity.

    Empty ``lessons_learned`` (+ no errors attr) ⇒ ``failure_patterns=[]`` ⇒ the
    PROMPT opportunity is never emitted ⇒ TOOL is ``opportunities[0]``. The
    concrete underperforming-tool records make ``_tool_opportunities`` emit a
    *specific* TOOL opportunity (a named, simple tool) rather than the vague
    generic fallback — raising the odds the LLM emits a clean, deployable handler.
    """
    from src.graph.enums import Confidence
    from src.graph.models import ReflectionResult

    reflection = ReflectionResult(
        summary="Run reached its objective deliverable cleanly; no failures.",
        lessons_learned=[],  # ← empty ⇒ failure_patterns=[] ⇒ TOOL at index 0
        confidence=Confidence.HIGH,
        should_evolve=True,
        should_replan=False,
    )
    # 3 calls, 1 success (33%) + 2 empty → _tool_opportunities emits a
    # high-priority "Tool 'text_stats' underperforming" TOOL opportunity.
    execution_history: list[dict[str, Any]] = [
        {"tool_name": "text_stats", "success": True, "output": "chars=42"},
        {"tool_name": "text_stats", "success": False, "output": "", "error": "boom"},
        {"tool_name": "text_stats", "success": False, "output": "", "error": "boom"},
    ]
    return {
        "reflection": reflection,
        "execution_history": execution_history,
        "generation": 0,
        "evolve_reexecute_done": False,
    }


async def main(verbose: bool) -> int:
    """Run the proof; return 0 on PASS, 1 INCONCLUSIVE, 2 misconfigured."""
    _configure_logging(verbose)

    settings, gateway, tools = await _build_deps()
    reexecute_on = bool(getattr(settings.evolution, "evolution_reexecute_tool", False))
    print(f"EVOLUTION_REEXECUTE_TOOL = {reexecute_on}")
    print(f"loaded tools             = {len(tools.list_tools()) if tools else 'n/a'}")
    if not reexecute_on:
        print("ERROR: EVOLUTION_REEXECUTE_TOOL is off — set it true and re-run.")
        return 2

    from src.graph.nodes.evolve import evolve_node
    from src.graph.routers import route_after_evolve

    state = _seed_state()

    print(
        "\n[1/3] evolve_node (real engine + gateway + sandbox-from-settings + git tracker)..."
    )
    result = await evolve_node(state, gateway=gateway, tools=tools)  # type: ignore[arg-type]
    if result is None:
        print("FAIL: evolve_node returned None (engine exception — see logs/).")
        return 1

    merged: dict[str, Any] = {**state, **result}
    offered = bool(merged.get("evolve_reexecute_offered"))
    done = bool(merged.get("evolve_reexecute_done"))
    history = merged.get("evolution_history") or []
    record = history[-1] if history else {}

    print(f"  mutation_type      = {record.get('mutation_type')}")
    print(f"  mutations_deployed = {record.get('mutations_deployed')}")
    print(f"  evolve_reexecute_offered = {offered}")
    print(f"  evolve_reexecute_done    = {done}")
    if record.get("reexecute_registered_tool"):
        print(f"  live-registered tool = {record['reexecute_registered_tool']}")

    print("\n[2/3] route_after_evolve(merged) — the edge decision...")
    route1 = route_after_evolve(merged)  # type: ignore[arg-type]
    print(f"  -> {route1!r}")

    print("\n[3/3] route_after_evolve(offered=False, done=True) — once-bound check...")
    second_cycle = dict(merged)
    second_cycle["evolve_reexecute_offered"] = False
    second_cycle["evolve_reexecute_done"] = True
    route2 = route_after_evolve(second_cycle)  # type: ignore[arg-type]
    print(f"  -> {route2!r}")

    edge_fired = route1 == "execute"
    bounded = route2 == "store_memory"

    print("\n" + "=" * 64)
    if offered and edge_fired and bounded:
        print("PASS — knob (d) confirmed LIVE:")
        print("  deployed TOOL mutation -> route_after_evolve 'execute' (one pass),")
        print("  then 'store_memory' (once-bound; the run cannot loop).")
        return 0

    print("INCONCLUSIVE this attempt:")
    print(f"  offered={offered}  route1={route1!r} (expected 'execute')")
    print(f"  route2={route2!r} (expected 'store_memory')")
    if not offered:
        print(
            "  The TOOL mutation did not deploy+live-register this cycle. Common\n"
            "  causes: the generated handler failed the sandbox/ab_test gate, or\n"
            "  failed SafetyPipeline registration. Generation is stochastic —\n"
            "  re-run; a clean single `async def` passes on most attempts."
        )
    return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--verbose", action="store_true", help="DEBUG logging to the log file"
    )
    args = ap.parse_args()
    rc = asyncio.run(main(args.verbose))
    sys.exit(rc)
