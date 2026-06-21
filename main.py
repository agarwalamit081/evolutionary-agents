"""Main CLI entry point for the Turing Agent.

Usage:
    python main.py --interactive
    python main.py --goal "Build a REST API"
    python main.py --provider openai --model gpt-4o-mini --goal "Explain quicksort"

The run-execution core (``execute_run``) lives in ``src/runner.py`` and is shared
with the queue worker (``src.worker.executors.default_agent_executor``); this
module is the thin CLI surface over it.
"""

from __future__ import annotations

import asyncio
import sys

import click
from loguru import logger

from src.config import get_settings
from src.observability.logging import reset_logging, setup_logging
from src.runner import _create_gateway, execute_run


@click.command()
@click.option("--goal", "-g", "goal_text", help="Goal for the agent to accomplish")
@click.option("--interactive", "-i", is_flag=True, help="Prompt for goal at runtime")
@click.option("--provider", "-p", default=None, help="LLM provider (e.g., openai, anthropic, deepseek)")
@click.option("--model", "-m", default=None, help="Specific model to use")
@click.option("--no-evolution", is_flag=True, help="Disable evolution phase")
@click.option("--max-iterations", default=None, type=int, help="Maximum graph iterations (default: from settings)")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--run-id", default=None, help="Unique run identifier for per-query logging")
@click.option("--stream", is_flag=True, help="Stream the final answer to the terminal token-by-token")
@click.option(
    "--eval",
    "run_eval",
    is_flag=True,
    help="Run the Battery-04 golden suite: each spec as a full agent run with correctness scoring",
)
@click.option(
    "--resume",
    "resume_run_id",
    default=None,
    help="Resume a prior run by its --run-id, continuing from its last checkpoint",
)
@click.option(
    "--results-dir",
    "results_dir",
    default=None,
    help="Override results_root for this run (deliverables land under <dir>/<run-id>/)",
)
@click.option(
    "--clean",
    is_flag=True,
    help="Clear this run's results subfolder before starting (requires --run-id)",
)
def main(
    goal_text: str | None,
    interactive: bool,
    provider: str | None,
    model: str | None,
    no_evolution: bool,
    max_iterations: int | None,
    verbose: bool,
    run_id: str | None,
    stream: bool,
    run_eval: bool,
    resume_run_id: str | None,
    results_dir: str | None,
    clean: bool,
) -> None:
    """Turing Agent — a self-evolving AI agent built with LangGraph."""
    # Setup logging
    reset_logging()
    settings = get_settings()
    # Phase 7: --results-dir overrides results_root for this run only.
    if results_dir:
        settings.agent.results_root = results_dir
    if verbose:
        settings.logging.log_level = "DEBUG"
    setup_logging(settings.logging)

    # --eval: run the golden Battery-04 suite end-to-end with correctness scoring.
    if run_eval:
        asyncio.run(_run_eval_suite(model))
        return

    # --resume: continue a prior run from its last checkpoint. The goal lives in
    # the checkpoint, so --goal is optional here; --run-id is set from the value
    # passed to --resume (the thread_id is derived from it for stable recall).
    resume = resume_run_id is not None
    if resume:
        run_id = resume_run_id
        if not goal_text:
            goal_text = ""  # recovered from the checkpoint inside execute_run
    else:
        # Get goal
        if interactive or not goal_text:
            goal_text = click.prompt("Enter your goal")
        if not goal_text:
            click.echo("Error: No goal provided. Use --goal or --interactive.")
            sys.exit(1)

    # Resolve the iteration cap from settings when the CLI didn't override it.
    # Done before the banner so the printed cap reflects the real value — the
    # CLI default is `None` (no --max-iterations), which previously echoed as
    # "Max iterations: None" and read as unbounded even though the fallback in
    # execute_run still bounded the run.
    if max_iterations is None:
        max_iterations = settings.agent.max_iterations

    click.echo(f"🎯 Goal: {goal_text or '(resuming prior run)'}")
    click.echo(f"   Provider: {provider or 'default'} | Model: {model or 'auto'}")
    click.echo(f"   Max iterations: {max_iterations} | Evolution: {'disabled' if no_evolution else 'enabled'}")
    if resume:
        click.echo(f"   Resume: run_id={run_id} (continuing from last checkpoint)")

    # Phase 7: --clean clears this run's results subfolder before starting so a
    # re-run starts from a clean per-run dir (requires --run-id to stay scoped).
    if clean:
        _clean_run_results(settings.agent.results_root, run_id)

    # Run the agent graph to completion.
    try:
        result = asyncio.run(
            execute_run(goal_text, max_iterations, no_evolution, run_id, model, resume=resume)
        )
    except (RuntimeError, ValueError) as exc:
        if resume:
            # Clean refusal when the checkpoint can't be found / no checkpointer.
            click.echo(f"Error: cannot resume — {exc}")
            sys.exit(1)
        raise

    click.echo("\n" + "=" * 60)
    click.echo("📋 Result:")
    if stream:
        # Stream the final answer token-by-token. A fresh gateway is created for
        # the streaming call (the run's gateway lived in a now-closed event
        # loop). _stream_final_answer never raises — it falls back to the static
        # final_output when no gateway is available or the stream fails.
        asyncio.run(
            _stream_final_answer(_create_gateway(get_settings(), model), goal_text, result)
        )
    else:
        click.echo(result.get("final_output", "No output"))
    click.echo(f"\n   Iterations: {result.get('iteration_count', 0)}")
    click.echo(f"   Completed: {result.get('is_complete', False)}")


def _clean_run_results(results_root: str, run_id: str | None) -> None:
    """Delete a run's results subfolder (Phase 7 ``--clean``).

    Removes ONLY ``<results_root>/<run_id>`` — never the whole ``results_root``
    and never anything that resolves outside it. Refuses an unsafe ``run_id``
    (path separators, ``.``/``..``) and exits cleanly when the subfolder is
    already absent.
    """
    import re as _re
    import shutil as _shutil
    from pathlib import Path as _Path

    if not run_id:
        click.echo("Error: --clean requires --run-id to target a run's subfolder")
        sys.exit(1)
    if not _re.fullmatch(r"[A-Za-z0-9_.\-]+", run_id) or run_id in {".", ".."}:
        click.echo(f"Error: --clean refused unsafe run_id: {run_id!r}")
        sys.exit(1)
    base = _Path(results_root).resolve()
    sub = (base / run_id).resolve()
    if sub == base or not sub.is_relative_to(base):
        click.echo(f"Error: --clean target escapes results_root: {sub}")
        sys.exit(1)
    if sub.exists():
        _shutil.rmtree(sub)
        click.echo(f"🧹 Cleared prior results subfolder: {sub}")
    else:
        click.echo(f"   --clean: no prior subfolder to clear ({sub})")


def _run_cost(result_dict: dict) -> float:
    """Sum cost_records (CostRecord dataclass or dict) into USD."""
    records = result_dict.get("cost_records") or []
    return float(
        sum(
            getattr(r, "cost_usd", 0) if hasattr(r, "cost_usd") else r.get("cost_usd", 0)
            for r in records
        )
    )


def _fmt_score(score: float | None) -> str:
    return "n/a" if score is None else f"{score:.2f}"


async def _run_eval_suite(model: str | None = None) -> None:
    """Run the Battery-04 golden suite end-to-end with correctness scoring.

    Each spec runs as a full agent run (the real dependency stack via
    ``execute_run``) with ``run_id`` = spec id, so the verify node runs the
    spec's correctness checks. Forces ``EVAL_ENABLED`` on for the suite and
    prints a per-spec table. Per-check rows are persisted to ``eval_results`` by
    the verify node's EvalStore (non-fatal).
    """
    from src.eval.golden import BATTERY04_GOALS

    settings = get_settings()
    # Force the correctness layer on for the suite (EVAL_ENABLED gates verify).
    settings.eval.eval_enabled = True

    click.echo("=" * 60)
    click.echo("🧪 Battery-04 evaluation suite — 4 golden specs")
    click.echo("=" * 60)

    rows: list[tuple[str, bool, float | None, float, int]] = []
    for spec in BATTERY04_GOALS:
        click.echo(f"\n▶ {spec.spec_id}: {spec.description[:70]}")
        result = await execute_run(
            spec.goal_text,
            spec.max_iterations,
            no_evolution=False,
            run_id=spec.spec_id,
            model=model,
        )
        complete = bool(result.get("is_complete", False))
        score = result.get("eval_correctness_score")
        cost = _run_cost(result)
        iters = int(result.get("iteration_count", 0))
        rows.append((spec.spec_id, complete, score, cost, iters))
        click.echo(
            f"  complete={complete} correctness_score={_fmt_score(score)} "
            f"cost=${cost:.4f} iters={iters}"
        )

    click.echo("\n" + "=" * 60)
    click.echo("📊 Suite summary")
    click.echo("=" * 60)
    click.echo(f"{'Spec':<16} {'Complete':<10} {'Score':<8} {'Cost':<10} Iters")
    for spec_id, complete, score, cost, iters in rows:
        click.echo(
            f"{spec_id:<16} {'yes' if complete else 'no':<10} "
            f"{_fmt_score(score):<8} ${cost:<9.4f} {iters}"
        )


async def _stream_final_answer(
    gateway: object | None,
    goal_text: str,
    result_dict: dict,
) -> None:
    """Stream a concise final deliverable to stdout token-by-token.

    Issues ONE synthesis call via ``gateway.astream(...)`` grounded in the run's
    ``final_output``, printing each yielded token immediately with no buffering.
    Never raises: if ``gateway`` is None or the stream raises any exception
    (including ``asyncio.CancelledError``), it falls back to printing the static
    ``final_output`` exactly as the non-stream path would.

    Args:
        gateway: The ``LLMGateway`` used for the run, or None.
        goal_text: The original goal.
        result_dict: The final agent state (reads ``final_output``,
            ``iteration_count``, ``is_complete``).
    """
    fallback = result_dict.get("final_output", "No output")

    if gateway is None:
        print(fallback, flush=True)
        return

    # Truncate the grounding context so the synthesis prompt stays compact.
    grounding = (result_dict.get("final_output", "") or "")[:2000]
    run_summary = (
        f"Iterations: {result_dict.get('iteration_count', 0)} | "
        f"Completed: {result_dict.get('is_complete', False)}"
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are the Turing Agent. Produce the final answer/deliverable "
                "for the goal below. Be complete and concise."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Goal: {goal_text}\n\n"
                f"Run summary: {run_summary}\n\n"
                f"Final output from the run (grounding context):\n{grounding}"
            ),
        },
    ]

    try:
        streamed_any = False
        astream = getattr(gateway, "astream")
        async for token in astream(messages):
            if token:
                print(token, end="", flush=True)
                streamed_any = True
        if streamed_any:
            print()  # trailing newline after the streamed answer
        else:
            print(fallback, flush=True)
    except asyncio.CancelledError as exc:
        logger.debug(f"Final-answer stream cancelled, falling back: {exc}")
        print(fallback, flush=True)
    except Exception as exc:  # noqa: BLE001 — streaming must never crash the run
        logger.debug(f"Final-answer stream failed, falling back: {exc}")
        print(fallback, flush=True)


if __name__ == "__main__":
    main()
