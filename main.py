"""Main CLI entry point for the Turing Agent.

Usage:
    python main.py --interactive
    python main.py --goal "Build a REST API"
    python main.py --provider openai --model gpt-4o-mini --goal "Explain quicksort"
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

import click
from loguru import logger

from src.config import get_settings
from src.config.settings import Settings as Settings  # noqa: TC002 — used in annotations
from src.observability.logging import reset_logging, setup_logging


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
            goal_text = ""  # recovered from the checkpoint inside _run_agent
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
    # "Max iterations: None" and read as unbounded even though the fallback at
    # _run_agent:291 still bounded the run.
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
            _run_agent(goal_text, max_iterations, no_evolution, run_id, model, resume=resume)
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


def _thread_id_for_run(run_id: str | None, goal_text: str) -> str:
    """Resolve the LangGraph thread_id for a run.

    Keyed on ``run_id`` when provided so a later ``--resume <run_id>`` can reuse
    the same thread and continue from its checkpoint across processes. Falls
    back to a pid/object-id key (process-local, non-resumable) otherwise.

    Args:
        run_id: Optional run identifier.
        goal_text: The goal (used only for the fallback key).

    Returns:
        A stable thread_id string.
    """
    if run_id is not None:
        return f"cli-{run_id}"
    return f"cli-{os.getpid()}-{id(goal_text)}"


def _new_attempt_id() -> str:
    """Generate a per-invocation eval-attempt discriminator.

    Timestamp-prefixed so lexicographic ordering == chronological ordering (two
    runs in the same second are disambiguated by the uuid suffix). This is what
    makes ``EvalStore.query_latest_attempt`` return the newest attempt's rows
    rather than a blend of every attempt under a stable ``thread_id``.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _resolve_eval_spec_id(run_id: str | None) -> str | None:
    """Resolve a run_id to its golden GoalSpec id (short or long form).

    Accepts either the spec id directly (``battery04_q01``) or the short form
    (``q01`` → ``battery04_q01``) so a battery query opts into Phase-3 eval
    scoring via ``--run-id q01`` while per-run results keep the clean
    ``results/q01/`` layout (and cross-query recall via ``resolve_existing``
    still works — a ``battery04_q01`` run_id would otherwise isolate q1's
    deliverables where q2/q3/q4 could not recall them).

    Returns ``None`` when no spec matches (an ordinary run_id is untouched).
    """
    if run_id is None:
        return None
    try:
        from src.eval.golden import lookup_goal_spec

        if lookup_goal_spec(run_id) is not None:
            return run_id
        candidate = f"battery04_{run_id}"
        if lookup_goal_spec(candidate) is not None:
            return candidate
    except Exception as e:
        logger.debug(f"GoalSpec lookup skipped: {e}")
    return None


async def _require_resumable_checkpoint(
    checkpointer: Any,
    thread_id: str,
    run_id: str | None,
) -> Any:
    """Validate a run can be resumed; return the existing checkpoint tuple.

    Args:
        checkpointer: The LangGraph checkpointer (must expose ``aget_tuple``).
        thread_id: The thread whose checkpoint resumes the run.
        run_id: The originating run_id (for a clear error message).

    Returns:
        The checkpoint tuple (truthy).

    Raises:
        ValueError: ``--resume`` was used without a ``--run-id``.
        RuntimeError: No checkpointer is available, or no checkpoint exists for
            the thread (the run never persisted state).
    """
    if run_id is None:
        raise ValueError("--resume requires --run-id to identify the checkpoint")
    if checkpointer is None:
        raise RuntimeError(
            "no checkpointer available (PostgreSQL unreachable); cannot resume"
        )
    existing = await checkpointer.aget_tuple({"configurable": {"thread_id": thread_id}})
    if existing is None:
        raise RuntimeError(
            f"no checkpoint found for run_id={run_id} (thread_id={thread_id}); "
            "the run may never have persisted state"
        )
    return existing


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


async def _run_agent(
    goal_text: str,
    max_iterations: int | None = None,
    no_evolution: bool = False,
    run_id: str | None = None,
    model: str | None = None,
    resume: bool = False,
) -> dict:
    """Run the agent graph to completion.

    Instantiates dependencies (LLMGateway, MemoryManager, ToolRegistry)
    and passes them through compile_task_graph() for dependency injection.

    Args:
        goal_text: The goal to accomplish.
        max_iterations: Maximum graph iterations.
        no_evolution: Skip evolution phase.
        model: Optional pinned model (registry key or litellm id) that
            overrides complexity routing for every LLM call in this run.
            Threads the CLI ``--model`` flag through to ``LLMGateway``.
        resume: When True, continue a prior run (identified by ``run_id``) from
            its last checkpoint instead of starting fresh. Requires a
            checkpointer and an existing checkpoint for the thread; refuses
            with a clear error otherwise. Threads the CLI ``--resume`` flag.

    Returns:
        Final agent state as a dict.
    """
    from src.config import get_settings
    from src.graph.factory import initial_state
    from src.graph.task_graph import compile_task_graph

    settings = get_settings()
    if max_iterations is None:
        max_iterations = settings.agent.max_iterations

    # Configure LangSmith tracing from settings
    if settings.langsmith.is_configured:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith.langsmith_api_key  # type: ignore[arg-type]
        os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith.langsmith_project)
        os.environ.setdefault("LANGSMITH_ENDPOINT", settings.langsmith.langsmith_endpoint)
        logger.info(
            f"LangSmith tracing enabled — project: {settings.langsmith.langsmith_project}"
        )
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

    # Per-query logging — creates a dedicated log file for this run
    if run_id is not None:
        try:
            from src.observability.logging import add_query_log_sink

            add_query_log_sink(run_id, settings.logging)
        except Exception as e:
            logger.debug(f"Per-query logging setup skipped: {e}")

    # Create initial state. thread_id is keyed on run_id when given so a later
    # --resume <run_id> reuses the same thread and continues from its checkpoint.
    thread_id = _thread_id_for_run(run_id, goal_text)
    # Phase 7: bind the run_id so file_writer/execute route this run's
    # deliverables under results_root/<run_id>/ and reads fall back to flat.
    # None (no --run-id) leaves resolution flat — legacy, non-regressing.
    try:
        from src.tools._paths import set_active_run_id

        set_active_run_id(run_id)
    except Exception as e:
        logger.debug(f"Per-run results subfoldering not bound: {e}")
    state = initial_state(goal_text, thread_id, max_iterations, no_evolution=no_evolution)

    # Per-run-attempt discriminator for the eval store. thread_id is STABLE
    # across re-runs of the same --run-id (it is the resume key), so eval_results
    # would otherwise blend every attempt of a run under one run_id. Generate a
    # fresh, timestamp-ordered attempt_id per invocation so a score means ONE
    # attempt; EvalStore.query_latest_attempt returns only the newest.
    state["eval_attempt_id"] = _new_attempt_id()

    # Phase 3: thread the run's golden GoalSpec id so the verify node runs the
    # spec's correctness checks (when EVAL_ENABLED). See _resolve_eval_spec_id
    # for the short-form (``--run-id q01``) resolution that keeps results under
    # ``results/q01/`` while still opting into scoring.
    spec_id = _resolve_eval_spec_id(run_id)
    if spec_id is not None:
        state["eval_goal_spec_id"] = spec_id
        # A battery run (``--run-id qNN`` resolves to a golden GoalSpec) opts
        # into the Phase-3 correctness layer: enable eval + enforce so the
        # verify node's golden/structural/execution checks actually run and a
        # present-but-wrong deliverable is caught and retried (within the
        # iteration budget) instead of being force-completed as "done". This is
        # the correctness safety net the F-h.3 goal-satisfied force-complete
        # relies on — without it, a shallow/fabricated deliverable slips through
        # (battery-04 q4: a {"status":"failed"} test_results.json stub).
        # eval_enforce is bounded by _run_correctness_checks: it only downgrades
        # complete→incomplete while iterations remain, so a strict check can
        # never loop a run past its iteration hard-cap.
        if not settings.eval.eval_enabled:
            settings.eval.eval_enabled = True
            logger.info(f"Battery spec {spec_id!r} resolved — EVAL enabled for this run")
        if not settings.eval.eval_enforce:
            settings.eval.eval_enforce = True
            logger.info(
                f"Battery spec {spec_id!r} resolved — EVAL_ENFORCE on "
                "(failing checks downgrade complete→incomplete, retried within budget)"
            )

    # Instantiate dependencies
    gateway = _create_gateway(settings, pinned_model=model)

    # Inject Redis prompt cache for LLM response caching
    if gateway is not None:
        # Bind the run's thread_id as the cost-ledger attribution key so every
        # LLM call's cost row is attributable to this run. thread_id is always
        # defined above (cli-{run_id} or cli-{pid}-{obj}); None only when the
        # gateway itself is None (heuristic fallback — no costs to attribute).
        gateway.set_run_id(thread_id)
        try:
            import redis.asyncio as _aioredis

            from src.llm.cache import PromptCache

            _redis_client = _aioredis.from_url(settings.redis.redis_url)
            cache = PromptCache(_redis_client, settings)
            gateway.set_cache(cache)
            logger.info("Redis prompt cache injected into gateway")
        except Exception as e:
            logger.debug(f"Prompt cache not available: {e}")

    memory = await _async_create_memory_manager(settings)
    tools = _create_tool_registry()

    # Load previously created dynamic tools from database. Passing settings
    # enables the B3 governance passes (semantic dedup + cumulative max_active_tools cap).
    if tools is not None:
        await _load_dynamic_tools(tools, settings)

    # Load active sub-agents from database. Passing settings enables the B3
    # governance passes (retire_redundant + enforce_caps) — without it the load
    # is ungoverned and the active population bloats past the cap.
    from src.agents.registry import SubAgentRegistry

    sub_agent_registry = SubAgentRegistry()
    await _load_sub_agents(sub_agent_registry, settings)

    # Create checkpointer if possible
    checkpointer = await _create_checkpointer(settings)

    # --resume: validate the run can be resumed and recover the goal for
    # logging/results. Refuses (RuntimeError/ValueError) when there is no
    # checkpointer or no checkpoint for the thread — surfaced cleanly by the CLI.
    if resume:
        cp_tuple = await _require_resumable_checkpoint(checkpointer, thread_id, run_id)
        try:
            channel_values = cp_tuple.checkpoint.get("channel_values", {})  # type: ignore[union-attr]
            cp_goal = channel_values.get("current_goal")
            if (
                not goal_text
                and cp_goal is not None
                and hasattr(cp_goal, "text")
                and cp_goal.text
            ):
                goal_text = cp_goal.text
                logger.info(f"Resuming run {run_id} (recovered goal: {goal_text[:60]})")
        except Exception as e:
            logger.debug(f"Could not recover goal from checkpoint: {e}")
    elif checkpointer is not None:
        # Fresh run reusing a run_id-derived thread_id: clear any prior
        # checkpoint for this thread so the run starts clean instead of
        # contaminating (or immediately terminating on) a prior run's state.
        # No-op for a first run with no checkpoint. Best-effort.
        try:
            await checkpointer.adelete_thread(thread_id)
        except Exception as e:
            logger.debug(f"Could not clear prior checkpoint for {thread_id}: {e}")

    # Compile graph with injected dependencies
    compiled = compile_task_graph(
        gateway=gateway,
        memory=memory,
        tools=tools,
        checkpointer=checkpointer,
        sub_agent_registry=sub_agent_registry,
    )

    # Run-scoped DB session for durable cost tracking. Opened before the graph
    # runs and closed in ``finally`` so cost_ledger rows land even on error
    # paths. Separate from the memory manager's session (which escapes its own
    # async-with at _async_create_memory_manager and is not safely closeable
    # here). Best-effort: if the DB is unavailable the gateway runs without a
    # tracker — the gateway already tolerates _cost_tracker=None.
    cost_session_cm = None
    if gateway is not None:
        try:
            from src.db.session import get_session
            from src.llm.cost_tracker import CostTracker

            cost_session_cm = get_session()
            cost_session = await cost_session_cm.__aenter__()
            gateway.set_cost_tracker(CostTracker(cost_session, settings))
            logger.info("Cost tracker wired (budget gate active)")
        except Exception as e:
            logger.debug(f"Cost tracker not available: {e}")
            cost_session_cm = None

    logger.info(f"Starting agent with goal: {goal_text[:80]}")
    recursion_limit = max(max_iterations * 8, 100)
    # Key checkpoints by thread_id when a checkpointer is wired so a later
    # --resume can continue the run. On resume, pass None as the input so
    # LangGraph continues from the last checkpoint rather than restarting.
    invoke_config: dict[str, Any] = {"recursion_limit": recursion_limit}
    if checkpointer is not None:
        invoke_config["configurable"] = {"thread_id": thread_id}
    try:
        result = await compiled.ainvoke(
            None if resume else state,
            config=invoke_config,
        )
        result_dict = dict(result)

        # Cost/token fallback: if the graph terminated without flushing
        # cost_records via store_memory (e.g. error_termination), recover from the
        # gateway's in-memory accumulator so run-history reflects real spend.
        if gateway is not None:
            if not result_dict.get("cost_records"):
                gateway_records = gateway.get_cost_records()
                if gateway_records:
                    result_dict["cost_records"] = gateway_records
            if not result_dict.get("total_tokens_used"):
                records = result_dict.get("cost_records") or []
                if records:
                    result_dict["total_tokens_used"] = sum(
                        getattr(r, "input_tokens", 0) + getattr(r, "output_tokens", 0)
                        for r in records
                    )

        logger.info("Agent execution complete")
    finally:
        # Close the run-scoped cost-tracker session; never raise out of finally.
        if cost_session_cm is not None:
            try:
                await cost_session_cm.__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"Cost session close skipped: {e}")

    # Save results file to results/ directory
    try:
        import re as _re
        from datetime import datetime as _dt, timezone as _tz

        goal_text_safe = _re.sub(r'[^a-zA-Z0-9_-]', '_', goal_text[:50]).strip('_')
        timestamp = _dt.now(_tz.utc).strftime('%Y%m%d_%H%M%S')
        # Phase 7: route the run summary through the shared resolver so it lands
        # under results_root/<run_id>/ when per-run subfoldering is active.
        from src.tools._paths import normalize as _normalize

        results_file = _normalize(f"{goal_text_safe}_{timestamp}.md", base="results")
        results_file.parent.mkdir(parents=True, exist_ok=True)

        final_output = result_dict.get("final_output", "")
        is_complete = result_dict.get("is_complete", False)
        iteration_count = result_dict.get("iteration_count", 0)

        content = (
            f"# Agent Results\n\n"
            f"**Goal**: {goal_text}\n"
            f"**Timestamp**: {timestamp}\n"
            f"**Complete**: {'Yes' if is_complete else 'No'}\n"
            f"**Iterations**: {iteration_count}\n\n"
            f"## Output\n\n{final_output}\n"
        )
        results_file.write_text(content, encoding="utf-8")
        logger.info(f"Results file saved to: {results_file}")
    except Exception as e:
        logger.debug(f"Results file save skipped: {e}")

    # Generate run history (non-blocking, best-effort)
    try:
        from src.graph.run_history import RunHistoryGenerator

        history_gen = RunHistoryGenerator(
            workspace_root=settings.agent.workspace_root
        )
        history_path = await history_gen.generate(result_dict)
        logger.info(f"Run history written to: {history_path}")
    except Exception as e:
        logger.debug(f"Run history generation skipped: {e}")

    return result_dict


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
    ``_run_agent``) with ``run_id`` = spec id, so the verify node runs the
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
        result = await _run_agent(
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


def _create_gateway(settings: Settings, pinned_model: str | None = None):
    """Create LLMGateway if provider key is available.

    Args:
        settings: Application settings.
        pinned_model: Optional model (registry key or litellm id) that
            overrides complexity routing for every call in this run.
    """
    try:
        from src.llm.gateway import LLMGateway

        return LLMGateway(settings, pinned_model=pinned_model)
    except Exception:
        logger.debug("LLMGateway not available, using heuristic fallback")
        return None


def _create_memory_manager(settings: Settings):  # noqa: ARG001 — public API for sync callers
    """Create MemoryManager if Redis and PostgreSQL are available.

    Note: Returns a coroutine that must be awaited.
    """
    return _async_create_memory_manager(settings)


async def _async_create_memory_manager(settings: Settings):
    """Async factory for MemoryManager."""
    try:
        import redis.asyncio as aioredis

        from src.db.session import _get_session_factory
        from src.memory.manager import MemoryManager

        redis_client = aioredis.from_url(settings.redis.redis_url)
        factory = _get_session_factory()
        async with factory() as db_session:
            return MemoryManager(
                redis_client=redis_client,
                db_session=db_session,
                settings=settings,
            )
    except Exception:
        logger.debug("MemoryManager not available, using stub memory")
        return None


def _create_tool_registry():
    """Create ToolRegistry with all built-in tools."""
    try:
        from src.tools import create_default_registry

        return create_default_registry()
    except Exception:
        logger.debug("ToolRegistry not available")
        return None


async def _load_dynamic_tools(tools: object, settings: Settings) -> None:
    """Load previously created dynamic tools from the database.

    Best-effort — non-fatal if the database is unavailable. Passing
    ``settings.agent`` runs the B3 governance passes (semantic dedup, cumulative
    ``max_active_tools`` cap, redundancy retirement) around the load.
    """
    try:
        from src.tools.dynamic.persister import ToolPersister
        from src.tools.registry import ToolRegistry

        if not isinstance(tools, ToolRegistry):
            return

        persister = ToolPersister()
        loaded = await persister.load_active_tools(tools, settings=settings.agent)
        if loaded:
            logger.info(f"Loaded {len(loaded)} dynamic tools from DB: {', '.join(loaded)}")
    except Exception as e:
        logger.debug(f"Dynamic tool loading skipped: {e}")


async def _load_sub_agents(registry: object, settings: Settings) -> None:
    """Load previously created sub-agents from the database.

    Best-effort — non-fatal if the database is unavailable. Passing
    ``settings.agent`` runs the B3 governance passes (``retire_redundant`` +
    ``enforce_caps``) so the active population stays within the cumulative cap.
    """
    try:
        from src.agents.persister import SubAgentPersister
        from src.agents.registry import SubAgentRegistry

        if not isinstance(registry, SubAgentRegistry):
            return

        persister = SubAgentPersister()
        loaded = await persister.load_active_agents(registry, settings=settings.agent)
        if loaded:
            logger.info(f"Loaded {len(loaded)} sub-agents from DB: {', '.join(loaded)}")
    except Exception as e:
        logger.debug(f"Sub-agent loading skipped: {e}")


async def _create_checkpointer(settings: Settings):
    """Create AsyncPostgresSaver checkpointer if PostgreSQL is available."""
    try:
        from src.graph.checkpoint import create_checkpointer

        return await create_checkpointer(settings.database.database_url)
    except Exception:
        logger.debug("Checkpointer not available, running without persistence")
        return None


if __name__ == "__main__":
    main()
