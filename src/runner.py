"""Canonical run executor — the shared core behind the CLI and the worker.

``execute_run`` runs the agent graph to completion: it instantiates the run's
dependencies (LLMGateway, MemoryManager, ToolRegistry, SubAgentRegistry,
AsyncPostgresSaver checkpointer, cost tracker), wires LangSmith tracing, binds
the per-run results path + thread_id, and drives ``compile_task_graph().ainvoke``
to completion. It is the single source of truth for "run one goal" — the CLI
(``main.py``) and the queue worker (``src.worker.executors.default_agent_executor``)
both call it, so a host CLI run and an API-routed run follow the EXACT same
dependency-wired, checkpointed, cost-tracked path (no duplicated run logic).

Origin. This module was extracted from ``main.py`` in Phase 3 (P3a) so the worker
no longer imports the CLI entrypoint (``main`` pulls in the full Click surface).
Pre-extraction the worker did ``from main import _run_agent``; that coupling is
gone — the worker imports ``execute_run`` from here, a plain library module with
no CLI dependencies. The bodies are carried over verbatim; only the name changed
(``_run_agent`` → ``execute_run``).

Import discipline. The graph/gateway/memory/DB imports are deliberately deferred
inside the functions (as in ``main.py``) so importing this module is cheap and
side-effect-free — the worker and tests can import it without pulling in the
full agent stack.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from src.config.settings import Settings  # noqa: TC003 — used in annotations

# Optional per-run progress reporter. The queue worker binds one (via the
# executor) so ``execute_run`` can stream the live ``iteration_count`` into the
# run-status hash mid-run — without it, ``GET /runs/<id>`` reports
# iteration_count=0 for the whole run until completion (#255). ``None`` (the CLI
# path, which has no status store) keeps ``execute_run`` on the atomic
# ``ainvoke`` path with zero behavior change.
RunProgressCallback = Callable[[int], Awaitable[None]]


class RunCancelled(Exception):
    """Raised by ``execute_run`` when a graceful cancel was requested — a Redis
    flag set via ``POST /runs/{run_id}/cancel`` (E), checked at the per-iteration
    progress callback. Caught by the worker as the terminal CANCELLED status with
    ~1-iteration latency: the in-flight iteration completes first, then the flag
    is observed and the run stops cleanly (no container restart needed, as the
    q09 halt required). Defined here (the run engine module) so both execute_run
    (the raiser) and the worker (the catcher) import it one-way without a cycle.
    """


def _thread_id_for_run(run_id: str | None, goal_text: str, *, origin: str = "cli") -> str:
    """Resolve the LangGraph thread_id for a run.

    Keyed on ``run_id`` when provided so a later ``--resume <run_id>`` can reuse
    the same thread and continue from its checkpoint across processes. Falls
    back to a pid/object-id key (process-local, non-resumable) otherwise.

    Args:
        run_id: Optional run identifier.
        goal_text: The goal (used only for the fallback key).
        origin: Origin prefix (``"cli"`` for the CLI, ``"api"`` for queue-routed
            runs) so an API run and a CLI run sharing a run_id do NOT collide on
            the same checkpoint thread. The worker executor passes ``"api"``.

    Returns:
        A stable thread_id string.
    """
    if run_id is not None:
        return f"{origin}-{run_id}"
    return f"{origin}-{os.getpid()}-{id(goal_text)}"


def _new_attempt_id() -> str:
    """Generate a per-invocation eval-attempt discriminator.

    Timestamp-prefixed so lexicographic ordering == chronological ordering (two
    runs in the same second are disambiguated by the uuid suffix). This is what
    makes ``EvalStore.query_latest_attempt`` return the newest attempt's rows
    rather than a blend of every attempt under a stable ``thread_id``.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _strip_date_suffix(run_id: str) -> str | None:
    """Strip a trailing ``-YYYYMMDD`` (exactly 8 digits) date suffix, if present.

    The nightly capability-curve scheduler (``src.scheduler``) enqueues each
    battery spec under a date-suffixed ``run_id`` (``battery04_q01-20260622``)
    for per-night results isolation. The resolver must still map that back to the
    spec id (``battery04_q01``) so the verify node scores the run. Returns the
    stripped id, or ``None`` when no 8-digit suffix follows a hyphen. String-only
    (no ``re`` import) — ``rpartition`` splits on the LAST hyphen so a spec id
    containing no hyphen (``battery04_q01``) returns ``None`` untouched.

    Also strips an optional generation tag (``-gen0``/``-gen1``/``-gen2``) that
    precedes the date suffix: the multi-generation self-improvement curve enqueues
    each generation under ``{spec_id}-gen{N}-YYYYMMDD`` so a G0/G1/G2 run share a
    date yet stay isolated. The tag is dropped too so the resolver still maps
    ``q01-gen0-20260703`` back to ``battery04_q01`` (otherwise eval scoring is
    silently skipped and the generation curve has no score signal).
    """
    if "-" not in run_id:
        return None
    base, _, tail = run_id.rpartition("-")
    if len(tail) != 8 or not tail.isdigit():
        return None
    # Drop an optional ``-gen<N>`` generation tag preceding the date suffix.
    if "-" in base:
        gbase, _, gtail = base.rpartition("-")
        if gtail.startswith("gen") and gtail[3:].isdigit():
            base = gbase
    return base if base else None


def _resolve_eval_spec_id(run_id: str | None) -> str | None:
    """Resolve a run_id to its golden GoalSpec id (short, long, or date-suffixed).

    Accepts either the spec id directly (``battery04_q01``) or the short form
    (``q01`` → ``battery04_q01``) so a battery query opts into Phase-3 eval
    scoring via ``--run-id q01`` while per-run results keep the clean
    ``results/q01/`` layout (and cross-query recall via ``resolve_existing``
    still works — a ``battery04_q01`` run_id would otherwise isolate q1's
    deliverables where q2/q3/q4 could not recall them).

    Also strips a trailing ``-YYYYMMDD`` date suffix the nightly scheduler
    appends (``battery04_q01-20260622`` → ``battery04_q01``) so scheduled battery
    runs are scored against their spec while keeping per-night results isolation.

    Returns ``None`` when no spec matches (an ordinary run_id is untouched).
    """
    if run_id is None:
        return None
    try:
        from src.eval.golden import lookup_goal_spec

        def _match(candidate: str) -> str | None:
            if lookup_goal_spec(candidate) is not None:
                return candidate
            prefixed = f"battery04_{candidate}"
            if lookup_goal_spec(prefixed) is not None:
                return prefixed
            return None

        direct = _match(run_id)
        if direct is not None:
            return direct
        stripped = _strip_date_suffix(run_id)
        if stripped is not None:
            return _match(stripped)
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


async def execute_run(
    goal_text: str,
    max_iterations: int | None = None,
    no_evolution: bool = False,
    run_id: str | None = None,
    model: str | None = None,
    resume: bool = False,
    origin: str = "cli",
    on_progress: RunProgressCallback | None = None,
    results_per_run_subdir: bool | None = None,
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
        origin: Checkpoint thread_id prefix (``"cli"`` default). The queue worker
            passes ``"api"`` so an API-routed run never collides with a CLI run
            sharing the same run_id on the checkpointer.
        on_progress: Optional async callback invoked with the live
            ``iteration_count`` whenever it changes during the run. When
            provided, the graph is driven via ``astream`` (stream_mode="values")
            so each super-step's iteration_count is reported mid-run; the worker
            uses this to mirror progress into the run-status hash (#255). When
            ``None`` (the CLI path), the run uses the atomic ``ainvoke`` path
            unchanged. The callback is observability-only — a raise inside it is
            caught and logged, never aborting the run.
        results_per_run_subdir: Optional per-run override of
            ``AgentSettings.results_per_run_subdir``. ``None`` (default) keeps the
            global setting. The scheduled battery passes ``False`` so its
            cross-dependent goals share the flat results root their hardcoded
            paths expect (see RunJob.results_per_run_subdir). The prior value is
            restored in ``finally`` so a long-lived worker process never leaks the
            override into the next (ad-hoc) run.

    Returns:
        Final agent state as a dict.
    """
    from src.config import get_settings
    from src.graph.factory import initial_state
    from src.graph.task_graph import compile_task_graph

    settings = get_settings()
    # max_iterations stays as the caller provided (possibly None). When None
    # (the CLI/worker default) state["max_iterations"] is left unset so the
    # routers derive the cap from the classified goal complexity via
    # effective_max_iterations (B1). An explicit pin (CLI --max-iterations, an
    # eval spec, or a worker job) is written into state and always wins. The
    # recursion-limit basis below uses the flat AgentSettings.max_iterations
    # regardless — it is computed before classify, so it cannot be complexity-
    # aware (see src/graph/iteration_cap.py).

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

    # Per-query logging — creates a dedicated log file for this run. The
    # handler id is captured so the sink is torn down before returning: a
    # long-lived worker/battery process must not bleed this run's sink into the
    # next run's log file (the #313 "goal drift" was two separate runs merged
    # into one log by exactly this leak).
    query_sink_id: int | None = None
    if run_id is not None:
        try:
            from src.observability.logging import add_query_log_sink

            query_sink_id = add_query_log_sink(run_id, settings.logging)
        except Exception as e:
            logger.debug(f"Per-query logging setup skipped: {e}")

    # Create initial state. thread_id is keyed on run_id when given so a later
    # --resume <run_id> reuses the same thread and continues from its checkpoint.
    thread_id = _thread_id_for_run(run_id, goal_text, origin=origin)
    # Optional per-run override of results isolation. The scheduled battery opts
    # OUT (flat shared root) so its cross-dependent goals resolve each other's
    # hardcoded paths. Captured for restore in finally — a long-lived worker
    # process must never leak the override into the next (ad-hoc) run. Applied
    # before set_active_run_id so _subdir_active() reads the overridden value
    # throughout the run.
    prior_subdir = settings.agent.results_per_run_subdir
    if results_per_run_subdir is not None and results_per_run_subdir != prior_subdir:
        settings.agent.results_per_run_subdir = results_per_run_subdir
        logger.info(
            f"Results isolation override for run {run_id!r}: "
            f"{prior_subdir} → {results_per_run_subdir}"
        )
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
            # Share the Redis client with the rate limiter so the worker fleet
            # (multiple gateway instances across processes) coordinates against
            # ONE provider RPM/TPM budget instead of each process slamming its
            # own 60-RPM bucket. Best-effort + opt-in (degrades to in-memory).
            gateway.set_rate_limiter_redis(_redis_client)
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
    else:
        # Fresh run: clear any prior checkpoint for this thread so it starts
        # clean instead of contaminating (or terminating on) a prior run's
        # state — no-op for a first run. Best-effort.
        if checkpointer is not None:
            try:
                await checkpointer.adelete_thread(thread_id)
            except Exception as e:
                logger.debug(f"Could not clear prior checkpoint for {thread_id}: {e}")
        # Also clear THIS run's results subdir when per-run subfoldering is on,
        # so a re-enqueued run_id does not inherit a prior attempt's deliverables
        # (disk-contamination fix). Resume MUST NOT reach here — the subdir
        # deliverables are part of the resumable run state. Best-effort; a clean
        # failure never aborts the run (same posture as the checkpoint clear).
        if run_id:
            try:
                from src.tools._paths import _subdir_active, clean_run_subdir

                if _subdir_active():
                    if clean_run_subdir(run_id):
                        logger.info(f"Cleared prior results subdir for run {run_id}")
            except Exception as e:
                logger.debug(f"Could not clear results subdir for {run_id}: {e}")

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
            tracker = CostTracker(cost_session, settings)
            gateway.set_cost_tracker(tracker)
            logger.info("Cost tracker wired (budget gate active)")
            # Baseline the per-run caps at this attempt's start: each cap
            # measures THIS attempt's spend (cumulative - baseline), so a
            # re-enqueued or resumed run_id does NOT inherit its prior debt and
            # trip the cap before doing any work (battery-04 q09 re-enqueue
            # inherited 407K tokens -> instant trip). Both the TOKEN cap and the
            # USD COST cap are baselined together. On any failure the baselines
            # stay 0 -> today's cumulative behavior (safe; never over-grants
            # budget).
            try:
                baseline = await tracker.get_run_token_usage(thread_id)
                tracker.set_run_baseline(baseline)
                baseline_cost = await tracker.get_run_spend(thread_id)
                tracker.set_run_cost_baseline(baseline_cost)
                logger.info(
                    f"Run baselines: {baseline} tokens, ${baseline_cost:.4f} "
                    f"(attempt budget measured from here)"
                )
            except Exception as e:  # noqa: BLE001 — baseline is best-effort
                logger.debug(f"Could not capture run baselines: {e}")
        except Exception as e:
            logger.debug(f"Cost tracker not available: {e}")
            cost_session_cm = None

    logger.info(f"Starting agent with goal: {goal_text[:80]}")
    # Build-time fan-out ceiling. max_iterations may be None (derive-at-runtime);
    # the recursion limit always uses the flat basis so a COMPLEX run never hits
    # GraphRecursionError before its complexity cap (B1).
    recursion_limit = max((max_iterations or settings.agent.max_iterations) * 8, 100)
    # Key checkpoints by thread_id when a checkpointer is wired so a later
    # --resume can continue the run. On resume, pass None as the input so
    # LangGraph continues from the last checkpoint rather than restarting.
    invoke_config: dict[str, Any] = {"recursion_limit": recursion_limit}
    if checkpointer is not None:
        invoke_config["configurable"] = {"thread_id": thread_id}
    try:
        if on_progress is None:
            # CLI path (no status store): atomic invoke — zero behavior change.
            result = await compiled.ainvoke(
                None if resume else state,
                config=invoke_config,
            )
            result_dict = dict(result)
        else:
            # Worker path (#255): stream each super-step so the live
            # iteration_count reaches the run-status hash mid-run via on_progress.
            # The last yielded state == the final state ainvoke would return, so
            # result_dict is equivalent. Report only on change to avoid Redis churn.
            result_dict: dict[str, Any] = {}
            last_reported = -1
            async for chunk in compiled.astream(
                None if resume else state,
                config=invoke_config,
                stream_mode="values",
            ):
                if isinstance(chunk, dict):
                    result_dict = chunk
                    ic = int(chunk.get("iteration_count", 0) or 0)
                    if ic != last_reported:
                        last_reported = ic
                        try:
                            await on_progress(ic)
                        except RunCancelled:
                            # E (cancel): the worker's progress callback raised
                            # ``RunCancelled`` after observing the Redis cancel
                            # flag. Propagate it (do NOT swallow it as
                            # "observability-only") so it exits the astream loop,
                            # runs the run-scoped ``finally`` cleanup, and reaches
                            # the worker ``_process`` RunCancelled handler →
                            # terminal CANCELLED + acked. MUST precede the generic
                            # ``except Exception`` below, which would otherwise
                            # log-and-drop it (cancel silently failing).
                            raise
                        except Exception as e:  # noqa: BLE001 — progress is observability-only
                            logger.debug(f"run progress callback error (ignored): {e}")

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
        # Restore the results-isolation override (applied above) so a long-lived
        # worker process never leaks a battery's flat-root opt-out into the next
        # (ad-hoc) run. Never raises out of finally.
        if (
            results_per_run_subdir is not None
            and settings.agent.results_per_run_subdir != prior_subdir
        ):
            try:
                settings.agent.results_per_run_subdir = prior_subdir
            except Exception as e:  # noqa: BLE001 — best-effort restore
                logger.debug(f"Results isolation restore skipped: {e}")
        # Close the run-scoped cost-tracker session; never raise out of finally.
        if cost_session_cm is not None:
            try:
                await cost_session_cm.__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"Cost session close skipped: {e}")
        # Release the checkpointer's dedicated connection so a long-lived
        # worker process doesn't leak one Postgres connection per run. The
        # connection is bound to saver.conn (opened directly in create_checkpointer
        # — NOT via a detached from_conn_string CM, which closes it on GC). No-op
        # when there is no checkpointer. Never raises out of finally.
        if checkpointer is not None:
            try:
                from src.graph.checkpoint import close_checkpointer

                await close_checkpointer(checkpointer)
            except Exception as e:
                logger.debug(f"Checkpointer close skipped: {e}")
        # Tear down this run's per-query log sink so a long-lived worker/battery
        # process does not bleed it into the next run's log file (#313). Placed
        # in ``finally`` so the sink is removed even when ``ainvoke`` raises (the
        # realistic exception path) — without this the worker's next job would
        # dump its logs into THIS run's ``logs/<run_id>.log``. None-safe; never
        # raises out of finally.
        if query_sink_id is not None:
            try:
                from src.observability.logging import remove_query_log_sink

                remove_query_log_sink(query_sink_id)
            except Exception as e:
                logger.debug(f"Per-query sink teardown skipped: {e}")

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
    except Exception as e:
        # Visibility: a bare swallow here previously hid a real regression
        # (from_conn_string API misuse) for the entire overhaul — every CLI run
        # silently ran without persistence and --resume was a no-op. Surface the
        # cause at WARNING so future checkpoint failures are diagnosable.
        logger.warning(
            f"Checkpointer not available, running without persistence: "
            f"{type(e).__name__}: {e}"
        )
        return None
