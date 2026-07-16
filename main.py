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
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

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
@click.option(
    "--capability-curve",
    "run_curve",
    is_flag=True,
    help="Print the nightly battery capability curve + regression verdict (read-only; no LLM, no DB writes)",
)
@click.option("--since", "curve_since", default=None, help="With --capability-curve: window start (ISO date/datetime, e.g. 2026-06-01)")
@click.option("--until", "curve_until", default=None, help="With --capability-curve: window end (ISO date/datetime; date-only is end-of-day inclusive)")
@click.option("--export", "curve_export", default=None, help="With --capability-curve: write JSON/CSV (.json/.csv by suffix) and exit")
@click.option("--plot", "curve_plot", default=None, help="With --capability-curve: write a PNG of the battery trend (matplotlib optional)")
@click.option(
    "--curve-model",
    "curve_model",
    default=None,
    help="With --capability-curve: slice the trend + verdict to one producer model (e.g. glm-4.7); "
    "default is the blended system-wide trend",
)
@click.option(
    "--aflow",
    "run_aflow",
    is_flag=True,
    help="Offline AFlow workflow-topology optimizer — search per-category technique policies (G3b, opt-in)",
)
@click.option(
    "--retrieval-eval",
    "run_retrieval_eval",
    is_flag=True,
    help="Measure memory recall quality (precision@k + MRR) over a seeded capability fixture (E1)",
)
@click.option(
    "--retrieval-k",
    "retrieval_k",
    default=None,
    type=int,
    help="With --retrieval-eval: top-k cutoff for precision@k (default 3)",
)
@click.option(
    "--backfill-embeddings",
    "backfill_embeddings",
    type=click.Choice(["capability", "cold", "all"]),
    default=None,
    help="Backfill NULL embedding/capability vectors (capability=tools+sub-agents api-only, "
    "cold=cold memories all-vectors) and exit. Idempotent (WHERE col IS NULL).",
)
@click.option(
    "--backfill-concurrency",
    "backfill_concurrency",
    default=5,
    type=int,
    help="With --backfill-embeddings: max concurrent embedding API calls (default 5).",
)
@click.option(
    "--backfill-dry-run",
    "backfill_dry_run",
    is_flag=True,
    help="With --backfill-embeddings: report the NULL-row count WITHOUT calling the "
    "embedding API (no spend); nothing is persisted.",
)
@click.option(
    "--score-spec",
    "score_spec_id",
    default=None,
    help="Score on-disk deliverables against a golden GoalSpec (the recomputation "
    "ground-truth engine used by --eval), then exit. Requires --deliverable path(s).",
)
@click.option(
    "--deliverable",
    "score_deliverables",
    multiple=True,
    help="Deliverable file path to score with --score-spec (repeatable: "
    "--deliverable p1 --deliverable p2). Spec-expected deliverables are added "
    "automatically.",
)
@click.option(
    "--verify-models",
    "verify_models",
    is_flag=True,
    help="Smoke-test that each named model resolves and responds over the gateway's "
    "real routing (registry _build_kwargs), then exit. Exit 0 if all healthy, 1 if any "
    "fail. Use --verify-model to name models (default set if none).",
)
@click.option(
    "--verify-model",
    "verify_model_names",
    multiple=True,
    help="Model name to smoke with --verify-models (repeatable; default "
    "qwen3.5-flash + qwen3.7-plus if none named).",
)
@click.option(
    "--cost",
    "run_cost",
    is_flag=True,
    help="Print a spend breakdown from the cost_ledger and exit (CostTracker is "
    "write-only; this is the aggregate READ path). No secrets — only call counts, "
    "spend, and tokens. Filters: --cost-run-id/--cost-model/--cost-since/--cost-today; "
    "--cost-by-model collapses across runs.",
)
@click.option("--cost-run-id", "cost_run_id", default=None, help="With --cost: filter to a single run_id.")
@click.option("--cost-model", "cost_model", default=None, help="With --cost: filter to a single model id.")
@click.option(
    "--cost-since",
    "cost_since",
    default=None,
    help="With --cost: only rows on/after DATE (YYYY-MM-DD, UTC).",
)
@click.option("--cost-today", "cost_today", is_flag=True, help="With --cost: only rows from today (UTC).")
@click.option(
    "--cost-by-model",
    "cost_by_model",
    is_flag=True,
    help="With --cost: collapse the breakdown across runs (one row per model).",
)
@click.option(
    "--inspect-mutation",
    "inspect_mutation",
    is_flag=True,
    help="Print a stored evolution mutation row (id/type/path/status/shape) and exit. "
    "Use with --mutation-id to target one (full UUID or short prefix); the latest "
    "row is shown when no id is given. Shape is classified by promote.classify_payload "
    "(the JSON-vs-free-text check the promotion gate uses). No secrets are printed.",
)
@click.option(
    "--mutation-id",
    "mutation_id",
    default=None,
    help="With --inspect-mutation: a mutation id (full UUID or a short prefix).",
)
@click.option(
    "--inspect-full",
    "inspect_full",
    is_flag=True,
    help="With --inspect-mutation: print the FULL mutated_content (default truncates).",
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
    run_curve: bool,
    curve_since: str | None,
    curve_until: str | None,
    curve_export: str | None,
    curve_plot: str | None,
    curve_model: str | None,
    run_retrieval_eval: bool,
    retrieval_k: int | None,
    run_aflow: bool,
    backfill_embeddings: str | None,
    backfill_concurrency: int,
    backfill_dry_run: bool,
    score_spec_id: str | None,
    score_deliverables: tuple[str, ...],
    verify_models: bool,
    verify_model_names: tuple[str, ...],
    run_cost: bool,
    cost_run_id: str | None,
    cost_model: str | None,
    cost_since: str | None,
    cost_today: bool,
    cost_by_model: bool,
    inspect_mutation: bool,
    mutation_id: str | None,
    inspect_full: bool,
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

    # --backfill-embeddings: idempotently populate NULL embedding/capability
    # vectors (the data half of already-shipped schema migrations). Read+write
    # maintenance op — no agent run. Exit code: 1 if rows needed backfill but
    # none were stored (all failed/skipped), else 0.
    if backfill_embeddings:
        sys.exit(
            asyncio.run(
                _run_backfill_embeddings(
                    table=backfill_embeddings,
                    concurrency=backfill_concurrency,
                    dry_run=backfill_dry_run,
                )
            )
        )

    # --score-spec: score on-disk deliverables against a golden GoalSpec via the
    # same recomputation engine --eval uses (anti-fabrication ground truth).
    # Read-only host CLI op — no agent run, no API key needed. Exit code:
    # 0 if the spec passed, 1 if it failed, 2 if the spec id is unknown.
    if score_spec_id:
        sys.exit(
            asyncio.run(
                _run_score_spec(score_spec_id, list(score_deliverables))
            )
        )

    # --verify-models: smoke each named model over the gateway's real routing
    # (the registry's _build_kwargs — provider-agnostic), then exit. Confirms a
    # model_id actually resolves + responds before trusting it in a fallback
    # chain (a 404 burns a retry slot on every failed run). Live provider key
    # required; prints NO secrets — only pass/fail, provider/model, token counts,
    # cost, and a sanitized error category. Exit 0=all healthy, 1=any failed.
    if verify_models:
        sys.exit(asyncio.run(_run_verify_models(list(verify_model_names))))

    # --cost: read-only spend breakdown from the cost_ledger (CostTracker is
    # write-only; this is the aggregate READ path). No agent run, no secrets —
    # only call counts, spend, and tokens. --cost-today and --cost-since are
    # mutually exclusive (both filter the same created_at column).
    if run_cost:
        if cost_today and cost_since:
            click.echo("--cost-today and --cost-since are mutually exclusive")
            sys.exit(2)
        sys.exit(
            asyncio.run(
                _run_cost_report(
                    run_id=cost_run_id or "",
                    model=cost_model or "",
                    by_model=cost_by_model,
                    today=cost_today,
                    since=cost_since or "",
                )
            )
        )

    # --inspect-mutation: read-only inspection of a stored evolution mutation row
    # (id/type/path/status/shape). The SHAPE is classified by
    # promote.classify_payload — the JSON-vs-free-text heuristic that determines
    # whether parse_prompt_payload/parse_code_payload will parse (and thus canary
    # + promote) the row. No secrets are printed; no agent run, no API key.
    if inspect_mutation:
        sys.exit(
            asyncio.run(_run_inspect_mutation(mutation_id, inspect_full))
        )

    # --capability-curve: read-only inspection of the nightly battery trend +
    # regression verdict (the measured-self-improvement evidence). No LLM/DB writes.
    if run_curve:
        asyncio.run(
            _run_capability_curve(
                since=curve_since,
                until=curve_until,
                export=curve_export,
                plot=curve_plot,
                model=curve_model,
            )
        )
        return

    # --retrieval-eval: measure memory recall quality (precision@k + MRR) over a
    # seeded capability fixture — the measurement backbone for the recall pillar
    # (findings-05). Seeds the fixture into warm memory, runs skill recall, and
    # reports + persists metrics. No agent run, no LLM.
    if run_retrieval_eval:
        from src.eval.retrieval import DEFAULT_K

        asyncio.run(_run_retrieval_eval(k=retrieval_k or DEFAULT_K))
        return

    # --aflow: offline workflow-topology optimizer — search per-category
    # technique policies (baseline → propose → evaluate → keep-if-better over
    # real agent runs). Forces EVAL_ENABLED. Opt-in (AFLOW_ENABLED); the
    # optimizer never raises (a runtime failure is a printed AflowResult).
    if run_aflow:
        asyncio.run(_run_aflow())
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
        _clean_run_results(run_id)

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


def _clean_run_results(run_id: str | None) -> None:
    """Delete a run's results subfolder (Phase 7 ``--clean``).

    Validation + resolution delegate to ``_paths.run_subdir_path`` — the single
    source of truth shared with the worker's fresh-attempt clean — so the CLI
    and the worker agree on what a safe run_id is and where its subdir resolves.
    Removes ONLY ``<results_root>/<run_id>`` and never anything escaping it.
    Exits cleanly when the subfolder is already absent (no behavior change).
    """
    from src.tools._paths import clean_run_subdir, run_subdir_path

    if not run_id:
        click.echo("Error: --clean requires --run-id to target a run's subfolder")
        sys.exit(1)
    try:
        sub = run_subdir_path(run_id)
    except ValueError as exc:
        click.echo(f"Error: --clean refused unsafe run_id: {exc}")
        sys.exit(1)
    if not sub.exists():
        click.echo(f"   --clean: no prior subfolder to clear ({sub})")
        return
    if clean_run_subdir(run_id):
        click.echo(f"🧹 Cleared prior results subfolder: {sub}")
    else:
        click.echo(f"Error: --clean could not remove {sub}")
        sys.exit(1)


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


async def _run_backfill_embeddings(
    *, table: str, concurrency: int, dry_run: bool
) -> int:
    """Run an idempotent embedding backfill (capability and/or cold) and exit.

    This is the data half of already-shipped schema migrations that added the
    nullable ``capability_embedding`` / ``cold_memories.embedding`` columns —
    rows created before those fixes have NULL vectors and are invisible to
    semantic recall/dedup. The backfill embeds each NULL row and writes the
    vector back in place; it is idempotent (re-runs are no-ops).

    ``--dry-run`` reports the NULL-row count WITHOUT calling the embedding API
    (no spend) and persists nothing.

    Returns an exit code: 0 when nothing needed backfill, or when at least one
    vector was stored; 1 only when rows needed backfill but none were stored
    (every one failed or was skipped) — a signal the embedding API / key should
    be checked before re-running.
    """
    from src.db.backfills import run_backfill
    from src.db.session import get_session
    from src.memory.embeddings import EmbeddingGenerator

    settings = get_settings()
    generator = EmbeddingGenerator(settings)

    click.echo("=" * 60)
    click.echo(
        f"🧮 Embedding backfill — table={table} concurrency={concurrency} "
        f"dry_run={dry_run}"
    )
    click.echo("=" * 60)

    async with get_session() as session:
        stats = await run_backfill(
            table=table,
            concurrency=concurrency,
            dry_run=dry_run,
            session=session,
            generator=generator,
        )

    click.echo("-" * 60)
    click.echo(
        f"scanned={stats.scanned} stored={stats.stored} "
        f"skipped_hash={stats.skipped_hash} skipped_no_text={stats.skipped_no_text} "
        f"failed={stats.failed}"
    )
    if dry_run:
        click.echo("(dry-run: nothing persisted; no embedding API spend)")
    return 1 if stats.scanned and stats.stored == 0 else 0


def _evidence_str(ev: object) -> str:
    """Render a check's evidence compactly — JSON for dict/list, else str()."""
    if not ev:
        return ""
    if isinstance(ev, (dict, list)):
        return json.dumps(ev, sort_keys=True, default=str)
    return str(ev)


async def _run_score_spec(spec_id: str, deliverables: list[str]) -> int:
    """Score on-disk deliverables against a golden ``GoalSpec`` and exit.

    Resolves ``GOLDEN_SPECS[spec_id]`` and runs ``run_checks`` — the same
    recomputation ground-truth engine the verify node and ``--eval`` use, so a
    passing score means every structural/execution/golden constraint
    *recomputes* from the deliverable content (not that the agent merely claimed
    it). No agent run, no LLM call, no API key needed — pure host-side scoring.

    Returns 0 if the spec passed, 1 if it failed, 2 if ``spec_id`` is unknown.
    """
    from src.eval.checks import run_checks
    from src.eval.golden import GOLDEN_SPECS

    click.echo("=" * 60)
    click.echo(f"🎯 Score spec — {spec_id}  deliverables={len(deliverables)}")
    click.echo("=" * 60)

    if spec_id not in GOLDEN_SPECS:
        click.echo(f"Unknown spec id: {spec_id}")
        click.echo(f"Available: {', '.join(sorted(GOLDEN_SPECS))}")
        return 2

    spec = GOLDEN_SPECS[spec_id]
    if spec.goal_text:
        click.echo(f"goal: {spec.goal_text}")
    for path in deliverables:
        click.echo(f"  • {path}")

    result = await run_checks(spec, deliverables, {})

    click.echo("-" * 60)
    click.echo(
        f"\n>>> {spec_id}: OVERALL SCORE {result.overall_score:.3f}  "
        f"passed={result.passed}\n"
    )
    for check in result.checks:
        flag = "PASS" if check.passed else "FAIL"
        skip = " (skipped)" if getattr(check, "skipped", False) else ""
        err = f"  ERR={check.error}" if getattr(check, "error", None) else ""
        click.echo(
            f"  [{flag}]{skip} {check.check_name:38s} "
            f"score={check.score:.2f}{err}"
        )
        evidence = _evidence_str(check.evidence)
        if evidence:
            click.echo(f"        {evidence}")
    click.echo("=" * 60)
    return 0 if result.passed else 1


_VERIFY_PROMPT = "Reply with exactly one word: pong"


def _default_smoke_models() -> tuple[str, ...]:
    """Drift-proof default model set for ``--verify-models``.

    Derived from the live routing tier maps — NOT a hardcoded tuple — so a
    primary swap (e.g. glm-5.1 → glm-5.2, qwen3.5-flash → qwen3.6-flash) is
    reflected automatically. The prior static list silently covered only 2 of
    ~10 live models and went stale the moment the COMPLEX/CRITICAL primary
    moved to glm-5.2.

    Covers, deduped: every configured routing primary (``COMPLEXITY_TIER_MAP``
    + per-node ``NODE_TIER_MAP``), the defensive ``DEFAULT_COMPLEXITY_TIER``
    (covers the re-enabled Anthropic path for an unmapped complexity), and the
    first cross-provider fallback of the COMPLEX primary (the #783 provider-
    diversity / storm-resilience chain — the one path the bare primaries never
    exercise).

    Map-derived rather than key-aware ``route()`` on purpose: a configured
    primary whose provider key is missing/broken must be EXPOSED by a failed
    ping, not silently skipped to its fallback. Catching a dead chain entry
    before it burns retry slots in real runs is the whole point of the smoke.
    """
    from src.config.model_registry import FALLBACK_CHAINS
    from src.graph.enums import TaskComplexity
    from src.llm.model_router import (
        COMPLEXITY_TIER_MAP,
        DEFAULT_COMPLEXITY_TIER,
        NODE_TIER_MAP,
    )

    keys: dict[str, None] = {}
    for _, chain_key in (*COMPLEXITY_TIER_MAP.values(), *NODE_TIER_MAP.values()):
        keys.setdefault(chain_key, None)
    keys.setdefault(DEFAULT_COMPLEXITY_TIER[1], None)
    complex_primary = COMPLEXITY_TIER_MAP.get(TaskComplexity.COMPLEX, (None, ""))[1]
    chain = FALLBACK_CHAINS.get(complex_primary, [])
    if chain:
        keys.setdefault(chain[0], None)
    return tuple(keys)


def _scrub_secrets(msg: str, *keys: str) -> str:
    """Strip any provider key value (and sk- tokens) from an error message."""
    scrubbed = msg.replace("\n", " ").strip()
    for key in keys:
        if key:
            scrubbed = scrubbed.replace(key, "<redacted>")
    # Generic guard for keys we did not enumerate (never echo a bearer token).
    if "sk-" in scrubbed:
        import re

        scrubbed = re.sub(r"sk-[A-Za-z0-9_\-]{6,}", "<redacted>", scrubbed)
    return scrubbed[:200]


async def _run_verify_models(names: list[str]) -> int:
    """Smoke each model over the gateway's real routing and exit.

    Issues one trivial completion per model via ``gateway.acompletion`` — the
    same path the agent uses — so a passing ping means the registry's
    ``_build_kwargs`` routing (provider/api_base/key) is live, not merely
    unit-tested. Grounds model registration in reality: an unverified
    ``model_id`` in a ``FALLBACK_CHAIN`` burns a retry slot on every failed run.

    Prints NO secrets — only pass/fail, the resolved provider/model, token
    counts, cost, and a sanitized error category (keys are scrubbed). Returns 0
    if every model responded, 1 if any failed (OR of all pings).
    """
    from src.llm.gateway import LLMGateway

    settings = get_settings()
    gateway = LLMGateway(settings)
    models = tuple(names) or _default_smoke_models()

    click.echo("=" * 60)
    click.echo(f"🔌 Model smoke — {len(models)} model(s) via gateway routing")
    click.echo("=" * 60)

    rc = 0
    healthy = 0
    for model in models:
        try:
            resp = await gateway.acompletion(
                messages=[{"role": "user", "content": _VERIFY_PROMPT}],
                model=model,
                temperature=0.0,
                max_tokens=16,
            )
        except Exception as exc:  # noqa: BLE001 — smoke surfaces any failure
            rc = 1
            detail = _scrub_secrets(
                f"{type(exc).__name__}: {exc}",
                settings.llm.dashscope_api_key or "",
                settings.llm.openai_api_key or "",
                settings.llm.anthropic_api_key or "",
            )
            click.echo(f"  [FAIL] {model:28s} {detail}")
            continue
        healthy += 1
        click.echo(
            f"  [OK]   {model:28s} provider={resp.provider} model={resp.model} "
            f"tokens(in={resp.input_tokens},out={resp.output_tokens}) "
            f"cost=${resp.cost_usd:.6f} content={resp.content!r}"
        )

    click.echo("-" * 60)
    click.echo(f"{healthy}/{len(models)} healthy")
    return rc


async def _run_cost_report(
    *, run_id: str, model: str, by_model: bool, today: bool, since: str
) -> int:
    """Print a spend breakdown from the cost_ledger and exit (read-only).

    Delegates the ORM aggregates to ``src.llm.cost_queries.cost_breakdown`` (the
    READ counterpart to the write-only ``CostTracker``) and renders the result
    via the pure ``format_cost_breakdown``. No agent run; NO secrets — only call
    counts, spend, and token totals. Always returns 0 (a read-only report, even
    when zero rows match).
    """
    from src.db.session import get_session
    from src.llm.cost_queries import cost_breakdown, format_cost_breakdown

    click.echo("=" * 60)
    click.echo("💰 Cost-ledger spend breakdown")
    click.echo("=" * 60)
    async with get_session() as session:
        breakdown = await cost_breakdown(
            run_id=run_id,
            model=model,
            by_model=by_model,
            today=today,
            since=since,
            session=session,
        )
    click.echo(format_cost_breakdown(breakdown))
    click.echo("-" * 60)
    return 0


async def _run_inspect_mutation(ident: str | None, full: bool) -> int:
    """Print a stored evolution mutation row + its payload shape, then exit.

    Read-only inspection of the ``mutations`` table. The ``shape`` line reports
    whether the promotion gate will *parse* (and thus canary + promote) the row,
    classified by ``src.evolution.promote.classify_payload`` (the JSON-vs-free-text
    heuristic that mirrors ``parse_prompt_payload``). No secrets are printed; no
    agent run, no API key.

    ``ident`` is a full UUID, a short prefix (as run logs), or ``None`` (latest
    row). Returns 0 when a row is found, 1 when nothing matches.
    """
    import uuid

    from sqlalchemy import Text, cast, select

    from src.db.models import Mutation
    from src.db.session import get_session
    from src.evolution.promote import classify_payload

    stmt = select(Mutation)
    if ident:
        if "-" in ident:
            stmt = select(Mutation).where(Mutation.id == uuid.UUID(ident))
        else:
            # Short prefix (run logs a short id like 4c5c11d4); match newest first.
            stmt = (
                select(Mutation)
                .where(cast(Mutation.id, Text).like(f"{ident}%"))
                .order_by(Mutation.created_at.desc())
            )
    else:
        stmt = select(Mutation).order_by(Mutation.created_at.desc())

    async with get_session() as session:
        row = (await session.execute(stmt)).scalars().first()

    if row is None:
        ident_repr = "latest" if not ident else repr(ident)
        click.echo(f"No mutation found for id={ident_repr}")
        return 1

    content = row.mutated_content or ""
    proposal: dict[str, object] = {
        "mutation_type": row.mutation_type,
        "mutated_content": content,
        "target_path": row.target_path,
    }
    snippet = content if full else content[:800]
    truncated = " …[truncated]" if (not full and len(content) > 800) else ""

    click.echo(f"id            : {row.id}")
    click.echo(f"mutation_type : {row.mutation_type}")
    click.echo(f"target_path   : {row.target_path}")
    click.echo(f"status        : {row.status}")
    click.echo(f"model_used    : {row.model_used}")
    click.echo(f"description   : {row.description}")
    click.echo(f"content_len   : {len(content)} chars")
    click.echo(f"shape         : {classify_payload(proposal)}")
    click.echo("--- mutated_content ---")
    click.echo(f"{snippet}{truncated}")
    return 0


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


def _parse_window(value: str | None, *, end_of_day: bool = False) -> datetime | None:
    """Parse an ISO date/datetime CLI arg to an aware datetime (None when absent).

    Date-only values (e.g. ``2026-06-01``) are interpreted as UTC midnight; when
    ``end_of_day`` is set (the ``--until`` bound) a date-only value is widened to
    23:59:59.999999 of that day so the whole calendar day is included. Naive
    datetimes are assumed UTC (``eval_results.created_at`` is TIMESTAMPTZ).
    """
    if not value:
        return None
    s = value.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError as exc:
        raise click.BadParameter(f"bad ISO date/datetime: {value!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    date_only = ("T" not in s) and (" " not in s)
    if date_only and end_of_day:
        dt = (dt + timedelta(days=1)) - timedelta(microseconds=1)
    return dt


async def _run_capability_curve(
    *,
    since: str | None,
    until: str | None,
    export: str | None,
    plot: str | None,
    model: str | None = None,
) -> None:
    """Print the battery capability curve + regression verdict (read-only).

    Pure read of ``eval_results``: the per-night battery mean, the latest per-goal
    score, and the grounded regression verdict — the measured-self-improvement
    evidence. No LLM, no DB writes. ``--export`` writes JSON/CSV by suffix;
    ``--plot`` writes a PNG (matplotlib optional). ``model`` (the ``--curve-model``
    flag) slices the trend + verdict to one producer model so the curve reads as a
    model-specific self-improvement trend rather than the blended system-wide one.
    """
    from src.eval.curve import CapabilityCurve  # noqa: PLC0415
    from src.eval.store import EvalStore  # noqa: PLC0415

    win_since = _parse_window(since, end_of_day=False)
    win_until = _parse_window(until, end_of_day=True)
    curve = CapabilityCurve(EvalStore())
    snap = await curve.snapshot(since=win_since, until=win_until, model=model)

    title = f"📈 Capability curve — producer {model}" if model else "📈 Capability curve — nightly battery mean correctness"
    click.echo("=" * 60)
    click.echo(title)
    click.echo("=" * 60)

    battery = snap["battery_trend"]
    if not battery:
        click.echo(
            "No eval_results rows in the window. Run a battery first: "
            "`python main.py --eval` or the scheduler."
        )
    else:
        click.echo(f"{'date':<12} {'mean':<8} {'n_goals':<8}")
        for row in battery:
            click.echo(
                f"{str(row.get('date')):<12} "
                f"{_fmt_score(row.get('mean_score')):<8} {row.get('n_goals', 0):<8}"
            )

    click.echo("\n" + "-" * 60)
    click.echo("Latest per-goal score")
    click.echo("-" * 60)
    click.echo(f"{'goal':<18} {'date':<12} {'mean':<8} {'n_checks':<8}")
    for row in snap["latest_per_goal"]:
        click.echo(
            f"{str(row.get('goal_id')):<18} {str(row.get('date') or '-'):<12} "
            f"{_fmt_score(row.get('mean_score')):<8} {row.get('n_checks', 0):<8}"
        )

    verdict = snap["verdict"]
    min_points = get_settings().capability_curve.min_points
    click.echo("\n" + "-" * 60)
    if verdict.get("inconclusive"):
        click.echo(
            f"Regression verdict: INCONCLUSIVE — only {verdict.get('n_points')} night(s) "
            f"in window (need >= {min_points}; see CAPABILITY_CURVE_MIN_POINTS)"
        )
    else:
        status = "REGRESSED ⚠" if verdict.get("regressed") else "OK"
        click.echo(
            f"Regression verdict: {status}  current={_fmt_score(verdict.get('current'))} "
            f"best_prior={_fmt_score(verdict.get('best_prior'))} "
            f"delta={_fmt_score(verdict.get('delta'))} "
            f"(floor={verdict.get('floor')} delta_floor={verdict.get('delta_floor')} "
            f"n={verdict.get('n_points')})"
        )

    if export:
        suffix = Path(export).suffix.lower()
        if suffix == ".json":
            curve.export_json(export, snap)
            click.echo(f"\nExported JSON → {export}")
        elif suffix == ".csv":
            curve.export_csv(export, snap)
            click.echo(f"\nExported CSV → {export}")
        else:
            raise click.BadParameter(
                f"--export supports .json/.csv (got {suffix or 'no suffix'}); use --plot for PNG"
            )
    if plot:
        if curve.plot_png(plot, snap):
            click.echo(f"\nExported PNG → {plot}")
        else:
            click.echo("\nPNG skipped (matplotlib unavailable; install it or use --export .json/.csv).")


async def _run_aflow() -> None:
    """Offline AFlow workflow-topology optimizer (Phase 5 G3b).

    For each target node × seed-spec category, measures the baseline technique
    selection, proposes ``max_candidates`` policies via one gateway call,
    evaluates each against a real agent run (``eval_correctness_score``), keeps
    the best if it clears the improvement margin, and persists it via
    ``AflowPolicyStore``. Forces ``EVAL_ENABLED`` (the fitness needs the verify
    correctness layer). Per-(node, category) baseline/candidates/winner printed.
    The optimizer never raises — a runtime failure is a printed ``AflowResult``.

    Fitness = full ``execute_run`` per seed (the dominant cost); bounded by
    ``AFLOW_MAX_CANDIDATES`` × the seed set per (node, category). Evolution is
    off inside fitness runs so the technique policy is the only variable.
    """
    from src.eval.golden import BATTERY04_GOALS  # noqa: PLC0415
    from src.graph.search.aflow import (  # noqa: PLC0415
        AFlowOptimizer,
        AflowPolicyStore,
        bucket_specs_by_category,
    )

    settings = get_settings()
    aflow = settings.aflow
    if not aflow.enabled:
        click.echo(
            "AFLOW_ENABLED is false — AFlow is opt-in. Set AFLOW_ENABLED=true "
            "(and AFLOW_MAX_CANDIDATES low) to run."
        )
        return
    # Fitness needs the verify correctness layer; force it on for this session.
    settings.eval.eval_enabled = True

    gateway = _create_gateway(settings)
    store = AflowPolicyStore()

    async def run_fn(spec) -> float | None:  # noqa: ANN001 — GoalSpec, lazy import avoided
        run_id = f"aflow-{spec.spec_id}"
        result = await execute_run(
            spec.goal_text,
            spec.max_iterations,
            no_evolution=True,
            run_id=run_id,
        )
        return result.get("eval_correctness_score")

    optimizer = AFlowOptimizer(gateway, store, run_fn, aflow)
    target_nodes = [n.strip() for n in aflow.target_nodes.split(",") if n.strip()] or [
        "execute"
    ]
    bucket = bucket_specs_by_category(list(BATTERY04_GOALS))

    click.echo("=" * 60)
    click.echo("🔀 AFlow workflow-topology optimizer (technique-policy search)")
    click.echo("=" * 60)
    click.echo(
        f"target_nodes={target_nodes}  categories={sorted(bucket)}  "
        f"max_candidates={aflow.max_candidates}"
    )
    for node in target_nodes:
        for category, seeds in sorted(bucket.items()):
            result = await optimizer.optimize(node, category, seeds=seeds)
            click.echo("-" * 60)
            click.echo(
                f"{node}/{category}  promoted={result.promoted} skipped={result.skipped} "
                f"reason={result.reason or '-'}"
            )
            click.echo(
                f"  baseline={_fmt_score(result.baseline)} "
                f"best={_fmt_score(result.best_score)} names={result.names or '-'}"
            )


async def _run_retrieval_eval(k: int) -> None:
    """Measure memory recall quality (precision@k + MRR) over a seeded fixture (E1).

    Seeds ``DEFAULT_FIXTURE`` into warm memory, runs skill recall per
    ``DEFAULT_QUERIES``, and prints + persists the IR metrics. The fixture rows
    are tagged ``retrieval-eval`` so they are identifiable in warm memory; re-runs
    re-seed (duplicates are deduped at recall, so metrics stay sound). No agent
    run, no LLM. Persistence is best-effort (EvalStore is observability-only).
    """
    from src.eval.retrieval import (
        DEFAULT_FIXTURE,
        DEFAULT_QUERIES,
        memory_retriever,
        run_retrieval_eval,
        seed_fixture,
    )
    from src.runner import _async_create_memory_manager

    settings = get_settings()
    # Force the eval store on so the summary row persists (mirrors --eval).
    settings.eval.eval_store_enabled = True

    click.echo("=" * 60)
    click.echo(f"🔍 Retrieval-quality eval — k={k} ({len(DEFAULT_QUERIES)} queries)")
    click.echo("=" * 60)

    memory = await _async_create_memory_manager(settings)
    if memory is None:
        click.echo(
            "Memory unavailable (DB/Redis not wired). The metric engine is "
            "unit-tested in tests/test_eval/test_retrieval.py; run against a "
            "live stack to measure the real recall surface."
        )
        return

    seeded = await seed_fixture(memory, DEFAULT_FIXTURE)
    click.echo(f"Seeded {seeded}/{len(DEFAULT_FIXTURE)} fixture capabilities into warm memory.")
    retriever = memory_retriever(memory, tier="skill", recall_limit=max(k, 10))
    report = await run_retrieval_eval(DEFAULT_QUERIES, retriever, k)

    click.echo(f"\n{'Query':<42} {'P@k':<8} {'RR':<8} {'Hit'}")
    click.echo("-" * 60)
    for q in report.queries:
        click.echo(
            f"{q.query[:42]:<42} {q.precision_at_k:<8.3f} "
            f"{q.reciprocal_rank:<8.3f} {'yes' if q.hit else 'no'}"
        )
    click.echo("\n" + "-" * 60)
    click.echo(
        f"Mean precision@k={report.precision_at_k:.3f} | "
        f"MRR={report.mrr:.3f} | hit_rate={report.hit_rate:.3f}"
    )

    # Best-effort persistence: encode the two metrics as eval_results check rows
    # under goal_id "retrieval-eval" (distinct from the battery goals, so the
    # capability curve is unperturbed). Non-fatal on any write error.
    try:
        from src.eval.models import CheckResult, CorrectnessResult
        from src.eval.store import EvalStore

        correctness = CorrectnessResult(
            spec_id="retrieval-eval",
            overall_score=report.mrr,
            passed=report.mrr > 0.0,
            checks=[
                CheckResult(
                    check_name="retrieval_precision_at_k",
                    check_type="retrieval",
                    passed=report.precision_at_k > 0.0,
                    score=report.precision_at_k,
                    evidence={"k": k, "n_queries": len(report.queries)},
                ),
                CheckResult(
                    check_name="retrieval_mrr",
                    check_type="retrieval",
                    passed=report.mrr > 0.0,
                    score=report.mrr,
                    evidence={"hit_rate": report.hit_rate},
                ),
            ],
        )
        await EvalStore().record_correctness(
            correctness, goal_id="retrieval-eval", run_id="retrieval-eval"
        )
    except Exception as exc:  # noqa: BLE001 — observability-only
        logger.debug("Retrieval-eval metric persistence skipped: {}", exc)


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
