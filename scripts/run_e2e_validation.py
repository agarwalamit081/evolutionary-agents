#!/usr/bin/env python
"""End-to-end validation runner for the Turing Agent.

Runs 5 complex queries in parallel (default 3 workers; EvoAgentX pattern),
captures per-query metrics (models, tokens, costs, timing), and generates a
comprehensive markdown report at docs/e2e-validation-report.md.

Each query runs against an isolated deep-copy of settings, so per-query
``results_root`` redirection and gateway/memory/tool wiring never race under
concurrency.

Usage:
    source /home/amiagarw/aiml01/bin/activate
    python scripts/run_e2e_validation.py                 # 3 parallel workers
    python scripts/run_e2e_validation.py --workers 5     # explicit worker count
    python scripts/run_e2e_validation.py --sequential    # one query at a time
    E2E_NUM_WORKERS=2 python scripts/run_e2e_validation.py
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ─── 5 Validation Queries ─────────────────────────────────────────────

QUERIES: list[dict[str, str]] = [
    {
        "id": "query_1",
        "name": "Sub-Agent Spawning + Multi-Model Routing",
        "text": (
            "Research and compare three independent topics in parallel using specialized "
            "sub-agents: (1) the security implications of pickle vs JSON serialization "
            "in Python with concrete code examples, (2) the performance trade-offs "
            "between asyncio and threading for I/O-bound tasks with benchmark code, and "
            "(3) the scalability differences between SQL and NoSQL databases with "
            "real-world case studies. For each topic, create a dedicated analysis with "
            "runnable code examples saved to a file, then synthesize all three into a "
            "unified comparison report saved to triple_comparison.md."
        ),
    },
    {
        "id": "query_2",
        "name": "Dynamic Tool Creation + Long Execution",
        "text": (
            "Create two custom tools — 'rss_feed_fetcher' and 'html_table_builder' — by "
            "implementing, registering, and smoke-testing each one. Then use both tools "
            "in parallel to fetch the latest headlines from 3 tech news RSS feeds "
            "(TechCrunch, Ars Technica, The Verge) and generate an HTML comparison "
            "table. Use specialized sub-agents for the fetching and report-building "
            "steps. Save the final report to rss_report.html."
        ),
    },
    {
        "id": "query_3",
        "name": "Memory Folding (long task) + Combined Features",
        "text": (
            "Research and document 12 software design patterns (Singleton, Factory, "
            "Observer, Strategy, Adapter, Decorator, Facade, Command, Iterator, State, "
            "Template Method, and Visitor). For EACH pattern: explain the concept, "
            "provide a runnable Python code example, list at least 3 real-world use "
            "cases, and identify common pitfalls. Save each pattern to its own markdown "
            "file under design_patterns/ and create a comprehensive index.md linking "
            "all 12 patterns."
        ),
    },
    {
        "id": "query_4",
        "name": "Error Handling + Recovery + Sub-Agent Delegation",
        "text": (
            "Using specialized sub-agents in parallel — (1) fetch the top 10 GitHub "
            "repositories by stars in the 'machine-learning' topic with retry/backoff "
            "handling of 429 and 5xx errors, (2) fetch the top 10 from the "
            "'deep-learning' topic as a second source, and (3) cross-validate the two "
            "lists, deduplicate, and merge — produce a single markdown table with "
            "columns: repository name, owner, stars, primary language, and last updated "
            "date. Handle all API errors gracefully with retries. Save the merged report "
            "to github_ml_report.md."
        ),
    },
    {
        "id": "query_5",
        "name": "Full Integration (all features)",
        "text": (
            "Perform a comprehensive analysis of renewable energy trends: "
            "(1) Research solar, wind, and hydroelectric power adoption rates in "
            "2025-2026, (2) For each energy source, gather statistics on capacity, "
            "cost per kWh, and growth rate, (3) Create a 'chart_renderer' tool that "
            "turns structured data into a comparison chart and then use it, "
            "(4) Generate a detailed report with charts and data tables, "
            "(5) Save the report and chart to energy_analysis/. "
            "Use specialized sub-agents in parallel for data gathering and report "
            "generation."
        ),
    },
]


# ─── Helpers (reuse from main.py) ─────────────────────────────────────


def _setup_logging() -> None:
    """Initialize loguru logging for the validation run."""
    from src.observability.logging import reset_logging, setup_logging

    reset_logging()
    from src.config.settings import LoggingSettings

    settings = LoggingSettings(log_level="INFO")
    setup_logging(settings)


def _create_gateway(settings: Any) -> Any:
    """Create LLMGateway if provider key is available."""
    from src.llm.gateway import LLMGateway

    return LLMGateway(settings)


async def _create_memory_manager(settings: Any) -> Any:
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
        return None


def _create_tool_registry() -> Any:
    """Create ToolRegistry with all built-in tools."""
    try:
        from src.tools import create_default_registry

        return create_default_registry()
    except Exception:
        return None


async def _load_dynamic_tools(tools: Any) -> None:
    """Load previously created dynamic tools from the database."""
    try:
        from src.tools.dynamic.persister import ToolPersister
        from src.tools.registry import ToolRegistry

        if isinstance(tools, ToolRegistry):
            persister = ToolPersister()
            await persister.load_active_tools(tools)
    except Exception:
        pass


async def _load_sub_agents(registry: Any) -> None:
    """Load previously created sub-agents from the database."""
    try:
        from src.agents.persister import SubAgentPersister
        from src.agents.registry import SubAgentRegistry

        if isinstance(registry, SubAgentRegistry):
            persister = SubAgentPersister()
            await persister.load_active_agents(registry)
    except Exception:
        pass


async def _create_checkpointer(settings: Any) -> Any:
    """Create AsyncPostgresSaver checkpointer if PostgreSQL is available."""
    try:
        from src.graph.checkpoint import create_checkpointer

        return await create_checkpointer(settings.database.database_url)
    except Exception:
        return None


async def _clear_prompt_cache(settings: Any) -> None:
    """Flush the Redis prompt cache so queries are served fresh.

    Only deletes keys under the PromptCache prefix (turing:llm_cache:),
    leaving hot memory and other Redis data intact.
    """
    from loguru import logger

    try:
        import redis.asyncio as aioredis

        from src.llm.cache import PromptCache

        redis_client = aioredis.from_url(settings.redis.redis_url)
        cache = PromptCache(redis_client, settings)
        await cache.invalidate("*")
        await redis_client.aclose()
        logger.info("Prompt cache cleared (turing:llm_cache:*)")
    except Exception as e:
        logger.debug(f"Prompt cache clear skipped: {e}")


# ─── Per-Query Runner ──────────────────────────────────────────────────


async def _run_single_query(
    query: dict[str, str],
    settings: Any,
    max_iterations: int = 18,
) -> dict[str, Any]:
    """Run a single query through the full agent pipeline.

    Returns a metrics dict with models, tokens, costs, timing, etc.
    """
    from loguru import logger

    from src.graph.factory import initial_state
    from src.graph.task_graph import compile_task_graph

    query_id = query["id"]
    query_text = query["text"]

    # Deep-copy settings so this query owns an isolated, mutable copy. The
    # shared ``settings`` object is never mutated — essential under parallel
    # execution, where N concurrent queries would otherwise race on
    # ``agent.results_root``. Each copy redirects its artifacts into a
    # per-query subfolder; no save/restore is needed because the copy is
    # local and discarded when the query returns.
    query_settings = settings.model_copy(deep=True)
    query_results_root = str(Path(query_settings.agent.results_root) / query_id)
    query_settings.agent.results_root = query_results_root
    Path(query_results_root).mkdir(parents=True, exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info(f"Starting {query_id}: {query['name']}")
    logger.info(f"Goal: {query_text[:100]}...")
    logger.info(f"{'='*60}")

    start_time = time.monotonic()

    # Per-query log sink — handler id captured so the sink is torn down before
    # this query returns, preventing two queries' logs from merging in one file
    # under parallel/sequential execution (#313).
    query_sink_id = None
    try:
        from src.observability.logging import add_query_log_sink

        query_sink_id = add_query_log_sink(query_id, query_settings.logging)
    except Exception as e:
        logger.debug(f"Per-query logging skipped: {e}")

    # Build initial state
    thread_id = f"e2e-{query_id}-{int(time.time())}"
    state = initial_state(query_text, thread_id, max_iterations)

    # Instantiate dependencies
    gateway = _create_gateway(query_settings)

    # Inject Redis prompt cache
    if gateway is not None:
        try:
            import redis.asyncio as _aioredis

            from src.llm.cache import PromptCache

            _redis_client = _aioredis.from_url(query_settings.redis.redis_url)
            cache = PromptCache(_redis_client, query_settings)
            gateway.set_cache(cache)
            logger.info("Redis prompt cache injected")
        except Exception as e:
            logger.debug(f"Prompt cache not available: {e}")

    memory = await _create_memory_manager(query_settings)
    tools = _create_tool_registry()

    if tools is not None:
        await _load_dynamic_tools(tools)

    from src.agents.registry import SubAgentRegistry

    sub_agent_registry = SubAgentRegistry()
    await _load_sub_agents(sub_agent_registry)

    checkpointer = await _create_checkpointer(query_settings)

    # Compile graph with injected deps
    compiled = compile_task_graph(
        gateway=gateway,
        memory=memory,
        tools=tools,
        checkpointer=checkpointer,
        sub_agent_registry=sub_agent_registry,
    )

    # Run the graph. recursion_limit is a hard backstop (in case a future
    # routing bug slips through the max-iterations guards) — generous enough
    # for max_iterations cycles (~several nodes each) but bounded.
    recursion_limit = max(max_iterations * 8, 100)
    result = await compiled.ainvoke(
        state, config={"recursion_limit": recursion_limit}
    )
    result_dict = dict(result)

    # Cost/token fallback: if the graph terminated without flushing
    # cost_records via store_memory (e.g. error_termination → END), recover
    # from the gateway's in-memory accumulator so the report always reflects
    # real spend.
    if not result_dict.get("cost_records") and gateway is not None:
        gateway_records = gateway.get_cost_records()
        if gateway_records:
            result_dict["cost_records"] = gateway_records
    if not result_dict.get("total_tokens_used") and gateway is not None:
        records = result_dict.get("cost_records") or []
        if records:
            result_dict["total_tokens_used"] = sum(
                getattr(r, "input_tokens", 0) + getattr(r, "output_tokens", 0)
                for r in records
            )

    elapsed = time.monotonic() - start_time

    # Extract metrics from result state
    cost_records = result_dict.get("cost_records", [])
    total_tokens = result_dict.get("total_tokens_used", 0)
    iteration_count = result_dict.get("iteration_count", 0)
    is_complete = result_dict.get("is_complete", False)
    errors = result_dict.get("errors", [])
    final_output = result_dict.get("final_output", "")

    # Aggregate per-model usage
    model_usage: dict[str, dict[str, int | float]] = {}
    total_cost = 0.0
    for cr in cost_records:
        model = getattr(cr, "model", "unknown")
        provider = getattr(cr, "provider", "")
        inp = int(getattr(cr, "input_tokens", 0))
        out = int(getattr(cr, "output_tokens", 0))
        cost_val = float(getattr(cr, "cost_usd", 0))

        key = f"{model} ({provider})" if provider else model
        if key not in model_usage:
            model_usage[key] = {"input_tokens": 0, "output_tokens": 0, "total_cost": 0.0}
        model_usage[key]["input_tokens"] += inp
        model_usage[key]["output_tokens"] += out
        model_usage[key]["total_cost"] += cost_val
        total_cost += cost_val

    # Sub-agents
    sub_agents_spawned = result_dict.get("sub_agents_spawned", [])
    delegation_results = result_dict.get("delegation_results", [])

    # Tools
    tools_created = result_dict.get("tools_created", [])
    tool_results = result_dict.get("tool_results", [])

    # Memory folding
    fold_history = result_dict.get("fold_history", [])

    # Save per-query results file
    results_dir = Path(query_settings.agent.results_root)
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        goal_safe = re.sub(r'[^a-zA-Z0-9_-]', '_', query_text[:50]).strip('_')
        timestamp = datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')
        results_file = results_dir / f"{goal_safe}_{timestamp}.md"

        content = (
            f"# Agent Results — {query['name']}\n\n"
            f"**Query ID**: {query_id}\n"
            f"**Goal**: {query_text}\n"
            f"**Timestamp**: {timestamp}\n"
            f"**Complete**: {'Yes' if is_complete else 'No'}\n"
            f"**Iterations**: {iteration_count}\n"
            f"**Duration**: {elapsed:.1f}s\n"
            f"**Total Cost**: ${total_cost:.4f}\n\n"
            f"## Output\n\n{final_output}\n"
        )
        results_file.write_text(content, encoding="utf-8")
        logger.info(f"Results file saved: {results_file}")
    except Exception as e:
        logger.debug(f"Results file save skipped: {e}")
        results_file = None

    metrics = {
        "query_id": query_id,
        "query_name": query["name"],
        "query_text": query_text,
        "complete": is_complete,
        "iterations": iteration_count,
        "max_iterations": max_iterations,
        "duration_seconds": round(elapsed, 1),
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "model_usage": model_usage,
        "cost_records_count": len(cost_records),
        "sub_agents_spawned": len(sub_agents_spawned),
        "sub_agent_details": [
            {
                "name": a.get("name", "unknown"),
                "description": a.get("description", ""),
            }
            for a in sub_agents_spawned
        ],
        "delegations_succeeded": sum(
            1 for r in delegation_results if r.get("success", False)
        ),
        "delegations_total": len(delegation_results),
        # tools_called is never populated by execute_node, so derive the real
        # invocation count from tool_results (1:1 with calls; now includes
        # delegated sub-agent activity surfaced by the delegate node).
        "tools_called_count": len(tool_results),
        "tools_created_count": len(tools_created),
        # tools_created records carry `tool_name` (from tool_create_node), not
        # a top-level `name` — fall back to `name` for any legacy shape.
        "tool_creation_details": [
            {
                "name": t.get("tool_name", t.get("name", "unknown")),
                "description": t.get("description", ""),
            }
            for t in tools_created
        ],
        "tool_results_count": len(tool_results),
        "fold_count": len(fold_history),
        "fold_details": [
            {
                # fold records store fold_number (not "iteration"); the iteration
                # each fold fired at is injected by _check_and_fold. Fall back to
                # fold_number so a legacy record still renders something sane.
                "fold_number": f.get("fold_number", 0),
                "iteration": f.get("iteration", f.get("fold_number", 0)),
                "tokens_saved_estimate": f.get("tokens_saved_estimate", 0),
            }
            for f in fold_history
        ],
        "errors": [str(e) for e in errors[-10:]],
        "results_file": str(results_file) if results_file else None,
        "final_output_preview": final_output[:500] if final_output else "",
    }

    logger.info(
        f"Finished {query_id}: "
        f"complete={is_complete}, "
        f"iterations={iteration_count}, "
        f"tokens={total_tokens}, "
        f"cost=${total_cost:.4f}, "
        f"duration={elapsed:.1f}s"
    )

    # Tear down this query's per-query log sink so a subsequent (or concurrent)
    # query does not bleed its logs into this file (#313). None-safe; best-effort.
    if query_sink_id is not None:
        try:
            from src.observability.logging import remove_query_log_sink

            remove_query_log_sink(query_sink_id)
        except Exception as e:
            logger.debug(f"Per-query sink teardown skipped: {e}")

    return metrics


# ─── Report Generator ──────────────────────────────────────────────────


def _generate_report(all_metrics: list[dict[str, Any]], total_duration: float) -> str:
    """Generate the markdown validation report."""
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Models actually used across all queries (routing sends COMPLEX→claude-haiku,
    # CRITICAL→claude-sonnet, etc., so this is dynamic, not hardcoded).
    used_models = sorted({key for m in all_metrics for key in m["model_usage"]})
    model_line = (
        ", ".join(used_models)
        if used_models
        else "none recorded (heuristic fallback)"
    )

    lines: list[str] = [
        "# E2E Validation Report",
        "",
        f"**Generated**: {now}",
        f"**Total Duration**: {total_duration:.1f}s",
        f"**Queries Run**: {len(all_metrics)}",
        f"**Models Used**: {model_line}",
        "",
    ]

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| # | Query | Complete | Iterations | Tokens | Cost | Duration |")
    lines.append("|---|-------|----------|------------|--------|------|----------|")
    for i, m in enumerate(all_metrics, 1):
        complete_icon = "✅" if m["complete"] else "❌"
        lines.append(
            f"| {i} | {m['query_name'][:40]} | {complete_icon} | "
            f"{m['iterations']}/{m['max_iterations']} | "
            f"{m['total_tokens']:,} | ${m['total_cost']:.4f} | "
            f"{m['duration_seconds']}s |"
        )
    lines.append("")

    # Per-query details
    for i, m in enumerate(all_metrics, 1):
        lines.append(f"## Query {i}: {m['query_name']}")
        lines.append("")
        lines.append("### Query Text")
        lines.append("")
        lines.append(f"> {m['query_text']}")
        lines.append("")

        lines.append("### Models Used")
        lines.append("")
        if m["model_usage"]:
            lines.append("| Model (Provider) | Input Tokens | Output Tokens | Cost |")
            lines.append("|-----------------|-------------|--------------|------|")
            for model_key, usage in sorted(
                m["model_usage"].items(), key=lambda x: -x[1]["total_cost"]
            ):
                lines.append(
                    f"| {model_key} | {usage['input_tokens']:,} | "
                    f"{usage['output_tokens']:,} | ${usage['total_cost']:.4f} |"
                )
            lines.append("")
        else:
            lines.append("_No model usage recorded (heuristic fallback)_")
            lines.append("")

        lines.append("### Metrics")
        lines.append("")
        lines.append(f"- **Complete**: {'Yes' if m['complete'] else 'No'}")
        cap_note = (
            " (iteration cap reached — partial result)"
            if not m["complete"] and m["iterations"] >= m["max_iterations"]
            else ""
        )
        lines.append(f"- **Iterations**: {m['iterations']} / {m['max_iterations']}{cap_note}")
        lines.append(f"- **Total Tokens**: {m['total_tokens']:,}")
        lines.append(f"- **Total Cost**: ${m['total_cost']:.4f}")
        lines.append(f"- **Duration**: {m['duration_seconds']}s")
        lines.append(f"- **Cost Records**: {m['cost_records_count']}")
        lines.append("")

        lines.append("### Sub-Agents")
        lines.append("")
        lines.append(f"- **Spawned**: {m['sub_agents_spawned']}")
        lines.append(
            f"- **Delegations**: {m['delegations_succeeded']}/{m['delegations_total']} succeeded"
        )
        if m["sub_agent_details"]:
            for a in m["sub_agent_details"]:
                lines.append(f"  - **{a['name']}**: {a['description'][:80]}")
        lines.append("")

        lines.append("### Tools")
        lines.append("")
        lines.append(f"- **Tool Calls**: {m['tools_called_count']}")
        lines.append(f"- **Dynamic Tools Created**: {m['tools_created_count']}")
        lines.append(f"- **Tool Results**: {m['tool_results_count']}")
        if m["tool_creation_details"]:
            for t in m["tool_creation_details"]:
                lines.append(f"  - **{t['name']}**: {t['description'][:80]}")
        lines.append("")

        lines.append("### Memory Folding")
        lines.append("")
        lines.append(f"- **Fold Count**: {m['fold_count']}")
        if m["fold_details"]:
            for f in m["fold_details"]:
                lines.append(
                    f"  - Fold #{f['fold_number']} @ iteration "
                    f"{f['iteration']} (~{f['tokens_saved_estimate']} tokens saved)"
                )
        else:
            lines.append("- _No folds triggered (iteration count < fold interval or not enough messages)_")
        lines.append("")

        lines.append("### Errors")
        lines.append("")
        if m["errors"]:
            for err in m["errors"]:
                lines.append(f"- {err[:200]}")
        else:
            lines.append("- None recorded in agent state.")
        lines.append("")

        if m["results_file"]:
            lines.append(f"**Results File**: `{m['results_file']}`")
            lines.append("")

        if m["final_output_preview"]:
            lines.append("### Output Preview")
            lines.append("")
            lines.append("```")
            lines.append(m["final_output_preview"])
            lines.append("```")
            lines.append("")

    # Feature validation matrix
    lines.append("## Feature Validation Matrix")
    lines.append("")
    lines.append("| Feature | Q1 | Q2 | Q3 | Q4 | Q5 |")
    lines.append("|---------|----|----|----|----|----|")

    features = [
        ("Sub-Agent Spawning", lambda m: m["sub_agents_spawned"] > 0),
        ("Delegation Success", lambda m: m["delegations_succeeded"] > 0),
        ("Dynamic Tool Creation", lambda m: m["tools_created_count"] > 0),
        ("Tool Execution", lambda m: m["tool_results_count"] > 0),
        ("Memory Folding", lambda m: m["fold_count"] > 0),
        ("Results File Saved", lambda m: m["results_file"] is not None),
        ("Completion", lambda m: m["complete"]),
        ("Cost Tracking", lambda m: m["cost_records_count"] > 0),
        ("Model Usage Tracked", lambda m: len(m["model_usage"]) > 0),
    ]

    for feat_name, check in features:
        cells = []
        for m in all_metrics:
            cells.append("✅" if check(m) else "❌")
        lines.append(f"| {feat_name} | {' | '.join(cells)} |")
    lines.append("")

    # Aggregate totals
    total_cost = sum(m["total_cost"] for m in all_metrics)
    total_tokens = sum(m["total_tokens"] for m in all_metrics)
    total_sub_agents = sum(m["sub_agents_spawned"] for m in all_metrics)
    total_tools_created = sum(m["tools_created_count"] for m in all_metrics)
    total_folds = sum(m["fold_count"] for m in all_metrics)

    lines.append("## Aggregate Totals")
    lines.append("")
    lines.append(f"- **Total Cost**: ${total_cost:.4f}")
    lines.append(f"- **Total Tokens**: {total_tokens:,}")
    lines.append(f"- **Sub-Agents Spawned**: {total_sub_agents}")
    lines.append(f"- **Dynamic Tools Created**: {total_tools_created}")
    lines.append(f"- **Memory Folds**: {total_folds}")
    lines.append(f"- **Total Duration**: {total_duration:.1f}s")
    lines.append("")

    # All models across all queries
    all_models: dict[str, dict[str, int | float]] = {}
    for m in all_metrics:
        for model_key, usage in m["model_usage"].items():
            if model_key not in all_models:
                all_models[model_key] = {"input_tokens": 0, "output_tokens": 0, "total_cost": 0.0}
            all_models[model_key]["input_tokens"] += usage["input_tokens"]
            all_models[model_key]["output_tokens"] += usage["output_tokens"]
            all_models[model_key]["total_cost"] += usage["total_cost"]

    if all_models:
        lines.append("## All Models Used (Across All Queries)")
        lines.append("")
        lines.append("| Model (Provider) | Input Tokens | Output Tokens | Cost |")
        lines.append("|-----------------|-------------|--------------|------|")
        for model_key, usage in sorted(
            all_models.items(), key=lambda x: -x[1]["total_cost"]
        ):
            lines.append(
                f"| {model_key} | {usage['input_tokens']:,} | "
                f"{usage['output_tokens']:,} | ${usage['total_cost']:.4f} |"
            )
        lines.append("")

    return "\n".join(lines)


# ─── Main Runner ────────────────────────────────────────────────────────


def _failure_metrics(
    query: dict[str, str], exc: BaseException, max_iterations: int
) -> dict[str, Any]:
    """Build a zero-value metrics record for a query that raised.

    Keeps the report shape uniform whether a query completed, failed inside the
    graph, or crashed before/after it — so downstream aggregation and the
    feature matrix never key-error on a missing field.
    """
    from loguru import logger

    logger.error(f"Query {query['id']} failed: {exc}")
    return {
        "query_id": query["id"],
        "query_name": query["name"],
        "query_text": query["text"],
        "complete": False,
        "iterations": 0,
        "max_iterations": max_iterations,
        "duration_seconds": 0,
        "total_tokens": 0,
        "total_cost": 0.0,
        "model_usage": {},
        "cost_records_count": 0,
        "sub_agents_spawned": 0,
        "sub_agent_details": [],
        "delegations_succeeded": 0,
        "delegations_total": 0,
        "tools_called_count": 0,
        "tools_created_count": 0,
        "tool_creation_details": [],
        "tool_results_count": 0,
        "fold_count": 0,
        "fold_details": [],
        "errors": [f"FATAL: {exc}"],
        "results_file": None,
        "final_output_preview": "",
    }


async def run_validation(num_workers: int = 3) -> None:
    """Run all queries (in parallel by default) and generate the report.

    Args:
        num_workers: Max concurrent queries. 1 = sequential. Each query runs
            against its own deep-copied settings, so concurrency is safe.
    """
    from loguru import logger

    num_workers = max(1, num_workers)
    _setup_logging()

    from src.config import get_settings

    settings = get_settings()

    # Configure LangSmith
    if settings.langsmith.is_configured:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith.langsmith_api_key  # type: ignore[arg-type]
        os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith.langsmith_project)
        os.environ.setdefault("LANGSMITH_ENDPOINT", settings.langsmith.langsmith_endpoint)
        logger.info(f"LangSmith tracing enabled — project: {settings.langsmith.langsmith_project}")
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

    logger.info(f"Starting e2e validation with {len(QUERIES)} queries")
    logger.info(f"Default model: {settings.llm.default_llm_model}")
    mode = "sequential" if num_workers == 1 else f"parallel ({num_workers} workers)"
    logger.info(f"Execution mode: {mode}")

    # Clear the prompt cache so this run is served fresh (no stale responses).
    await _clear_prompt_cache(settings)

    total_start = time.monotonic()

    # Run queries concurrently with a bounded semaphore. ``gather`` preserves
    # QUERIES order in the result list (result[i] ↔ QUERIES[i]), so the report's
    # Q1..Q5 labels stay aligned. A query that raises never aborts the batch —
    # it maps to a failure-metrics record. Each query deep-copies settings
    # internally, so there is no shared-mutable state between workers.
    semaphore = asyncio.Semaphore(num_workers)

    async def _bounded(query: dict[str, str]) -> dict[str, Any]:
        async with semaphore:
            try:
                return await _run_single_query(query, settings, max_iterations=18)
            except Exception as e:
                return _failure_metrics(query, e, max_iterations=18)

    all_metrics = list(
        await asyncio.gather(*[_bounded(q) for q in QUERIES])
    )

    total_duration = time.monotonic() - total_start

    # Generate report
    report = _generate_report(all_metrics, total_duration)
    report_path = PROJECT_ROOT / "docs" / "e2e-validation-report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    logger.info(f"Report written to: {report_path}")
    logger.info(f"Total validation duration: {total_duration:.1f}s")

    # Print summary to console
    print("\n" + "=" * 60)
    print("E2E VALIDATION COMPLETE")
    print("=" * 60)
    for m in all_metrics:
        icon = "✅" if m["complete"] else "❌"
        print(f"  {icon} {m['query_id']}: {m['query_name'][:40]}")
        print(f"     Iterations: {m['iterations']}, Tokens: {m['total_tokens']:,}, Cost: ${m['total_cost']:.4f}")
    print(f"\n  Total: {total_duration:.1f}s, ${sum(m['total_cost'] for m in all_metrics):.4f}")
    print(f"  Report: {report_path}")
    print("=" * 60)


def _resolve_workers() -> int:
    """Resolve worker count from CLI flags and env (CLI > env > default)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="End-to-end validation runner for the Turing Agent."
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run queries one at a time (disables parallel workers).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (overrides E2E_NUM_WORKERS; default 3).",
    )
    args = parser.parse_args()

    if args.sequential:
        return 1
    if args.workers is not None:
        return max(1, args.workers)
    try:
        return max(1, int(os.environ.get("E2E_NUM_WORKERS", "3")))
    except ValueError:
        return 3


if __name__ == "__main__":
    asyncio.run(run_validation(num_workers=_resolve_workers()))
