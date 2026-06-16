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
def main(
    goal_text: str | None,
    interactive: bool,
    provider: str | None,
    model: str | None,
    no_evolution: bool,
    max_iterations: int,
    verbose: bool,
    run_id: str | None,
    stream: bool,
) -> None:
    """Turing Agent — a self-evolving AI agent built with LangGraph."""
    # Setup logging
    reset_logging()
    settings = get_settings()
    if verbose:
        settings.logging.log_level = "DEBUG"
    setup_logging(settings.logging)

    # Get goal
    if interactive or not goal_text:
        goal_text = click.prompt("Enter your goal")
    if not goal_text:
        click.echo("Error: No goal provided. Use --goal or --interactive.")
        sys.exit(1)

    click.echo(f"🎯 Goal: {goal_text}")
    click.echo(f"   Provider: {provider or 'default'} | Model: {model or 'auto'}")
    click.echo(f"   Max iterations: {max_iterations} | Evolution: {'disabled' if no_evolution else 'enabled'}")

    # Resolve the iteration cap from settings when the CLI didn't override it.
    if max_iterations is None:
        max_iterations = settings.agent.max_iterations

    # Run the agent graph to completion.
    result = asyncio.run(
        _run_agent(goal_text, max_iterations, no_evolution, run_id, model)
    )

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


async def _run_agent(
    goal_text: str,
    max_iterations: int | None = None,
    no_evolution: bool = False,
    run_id: str | None = None,
    model: str | None = None,
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

    # Create initial state
    thread_id = f"cli-{os.getpid()}-{id(goal_text)}"
    state = initial_state(goal_text, thread_id, max_iterations, no_evolution=no_evolution)

    # Instantiate dependencies
    gateway = _create_gateway(settings, pinned_model=model)

    # Inject Redis prompt cache for LLM response caching
    if gateway is not None:
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
    try:
        result = await compiled.ainvoke(
            state,
            config={"recursion_limit": recursion_limit},
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
        from pathlib import Path as _Path

        goal_text_safe = _re.sub(r'[^a-zA-Z0-9_-]', '_', goal_text[:50]).strip('_')
        timestamp = _dt.now(_tz.utc).strftime('%Y%m%d_%H%M%S')
        results_dir = _Path(settings.agent.results_root)
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / f"{goal_text_safe}_{timestamp}.md"

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
