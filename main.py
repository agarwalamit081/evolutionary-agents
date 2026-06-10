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
@click.option("--max-iterations", default=25, type=int, help="Maximum graph iterations")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(
    goal_text: str | None,
    interactive: bool,
    provider: str | None,
    model: str | None,
    no_evolution: bool,
    max_iterations: int,
    verbose: bool,
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

    # Run the agent
    result = asyncio.run(_run_agent(goal_text, max_iterations, no_evolution))

    click.echo("\n" + "=" * 60)
    click.echo("📋 Result:")
    click.echo(result.get("final_output", "No output"))
    click.echo(f"\n   Iterations: {result.get('iteration_count', 0)}")
    click.echo(f"   Completed: {result.get('is_complete', False)}")


async def _run_agent(
    goal_text: str,
    max_iterations: int = 25,
    no_evolution: bool = False,  # noqa: ARG001 — used in future evolution integration
) -> dict:
    """Run the agent graph to completion.

    Instantiates dependencies (LLMGateway, MemoryManager, ToolRegistry)
    and passes them through compile_task_graph() for dependency injection.

    Args:
        goal_text: The goal to accomplish.
        max_iterations: Maximum graph iterations.
        no_evolution: Skip evolution phase.

    Returns:
        Final agent state.
    """
    from src.config import get_settings
    from src.graph.factory import initial_state
    from src.graph.task_graph import compile_task_graph

    settings = get_settings()

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

    # Create initial state
    thread_id = f"cli-{os.getpid()}-{id(goal_text)}"
    state = initial_state(goal_text, thread_id, max_iterations)

    # Instantiate dependencies
    gateway = _create_gateway(settings)
    memory = await _async_create_memory_manager(settings)
    tools = _create_tool_registry()

    # Load previously created dynamic tools from database
    if tools is not None:
        await _load_dynamic_tools(tools)

    # Load active sub-agents from database
    from src.agents.registry import SubAgentRegistry

    sub_agent_registry = SubAgentRegistry()
    await _load_sub_agents(sub_agent_registry)

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

    logger.info(f"Starting agent with goal: {goal_text[:80]}")
    result = await compiled.ainvoke(state)

    logger.info("Agent execution complete")
    return dict(result)


def _create_gateway(settings: Settings):
    """Create LLMGateway if provider key is available."""
    try:
        from src.llm.gateway import LLMGateway

        return LLMGateway(settings)
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


async def _load_dynamic_tools(tools: object) -> None:
    """Load previously created dynamic tools from the database.

    Best-effort — non-fatal if the database is unavailable.
    """
    try:
        from src.tools.dynamic.persister import ToolPersister
        from src.tools.registry import ToolRegistry

        if not isinstance(tools, ToolRegistry):
            return

        persister = ToolPersister()
        loaded = await persister.load_active_tools(tools)
        if loaded:
            logger.info(f"Loaded {len(loaded)} dynamic tools from DB: {', '.join(loaded)}")
    except Exception as e:
        logger.debug(f"Dynamic tool loading skipped: {e}")


async def _load_sub_agents(registry: object) -> None:
    """Load previously created sub-agents from the database.

    Best-effort — non-fatal if the database is unavailable.
    """
    try:
        from src.agents.persister import SubAgentPersister
        from src.agents.registry import SubAgentRegistry

        if not isinstance(registry, SubAgentRegistry):
            return

        persister = SubAgentPersister()
        loaded = await persister.load_active_agents(registry)
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
