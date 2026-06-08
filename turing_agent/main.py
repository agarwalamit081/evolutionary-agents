"""Main CLI entry point for the Turing Agent.

Usage:
    python -m turing_agent --interactive
    python -m turing_agent --goal "Build a REST API"
    python -m turing_agent --provider openai --model gpt-4o-mini --goal "Explain quicksort"
"""

from __future__ import annotations

import asyncio
import os
import sys

import click
from loguru import logger

from turing_agent.config import get_settings
from turing_agent.observability.logging import reset_logging, setup_logging


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
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
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
    no_evolution: bool = False,
) -> dict:
    """Run the agent graph to completion.

    Args:
        goal_text: The goal to accomplish.
        max_iterations: Maximum graph iterations.
        no_evolution: Skip evolution phase.

    Returns:
        Final agent state.
    """
    from turing_agent.graph.factory import initial_state
    from turing_agent.graph.task_graph import compile_task_graph

    # Create initial state
    thread_id = f"cli-{os.getpid()}-{id(goal_text)}"
    state = initial_state(goal_text, thread_id, max_iterations)

    # Compile and run graph
    compiled = compile_task_graph()

    logger.info(f"Starting agent with goal: {goal_text[:80]}")
    result = await compiled.ainvoke(state)

    logger.info("Agent execution complete")
    return dict(result)


if __name__ == "__main__":
    main()
