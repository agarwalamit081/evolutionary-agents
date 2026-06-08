"""Agent run/status routes."""

from __future__ import annotations

import os

from fastapi import APIRouter
from loguru import logger
from pydantic import BaseModel

router = APIRouter()


class RunRequest(BaseModel):
    """Request to run the agent."""

    goal: str
    max_iterations: int = 25
    no_evolution: bool = False


class RunResponse(BaseModel):
    """Response from an agent run."""

    thread_id: str
    final_output: str
    is_complete: bool
    iteration_count: int
    status: str


@router.post("/run", response_model=RunResponse)
async def run_agent(request: RunRequest) -> RunResponse:
    """Run the agent to accomplish a goal."""
    from turing_agent.graph.factory import initial_state
    from turing_agent.graph.task_graph import compile_task_graph

    thread_id = f"api-{os.getpid()}-{hash(request.goal) % 10000}"
    state = initial_state(request.goal, thread_id, request.max_iterations)

    compiled = compile_task_graph()

    logger.info(f"API: Starting agent with goal: {request.goal[:80]}")
    result = await compiled.ainvoke(state)

    return RunResponse(
        thread_id=thread_id,
        final_output=result.get("final_output", ""),
        is_complete=result.get("is_complete", False),
        iteration_count=result.get("iteration_count", 0),
        status="completed" if result.get("is_complete") else "incomplete",
    )
