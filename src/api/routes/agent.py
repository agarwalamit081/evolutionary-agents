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
    from src.config import get_settings
    from src.graph.factory import initial_state
    from src.graph.task_graph import compile_task_graph

    settings = get_settings()

    thread_id = f"api-{os.getpid()}-{hash(request.goal) % 10000}"
    state = initial_state(request.goal, thread_id, request.max_iterations)

    # Instantiate dependencies
    gateway: object = None
    memory: object = None
    tools: object = None

    try:
        from src.llm.gateway import LLMGateway

        gateway = LLMGateway(settings)
    except Exception:
        logger.debug("LLMGateway not available for API request")

    try:
        import redis.asyncio as aioredis

        from src.db.session import get_session
        from src.memory.manager import MemoryManager

        redis_client = aioredis.from_url(settings.redis.redis_url)
        async for db_session in get_session():
            memory = MemoryManager(
                redis_client=redis_client,
                db_session=db_session,
                settings=settings,
            )
            break
    except Exception:
        logger.debug("MemoryManager not available for API request")

    try:
        from src.tools import create_default_registry

        tools = create_default_registry()
    except Exception:
        logger.debug("ToolRegistry not available for API request")

    compiled = compile_task_graph(
        gateway=gateway,
        memory=memory,
        tools=tools,
    )

    logger.info(f"API: Starting agent with goal: {request.goal[:80]}")
    result = await compiled.ainvoke(state)

    return RunResponse(
        thread_id=thread_id,
        final_output=result.get("final_output", ""),
        is_complete=result.get("is_complete", False),
        iteration_count=result.get("iteration_count", 0),
        status="completed" if result.get("is_complete") else "incomplete",
    )
