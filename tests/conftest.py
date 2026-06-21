"""Shared test fixtures for the Turing Agent test suite."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.graph.enums import Confidence, Strategy
from src.graph.factory import initial_state
from src.graph.models import PlanStep


# ─── Event Loop ─────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def event_loop():
    """Create a session-scoped event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ─── Mock Gateway ───────────────────────────────────────────────────


@pytest.fixture
def mock_gateway() -> MagicMock:
    """Create a mock LLMGateway with canned responses."""
    from src.llm.models import LLMResponse

    gateway = MagicMock()
    gateway.acompletion = AsyncMock(return_value=LLMResponse(
        content='{"complexity": "simple", "strategy": "react", "estimated_steps": 3, "confidence": 0.8, "reasoning": "test"}',
        model="gpt-4o-mini-2024-07-18",
        provider="openai",
        input_tokens=50,
        output_tokens=100,
        total_tokens=150,
        cost_usd=0.0001,
    ))
    gateway.astream = AsyncMock(return_value=AsyncMock())
    return gateway


# ─── Mock Memory ────────────────────────────────────────────────────


@pytest.fixture
def mock_memory() -> MagicMock:
    """Create a mock MemoryManager."""
    memory = MagicMock()
    memory.retrieve_context = AsyncMock(return_value=[
        {"content": "Test memory", "tier": "hot", "score": 0.9},
    ])
    memory.store_observation = AsyncMock(return_value=None)
    memory.store_skill = AsyncMock(return_value="test-uuid")
    return memory


# ─── Mock Tools ─────────────────────────────────────────────────────


@pytest.fixture
def mock_tools() -> MagicMock:
    """Create a mock ToolRegistry."""
    tools = MagicMock()
    tools.list_tools = MagicMock(return_value=[
        {
            "type": "function",
            "function": {
                "name": "code_executor",
                "description": "Execute Python code",
                "parameters": {"type": "object", "properties": {"code": {"type": "string"}}},
            },
        },
    ])
    tools.get_handler = MagicMock(return_value=AsyncMock(return_value="executed"))
    return tools


# ─── Sample State ───────────────────────────────────────────────────


@pytest.fixture
def sample_state() -> dict[str, Any]:
    """Create a sample AgentState with sensible defaults."""
    return dict(initial_state("Test goal: explain quicksort", "test-thread-001", 10))


@pytest.fixture
def state_with_plan() -> dict[str, Any]:
    """Create a state with a populated plan."""
    state = dict(initial_state("Test goal: implement a REST API", "test-thread-002", 10))
    state["plan_steps"] = [
        PlanStep(id="step1", description="Analyze requirements", status="pending"),
        PlanStep(id="step2", description="Implement endpoints", status="pending"),
        PlanStep(id="step3", description="Test the API", status="pending"),
    ]
    state["current_step_index"] = 0
    state["strategy"] = Strategy.REACT
    return state


@pytest.fixture
def state_after_execution() -> dict[str, Any]:
    """Create a state after some steps have been executed."""
    state = dict(initial_state("Test goal", "test-thread-003", 10))
    state["plan_steps"] = [
        PlanStep(id="step1", description="Step 1", status="completed", result="done"),
        PlanStep(id="step2", description="Step 2", status="completed", result="done"),
        PlanStep(id="step3", description="Step 3", status="pending"),
    ]
    state["completed_steps"] = [
        PlanStep(id="step1", description="Step 1", status="completed", result="done"),
        PlanStep(id="step2", description="Step 2", status="completed", result="done"),
    ]
    state["current_step_index"] = 2
    state["confidence"] = Confidence.HIGH
    return state


@pytest.fixture
def state_with_errors() -> dict[str, Any]:
    """Create a state pre-populated with errors."""
    state = dict(initial_state("Test goal with errors", "test-thread-004", 10))
    state["errors"] = ["something went wrong", "timeout exceeded"]
    return state


@pytest.fixture
def state_complete() -> dict[str, Any]:
    """Create a state with is_complete=True and final_output set."""
    state = dict(initial_state("Test completed goal", "test-thread-005", 10))
    state["is_complete"] = True
    state["final_output"] = "Done: task completed successfully"
    state["plan_steps"] = [
        PlanStep(id="step1", description="Step 1", status="completed", result="done"),
    ]
    state["completed_steps"] = [
        PlanStep(id="step1", description="Step 1", status="completed", result="done"),
    ]
    state["current_step_index"] = 1
    state["confidence"] = Confidence.HIGH
    return state


@pytest.fixture
def mock_redis() -> MagicMock:
    """Create a mock Redis client with a dict backend for hot memory tests."""
    store: dict[str, str] = {}

    redis_mock = MagicMock()
    redis_mock.get = AsyncMock(side_effect=lambda k: store.get(k))
    redis_mock.set = AsyncMock(side_effect=lambda k, v, **_kw: store.__setitem__(k, v))
    redis_mock.delete = AsyncMock(side_effect=lambda *keys: [store.pop(k, None) for k in keys])
    redis_mock.keys = AsyncMock(side_effect=lambda p: [k for k in store if k.startswith(p.replace("*", ""))])
    redis_mock.exists = AsyncMock(side_effect=lambda k: k in store)
    return redis_mock
