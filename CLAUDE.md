# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Turing Agent — a self-evolving AI agent built with LangGraph that continuously improves its reasoning, tooling, memory, and workflow through autonomous mutation. Uses **litellm** as unified LLM gateway to 10+ providers, **PostgreSQL + pgvector** as sole database, **Redis** for hot memory and caching, and **pydantic-settings** for all configuration. The design docs in `docs/design-docs/` are the authoritative specification — they take precedence over any existing `src/` code.

## Commands

```bash
# Start infrastructure (PostgreSQL + Redis)
docker compose up -d

# Run database migrations
alembic upgrade head

# Run the agent
uv run python main.py --provider <provider> --model <model> --goal "your goal"
uv run python main.py --interactive                                    # prompt for goal at runtime
uv run python main.py --no-evolution --goal "..."                      # skip evolution phase

# Run tests (no API key needed for unit + integration)
uv run python -m pytest tests/ -v -k "not e2e"

# Run full suite including E2E (requires LLM_API_KEY env var)
LLM_API_KEY=sk-... uv run python -m pytest tests/ -v

# Run a single test
uv run python -m pytest tests/test_graph/test_nodes.py -v -k "test_classify"
uv run python -m pytest tests/ -v -k "test_memory_store" -s   # -s shows print output

# Install dependencies
uv pip install -r requirements.txt

# Lint and type check
ruff check .
uv run python -m mypy --strict
```

**Note:** This project uses `uv pip` + `requirements.txt`, not `pyproject.toml`.

## Architecture

### LangGraph StateGraph Flow

```
START → classify → plan → retrieve_memory → execute ↔ reflect
  → verify → evolve? → store_memory → hitl? → END
```

Routing functions inspect `state` to determine the next node (e.g., `route_after_execution`, `route_after_reflection`, `route_after_verification`). The graph is built in `graph/task_graph.py`. Checkpointing uses `AsyncPostgresSaver` for persistent state.

### 6-Layer Architecture

| Layer | Modules | Responsibility |
|---|---|---|
| Presentation | `api/`, CLI (`main.py`) | Click CLI, FastAPI, HITL UI |
| Orchestration | `graph/` | StateGraph, nodes, conditional edges, checkpoints |
| Agent | `llm/model_router.py` | Task Agent, Meta/Evolution Agent, model routing |
| Evolution | `evolution/` | Mutation pipeline, A/B testing, skill crystallization |
| Capability | `tools/`, `memory/` | Tool registry, 3-tier memory, planner, verifier |
| Infrastructure | `llm/gateway.py`, `db/` | litellm gateway, PostgreSQL, Redis, Docker sandbox |

### Target Module Structure

| Module | Responsibility |
|---|---|
| `config/settings.py` | pydantic-settings `BaseSettings` classes for all config |
| `config/model_registry.py` | Model tier definitions and routing rules |
| `graph/task_graph.py` | Main LangGraph StateGraph definition |
| `graph/evolution_graph.py` | Evolution subgraph |
| `graph/nodes/` | Node functions: classify, plan, execute, reflect, verify, evolve, memory, hitl, error_handler |
| `graph/routers.py` | Conditional edge routing functions |
| `graph/state.py` | `AgentState`, `EvolutionState` TypedDicts |
| `llm/gateway.py` | `LLMGateway` class wrapping litellm |
| `llm/model_router.py` | Complexity → cost tier → model mapping |
| `llm/rate_limiter.py` | `aiolimiter` per-provider token buckets |
| `llm/cost_tracker.py` | PostgreSQL cost logging |
| `llm/cache.py` | Redis prompt cache |
| `llm/structured_output.py` | JSON mode + json-repair fallback |
| `memory/manager.py` | Unified 3-tier memory interface |
| `memory/hot.py` | Redis ephemeral cache |
| `memory/warm.py` | PostgreSQL skills, procedures, workflows |
| `memory/cold.py` | pgvector episodic knowledge and embeddings |
| `tools/registry.py` | Dynamic tool registry with `@tool` decorator |
| `tools/builtin/` | 7 built-in tools (code_executor, code_validator, web_search, etc.) |
| `tools/mcp_adapter.py` | fastmcp integration |
| `evolution/engine.py` | `SelfEvolutionEngine` — 4-phase pipeline |
| `evolution/ab_test.py` | A/B testing with statistical comparison |
| `evolution/crystallizer.py` | Pattern → skill extraction |
| `safety/pipeline.py` | 7-layer safety gate |
| `db/models.py` | SQLAlchemy ORM models (PostgreSQL + pgvector) |
| `db/migrations/` | Alembic migrations |

### Key Design Decisions

- **litellm** as unified LLM gateway — all chat completions go through `litellm.acompletion()`. `langchain-openai` and `langchain-anthropic` are kept ONLY for native `bind_tools()` support
- **PostgreSQL + pgvector** as sole database — no SQLite dual-mode in production. `aiosqlite` available for testing only
- **Native LangChain tool calling** — `llm.bind_tools(tool_list)` → parse `AIMessage.tool_calls`. No regex-based parsing
- **pydantic-settings** for all configuration — `BaseSettings` classes with `.env` file loading. No `os.environ` in application code
- **AsyncPostgresSaver** for LangGraph checkpoints — persistent state across runs and crash recovery
- **astream_events(version="v2")** for streaming — replaces any custom streaming wrappers
- **Budget enforcement** — 70% warn, 90% critical, 100% hard-cap with model tier downgrade

### Runtime Data

- **PostgreSQL** — all persistent data (memories, skills, executions, evolution telemetry, checkpoints)
- **Redis** — hot memory cache, prompt cache, rate limiting state
- **evolved_skills/** — Python files generated and deployed by the evolution engine

### State Schema

`AgentState` is a `TypedDict` using `Annotated` reducers. Key fields:
- `messages: Annotated[list, add_messages]`
- `task_description`, `task_type`, `complexity`, `objective`, `constraints`
- `plan: list[PlanStep]`, `current_step_index`, `plan_revision_count`
- `tool_calls: list[dict]`, `tool_results: list[dict]`, `execution_history: list[dict]`
- `reflection: ReflectionResult | None`, `reflection_count`, `max_reflections`
- `model_tier`, `model_used`, `provider_used`
- `total_cost_usd`, `token_budget`, `tokens_used`
- `should_evolve`, `evolution_proposal`, `evolution_approved`
- `hitl_required`, `hitl_question`, `hitl_response`
- `error_state`, `retry_count`

See `docs/design-docs/12-state-schema.md` for the complete field definitions and ownership matrix.

## Testing Structure

Three layers in `tests/` mirroring the source structure:
- **Unit** — individual classes (state, memory, tools, gateway), no LLM calls
- **Integration** — components working together (graph wiring, skill execution, memory consolidation), mocked LLM
- **E2E** (`@pytest.mark.e2e`) — full agent run, requires `LLM_API_KEY`

Tests mock `litellm.acompletion` for deterministic LLM responses. Database tests use `aiosqlite` fixtures for isolation.

## Environment Configuration

All configuration via `pydantic-settings` (`config/settings.py`). Key variables:
- `DATABASE_URL` — PostgreSQL connection string (asyncpg driver)
- `REDIS_URL` — Redis connection string
- `LLM_API_KEY` — primary LLM provider key
- `FAST_LLM_API_KEY` — fast/cheap model key for sub-agents

`.env.example` has the full template. See `docs/design-docs/03-environment-config.md` for all classes and validators.

## Design Documents

The 18 documents in `docs/design-docs/` are the authoritative specification for this project. When code and docs conflict, the docs are correct. See `docs/design-docs/00-project-overview.md` for the full document map.
