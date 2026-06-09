# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Turing Agent — a self-evolving AI agent built with LangGraph that continuously improves its reasoning, tooling, memory, and workflow through autonomous mutation. Uses **litellm** as unified LLM gateway to 10+ providers, **PostgreSQL + pgvector** as sole database, **Redis** for hot memory and caching, and **pydantic-settings** for all configuration.

**Package**: `src/` at project root. CLI entry: `python main.py`.

## Commands

```bash
# Start infrastructure (PostgreSQL + Redis)
docker compose up -d

# Run database migrations
alembic upgrade head

# Run the agent (use aiml01 environment)
source /home/amiagarw/aiml01/bin/activate
python main.py --goal "your goal"
python main.py --interactive                                    # prompt for goal at runtime
python main.py --no-evolution --goal "..."                      # skip evolution phase
python main.py --verbose --goal "..."                           # debug logging

# Run tests (no API key needed for unit + integration)
python -m pytest tests/ -v -k "not e2e"

# Run full suite including E2E (requires provider API key)
OPENAI_API_KEY=sk-... python -m pytest tests/ -v

# Run a single test
python -m pytest tests/test_graph/test_nodes/test_classify.py -v -k "test_classify_trivial"
python -m pytest tests/ -v -k "test_registry" -s   # -s shows print output

# Lint and type check
ruff check .
pyright src/
```

**Environment**: Use `source /home/amiagarw/aiml01/bin/activate` — do NOT use `uv run` (blocked by upstream vllm dep). Use provider-specific API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) NOT generic `LLM_API_KEY`.

## Architecture

### LangGraph StateGraph Flow

```
START → classify → plan → retrieve_memory → execute ↔ reflect
  → tool_create? → plan → verify → evolve? → store_memory → hitl? → END
```

**Dependency injection**: `build_task_graph(gateway, memory, tools)` wraps each node in closures that inject dependencies. When deps are `None`, nodes use heuristic fallback.

### 6-Layer Architecture

| Layer | Modules | Responsibility |
|---|---|---|
| Presentation | `api/`, `main.py` | Click CLI, FastAPI, HITL UI |
| Orchestration | `graph/` | StateGraph, nodes, conditional edges, checkpoints |
| Agent | `llm/gateway.py`, `llm/model_router.py` | LLM gateway, model routing, cost tracking |
| Evolution | `evolution/` | Mutation pipeline, A/B testing, skill crystallization |
| Capability | `tools/`, `memory/` | Tool registry, 3-tier memory, embeddings |
| Infrastructure | `llm/`, `db/`, `observability/` | litellm, PostgreSQL, Redis, Prometheus, OTel |

### Module Structure

| Module | Responsibility |
|---|---|
| `src/config/settings.py` | pydantic-settings `BaseSettings` classes for all config |
| `src/config/model_registry.py` | Model tier definitions and fallback chains |
| `src/graph/task_graph.py` | Main LangGraph StateGraph with DI pattern |
| `src/graph/nodes/` | Node functions: classify, plan, execute, reflect, verify, evolve, tool_create, memory, hitl, error_handler |
| `src/graph/prompts.py` | Centralized prompt templates for LLM-integrated nodes |
| `src/graph/schemas.py` | Pydantic models for structured LLM output |
| `src/graph/routers.py` | Conditional edge routing functions |
| `src/graph/checkpoint.py` | AsyncPostgresSaver factory |
| `src/graph/state.py` | `AgentState`, `EvolutionState` TypedDicts |
| `src/llm/gateway.py` | `LLMGateway` class wrapping litellm |
| `src/llm/model_router.py` | Complexity → cost tier → model mapping |
| `src/llm/structured_output.py` | JSON mode + json-repair fallback |
| `src/memory/manager.py` | Unified 3-tier memory interface |
| `src/memory/hot.py` | Redis ephemeral cache |
| `src/memory/warm.py` | PostgreSQL skills, procedures, workflows |
| `src/memory/cold.py` | pgvector episodic knowledge and embeddings |
| `src/memory/embeddings.py` | Embedding generation (litellm + hash fallback) |
| `src/tools/registry.py` | Dynamic tool registry |
| `src/tools/builtin/` | 7 built-in tools (code_executor, code_validator, web_search, etc.) |
| `src/tools/dynamic/` | Runtime tool generation (generator, persister, allowlist) |
| `src/tools/mcp_adapter.py` | MCP server tool integration |
| `src/evolution/engine.py` | `SelfEvolutionEngine` — 4-phase pipeline |
| `src/safety/pipeline.py` | 7-layer safety gate |
| `src/observability/metrics.py` | Prometheus counters/histograms |
| `src/observability/tracing.py` | OpenTelemetry tracing setup |
| `src/db/models.py` | SQLAlchemy ORM models (PostgreSQL + pgvector) |

### Key Design Decisions

- **litellm** as unified LLM gateway — all chat completions go through `litellm.acompletion()`
- **PostgreSQL + pgvector** as sole database — no SQLite in production. `aiosqlite` for testing only
- **Dependency injection via closures** — `build_task_graph(gateway, memory, tools)` wraps nodes. Heuristic fallback when deps are None
- **LLM-enhanced nodes** — classify, plan, execute, reflect, verify all use LLM when gateway available, fall back to heuristics
- **pydantic-settings** for all configuration — `BaseSettings` classes with `.env` file loading
- **AsyncPostgresSaver** for LangGraph checkpoints — persistent state across runs
- **Budget enforcement** — 70% warn, 90% critical, 100% hard-cap with model tier downgrade
- **Every model has 3-4 fallbacks** across different providers via `FALLBACK_CHAINS`
- **Runtime tool creation** — Agent detects missing tool capabilities, generates them via LLM with double-barrier security (allowlist + constrained namespace), registers in ToolRegistry, persists to DB. Max 3 tools per run

## Testing Structure

Three layers in `tests/` mirroring the source structure:
- **Unit** — individual classes (state, memory, tools, gateway), no LLM calls
- **Integration** — components working together (graph wiring, skill execution, memory consolidation), mocked LLM
- **E2E** (`@pytest.mark.e2e`) — full agent run, requires provider API key

Tests mock `litellm.acompletion` for deterministic LLM responses. `conftest.py` provides `mock_gateway`, `mock_memory`, `mock_tools`, `sample_state` fixtures.

## Environment Configuration

All configuration via `pydantic-settings` (`src/config/settings.py`). Key variables:
- `DATABASE_URL` — PostgreSQL connection string (asyncpg driver)
- `REDIS_URL` — Redis connection string
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc. — provider-specific keys
- `DEFAULT_LLM_PROVIDER` / `DEFAULT_LLM_MODEL` — default model selection

`.env.example` has the full template. See `docs/design-docs/03-environment-config.md` for all classes and validators.

## Design Documents

The 18 documents in `docs/design-docs/` are the authoritative specification for this project. When code and docs conflict, the docs are correct. See `docs/design-docs/00-project-overview.md` for the full document map.
