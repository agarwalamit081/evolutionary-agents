# Turing Agent — Self-Evolving AI Agent

A production-grade, self-evolving AI agent built with **LangGraph** that continuously improves its reasoning, tooling, memory, and workflow through autonomous mutation, A/B testing, and skill crystallization. Uses **litellm** as a unified gateway to 10+ LLM providers with intelligent cost-aware routing.

---

## Architecture

The agent follows a **6-layer architecture** with strict separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│  Presentation Layer       │  CLI (Click + Rich), FastAPI, HITL  │
├─────────────────────────────────────────────────────────────────┤
│  Orchestration Layer      │  LangGraph StateGraph, Checkpoints  │
├─────────────────────────────────────────────────────────────────┤
│  Agent Layer              │  Task Agent, Meta/Evolution Agent,  │
│                           │  Model Router, Task Classifier      │
├─────────────────────────────────────────────────────────────────┤
│  Evolution Layer          │  Reflector, Mutator, A/B Test,      │
│                           │  Skill Crystallizer, Version Ctrl   │
├─────────────────────────────────────────────────────────────────┤
│  Capability Layer         │  Tool Manager, Memory (3-tier),     │
│                           │  Planner, Verifier, Retriever       │
├─────────────────────────────────────────────────────────────────┤
│  Infrastructure Layer     │  LLM Gateway (litellm), PostgreSQL, │
│                           │  pgvector, Redis, Docker Sandbox    │
└─────────────────────────────────────────────────────────────────┘
```

### Agent Workflow

```
START → classify → plan → retrieve_memory → execute ↔ reflect
  → verify → evolve? → store_memory → hitl? → END
```

- **Task Agent** handles goal execution via ReAct loop with native LangChain tool calling
- **Meta/Evolution Agent** handles self-improvement via analyze → generate → test → deploy pipeline
- **Human-in-the-Loop** gates high-stakes actions (file writes, external APIs, self-replacement)

### Key Features

- **4 Cost Tiers** — from Very Cheap classification to Moderate complex reasoning
- **10+ Providers** — Anthropic, OpenAI, Google, DeepSeek, Z.AI, MiniMax, Mistral, Moonshot, Qwen, Groq, OpenRouter, Ollama
- **3-Tier Memory** — Hot (Redis) → Warm (PostgreSQL) → Cold (pgvector)
- **7-Layer Safety** — AST validation, type checking, semantic analysis, sandboxed execution, regression tests, HITL, version control
- **Skill Crystallization** — successful patterns auto-extracted into versioned, reusable skills (80–95% cost reduction over time)
- **A/B Testing** — statistical comparison before deploying any mutation
- **Budget Enforcement** — 70% warn, 90% critical, 100% hard-cap

---

## Provider Support

All providers accessed through **litellm** as a unified gateway. Models assigned to 4 cost tiers:

| Provider | Tier 0 (Very Cheap) | Tier 1 (Cheap) | Tier 2 (Moderate) |
|---|---|---|---|
| **Anthropic** | — | `claude-haiku-4-5-20251001` | `claude-sonnet-4-6` |
| **OpenAI** | `gpt-4o-mini-2024-07-18` | `gpt-5-nano-2025-08-07` | `gpt-5-mini-2025-08-07` |
| **Google** | `gemini-2.5-flash-lite` | `gemini-2.5-flash` | `gemini-3-flash-preview` |
| **DeepSeek** | — | `deepseek-v4-flash` | `deepseek-v4-pro` |
| **Z.AI** | `glm-4.7-flash` | `glm-4.5-air` | `glm-5-turbo` |
| **MiniMax** | — | `minimax-m2.5-highspeed` | `minimax-m2.5` |
| **Mistral** | — | `mistral-small-2603` | `mistral-medium-3-5` |
| **Moonshot** | — | `moonshot-v1-32k` | `kimi-k2.6` |
| **Qwen** | `qwen3.5-flash` | `qwen3.7-plus` | — |
| **Groq** | `llama-3.1-8b-instant` | `llama-3.3-70b-versatile` | — |
| **Ollama** | `phi4-mini:3.8b` | `qwen3.5:latest` | — |

See `docs/design-docs/02-model-selection.md` for the complete catalog with context windows, max output, and cost per 1M tokens.

---

## Project Structure

```
turing-agent/
├── main.py                     # CLI entry point (Click)
├── requirements.txt
├── .env.example
│
├── config/                     # pydantic-settings
│   ├── settings.py             # BaseSettings classes for all config
│   └── model_registry.py       # Model tier definitions and routing rules
│
├── graph/                      # LangGraph graphs + nodes
│   ├── task_graph.py           # Main StateGraph
│   ├── evolution_graph.py      # Evolution subgraph
│   ├── nodes/                  # classify, plan, execute, reflect, verify, etc.
│   ├── routers.py              # Conditional edge functions
│   └── state.py                # AgentState, EvolutionState TypedDicts
│
├── llm/                        # LLM Gateway (litellm)
│   ├── gateway.py              # LLMGateway class
│   ├── model_router.py         # Complexity → model mapping
│   ├── rate_limiter.py         # aiolimiter per-provider
│   ├── cost_tracker.py         # PostgreSQL cost logging
│   ├── cache.py                # Redis prompt cache
│   ├── structured_output.py    # JSON mode + json-repair
│   └── batch.py                # Batch API support
│
├── memory/                     # 3-tier memory system
│   ├── manager.py              # Unified memory interface
│   ├── hot.py                  # Redis hot memory (ephemeral)
│   ├── warm.py                 # PostgreSQL warm memory
│   ├── cold.py                 # pgvector cold memory (embeddings)
│   ├── consolidation.py        # Background consolidation ("dreaming")
│   └── embeddings.py           # sentence-transformers
│
├── tools/                      # Built-in + evolved tools
│   ├── registry.py             # Tool registry with @tool decorator
│   ├── builtin/                # 7 built-in tools
│   ├── sandbox.py              # Subprocess/Docker execution
│   └── mcp_adapter.py          # fastmcp integration
│
├── evolution/                  # Self-evolution engine
│   ├── engine.py               # SelfEvolutionEngine
│   ├── analyzer.py             # Performance pattern analysis
│   ├── mutator.py              # Code/prompt/workflow mutation
│   ├── ab_test.py              # A/B testing with statistical comparison
│   ├── crystallizer.py         # Skill crystallization
│   └── version_control.py      # Git-based versioning + rollback
│
├── safety/                     # 7-layer safety pipeline
│   ├── pipeline.py             # Orchestrates all 7 layers
│   ├── static_analyzer.py      # AST + ruff checks
│   ├── security_scanner.py     # Forbidden pattern detection
│   ├── semantic_checker.py     # Invariant validation
│   └── sandbox_executor.py     # Isolated code execution
│
├── observability/              # Logging, metrics, tracing
│   ├── logging.py              # loguru + structlog setup
│   ├── metrics.py              # Prometheus metrics
│   └── tracing.py              # OpenTelemetry setup
│
├── api/                        # FastAPI web interface
│   ├── app.py                  # FastAPI application
│   └── routes/                 # health, agent, evolution endpoints
│
├── db/                         # SQLAlchemy models + Alembic
│   ├── engine.py               # Async engine with asyncpg
│   ├── models.py               # ORM models
│   └── migrations/             # Alembic migrations
│
├── tests/                      # Mirror src structure
│   ├── test_graph/
│   ├── test_llm/
│   ├── test_memory/
│   ├── test_tools/
│   ├── test_evolution/
│   ├── test_safety/
│   └── test_api/
│
└── docs/
    └── design-docs/            # 18 design specification documents
```

---

## Quick Start

### 1. Start infrastructure

```bash
# PostgreSQL + Redis via Docker Compose (infra only)
docker compose up -d

# Run database migrations
alembic upgrade head
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — at minimum set:
#   DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/turing_agent
#   REDIS_URL=redis://localhost:6379/0
#   LLM_API_KEY=your-key-here
```

All configuration is managed via **pydantic-settings** `BaseSettings` classes. No `os.environ` in application code.

### 3. Run

```bash
# Run with default provider (from .env)
uv run python main.py --goal "Research the latest developments in LangGraph"

# Specify provider and model
uv run python main.py --provider deepseek --model deepseek-v4-flash --goal "..."
uv run python main.py --provider zai --model glm-5-turbo --goal "..."
uv run python main.py --provider anthropic --model claude-haiku-4-5-20251001 --goal "..."

# Interactive mode
uv run python main.py --interactive

# Skip evolution phase
uv run python main.py --no-evolution --goal "..."
```

### 4. Run tests

```bash
# Unit + integration (no API key needed)
uv run python -m pytest tests/ -v -k "not e2e"

# Full suite including E2E (requires LLM_API_KEY)
LLM_API_KEY=sk-... uv run python -m pytest tests/ -v

# Single test
uv run python -m pytest tests/test_graph/test_nodes.py -v -k "test_classify"
```

---

## Available Tools

All tools use the **LangChain `@tool` decorator** with type-annotated parameters. The LLM selects tools via `bind_tools()` and the agent executes parsed `AIMessage.tool_calls`.

| Tool | Description |
|---|---|
| `code_executor` | Run Python in a subprocess (async, timeout-safe) |
| `code_validator` | AST + security check on Python code |
| `web_search` | Web search via httpx |
| `file_reader` | Read any text file |
| `file_writer` | Write files (creates dirs automatically) |
| `self_inspect` | Read the agent's own source code |
| `memory_search` | Query agent's 3-tier memory |

Tools are registered in a dynamic `ToolRegistry`. Evolved tools generated by the evolution engine are added to the same registry at runtime.

---

## Memory System

3-tier hierarchy with automatic consolidation:

| Tier | Storage | Contents | Latency |
|---|---|---|---|
| **Hot** | Redis | Current task context, ephemeral cache | <1ms |
| **Warm** | PostgreSQL | Skills, procedures, workflows, preferences | ~5ms |
| **Cold** | pgvector | Episodic knowledge, embeddings, long-term patterns | ~20ms |

Background **consolidation** ("dreaming") moves data between tiers: episodic memories in cold storage are periodically analyzed for patterns, which are crystallized into skills in warm storage.

---

## Self-Evolution

4-phase pipeline running as a background process:

1. **Analyze** — query recent task outcomes, identify failure patterns and improvement opportunities
2. **Generate** — propose mutations (prompt, code, tool, workflow, memory, config)
3. **Test** — sandbox execution, regression suite, A/B comparison with statistical significance
4. **Deploy** — canary deployment (1% → 5% → 25% → 100%) with automatic rollback

**Skill crystallization** extracts successful execution patterns into versioned, reusable skills — progressively reducing costs by 80–95% as the agent learns.

---

## Safety

7 defense-in-depth layers for all evolution mutations:

| Layer | Check | Action on Failure |
|---|---|---|
| 1 | Syntax validation (AST parsing) | Reject |
| 2 | Type checking (mypy strict) | Reject |
| 3 | Security scan (forbidden patterns) | Reject |
| 4 | Semantic analysis (behavioral equivalence) | Flag for review |
| 5 | Sandboxed execution (Docker) | Reject |
| 6 | Regression testing (existing tasks) | Reject |
| 7 | Human-in-the-loop approval | Pause for review |

All mutations are version-controlled via Git. Rollback is instant.

---

## Configuration Reference

All config loaded via `pydantic-settings` from `.env` or environment variables. See `docs/design-docs/03-environment-config.md` for the complete spec.

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | — | PostgreSQL connection string |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `LLM_API_KEY` | — | Primary LLM provider API key |
| `FAST_LLM_API_KEY` | — | Fast/cheap model API key (sub-agents) |
| `LLM_PROVIDER` | `deepseek` | Default LLM provider |
| `LLM_MODEL` | `deepseek-v4-flash` | Default model |
| `FAST_LLM_PROVIDER` | `zai` | Fast model provider |
| `FAST_LLM_MODEL` | `glm-4.5-air` | Fast model |
| `TOKEN_BUDGET` | `100000` | Per-task token budget |
| `MAX_COST_USD` | `2.00` | Per-task cost limit |
| `EVOLUTION_ENABLED` | `true` | Enable self-evolution |
| `EVOLUTION_INTERVAL_SECONDS` | `3600` | Evolution cycle frequency |
| `HITL_ENABLED` | `true` | Enable human-in-the-loop gates |

---

## Design Documents

Full technical specifications are in `docs/design-docs/`:

| # | Document | Focus |
|---|---|---|
| 00 | Project Overview | Mission, architecture, principles |
| 01 | Dependencies | Package list, versions, rationale |
| 02 | Model Selection | Provider catalog, cost tiers, routing |
| 03 | Environment Config | .env template, pydantic-settings classes |
| 04 | Database Schema | PostgreSQL DDL, pgvector, Redis patterns |
| 05 | Architecture | 6-layer design, node definitions |
| 06 | Workflow Design | Graph edges, checkpointing, HITL flow |
| 07 | Self-Evolution Engine | Mutation pipeline, A/B testing, crystallization |
| 08 | Memory System | 3-tier hierarchy, consolidation, retrieval |
| 09 | LLM Integration | Caching, streaming, structured output patterns |
| 10 | Safety Guardrails | 7-layer defense, sandbox, version control |
| 11 | Deployment | Docker Compose, CLI, FastAPI, Prometheus |
| 12 | State Schema | AgentState TypedDict, Pydantic models |
| 13 | LLM Gateway | litellm wrapper, rate limiting, retry, cost tracking |
| 14 | Tool System | @tool decorator, registry, MCP integration |
| 15 | Testing Strategy | 3-layer testing, mock fixtures, CI config |
| 16 | Error Handling | Taxonomy, failover chains, budget enforcement |
| 17 | Implementation Guide | Target structure, 5-phase plan, migration path |
