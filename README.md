# Turing Agent — Self-Evolving AI Agent

A production-grade, self-evolving AI agent built with **LangGraph** that continuously improves its reasoning, tooling, memory, and workflow through autonomous mutation, A/B testing, and skill crystallization. Uses **litellm** as a unified gateway to 10+ LLM providers with intelligent cost-aware routing.

---

## Architecture

The agent follows a **6-layer architecture** with strict separation of concerns:

```mermaid
flowchart TB
    subgraph "Presentation Layer"
        P1["CLI (Click)"]
        P2["FastAPI"]
        P3["Human-in-the-Loop"]
    end

    subgraph "Orchestration Layer"
        O1["LangGraph StateGraph"]
        O2["AsyncPostgresSaver"]
    end

    subgraph "Agent Layer"
        A1["Task Agent"]
        A2["Meta / Evolution Agent"]
        A3["Model Router"]
        A4["Task Classifier"]
    end

    subgraph "Evolution Layer"
        E1["Reflector"]
        E2["Mutator"]
        E3["A/B Test"]
        E4["Skill Crystallizer"]
    end

    subgraph "Capability Layer"
        C1["Tool Registry"]
        C2["3-Tier Memory"]
        C3["Planner / Verifier"]
    end

    subgraph "Infrastructure Layer"
        I1["LLM Gateway (litellm)"]
        I2["PostgreSQL + pgvector"]
        I3["Redis"]
        I4["Docker Sandbox"]
    end

    P1 & P2 & P3 --> O1
    O1 --> A1 & A2
    A1 --> C1 & C2 & C3
    A2 --> E1 & E2 & E3 & E4
    A3 --> I1
    C1 --> I1
    C2 --> I2 & I3
    E2 & E3 --> I4
```

### Agent Workflow

```mermaid
flowchart TD
    START([START]) --> classify["classify"]
    classify --> plan["plan"]
    plan --> retrieve["retrieve_memory"]
    retrieve --> execute["execute"]

    execute --> reflect["reflect"]
    reflect -->|"Agent gaps"| spawn["agent_spawn"]
    reflect -->|"Tool gaps"| tc["tool_create"]
    reflect -->|"High confidence"| verify["verify"]
    reflect -->|"Low confidence"| execute

    spawn --> delegate["delegate"]
    tc --> plan
    delegate --> verify

    verify --> evolve["evolve"]
    verify --> store["store_memory"]
    evolve --> store
    store --> hitl["HITL Gate"]
    store --> END([END])
    hitl --> END

    classDef node fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#000
    classDef spawn fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000
    classDef tool fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000
    class classify,plan,retrieve,execute,reflect,verify,store node
    class spawn,delegate spawn
    class tc tool
```

### Agent Workflow (text)

```
START -> classify -> plan -> retrieve_memory -> execute <-> reflect
  -> agent_spawn? -> delegate -> tool_create? -> plan -> verify -> evolve? -> store_memory -> hitl? -> END
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
- **Budget Enforcement** — 70% warn, 90% critical, 100% hard-cap with model-tier downgrade
- **Provider Circuit Breaker** — per-provider breaker on the LLM gateway opens after consecutive transient (rate-limit/5xx/timeout) failures, skips to the next fallback-chain provider, and half-open probes recovery; auth (401/403) errors never trip it
- **Prompting-Technique Layer** — a selector maps task complexity + node + goal pattern to the right technique (few-shot for simple, chain-of-thought / least-to-most for complex, chain-of-verification / self-consistency for verify-critical), spliced into plan/execute/reflect/verify prompts from versioned Jinja2 templates
- **Provider-Native LLM Capabilities (opt-in)** — four provider-native features behind independent **default-off** env flags, so behavior is unchanged until toggled on: prompt caching (Anthropic `cache_control` breakpoints on long system prompts), request batching (`LLMGateway.abatch` — `asyncio.gather` + a concurrency cap), per-tier reasoning/thinking (Anthropic `thinking`, OpenAI `reasoning_effort`, DeepSeek/Z.AI `extra_body` — complex on / simple off), and native JSON-schema structured outputs (provider-native `response_format` instead of prompt instructions + `json_repair`). Each verified against litellm 1.83.14
- **Capability Governance** — stored tools (≤25 active) and sub-agents (≤60 active) are curated at load: semantic-dedup against existing capabilities, cumulative-cap retirement of the lowest-scoring, and redundancy retirement keep the ecosystem from bloating across runs
- **Cost-Ledger Resilience** — cost tracking is observability-only: a failed ledger write rolls back its own transaction and is logged, but never aborts the run
- **Verify Grounding** — the verify node spot-checks that cited result paths actually exist on disk before a run is marked complete
- **Autonomous Memory Folding** — the reflect node compresses long live conversations into episode/working/tool summaries (via `RemoveMessage`), persisting them to warm memory for recall on later runs
- **Runtime Tool Creation** — Agent detects missing capabilities, generates tools via LLM with double-barrier security, and registers them for immediate use
- **Sub-Agent Delegation** — Agent spawns specialized sub-agents as isolated LangGraph subgraphs, delegates subtasks, tracks performance with rolling metrics, and optimizes them via the evolution engine
- **Typed Correctness Eval Harness** — beyond process metrics, a correctness layer (Structural / Execution-sandbox / Golden-spec / LLM-judge-oracle checks) scores deliverables and persists results to an `eval_results` table; wired into the verify node behind `EVAL_ENABLED`, runnable standalone via `--eval`
- **Capability Curve + Regression Gate** — nightly `eval_results` scores roll into a per-night battery trend + a grounded regression verdict (floor + delta + min-points conjunction); inspect read-only via `--capability-curve`, and an opt-in scheduler gate (`CAPABILITY_CURVE_GATE_ENABLED`) alerts — or, with `CAPABILITY_CURVE_AUTO_ROLLBACK`, reverts a recent PROMPT promotion — on regression. The measured-self-improvement evidence the thesis needs
- **Verify Completion Discipline** — the agent refuses to force-complete unless the goal's expected deliverable is present, non-empty, and well-formed (placeholder-leak scan for `.md`/`.txt`, parse-check for `.csv`/`.json`); a missing deliverable triggers a re-plan, never a false success
- **Per-Tool Metrics + Performance Retirement** — each tool invocation records success/empty/latency; governance retires tools below a success-rate floor once they have enough runs, alongside semantic-dedup and cap retirement
- **Semantic/Fact Memory Tier** — durable entity-ish facts (`memory_type="fact"`) extracted during folding and recalled alongside skills/episodes
- **Cross-Process `--resume`** — a killed/interrupted run resumes from its last Postgres checkpoint via `--resume <run-id>`
- **Per-Run Results Subfolders** — writes organize under `results/<run-id>/` (reads fall back to the flat root for backward recall); `--results-dir` / `--clean` CLI flags
- **Evolution→Live Promotion Gate** — a PROMPT mutation that passes post-deploy verify promotes to a versioned, canary-gated pointer (auto-rollback on regression); opt-in via `EVOLUTION_PROMOTE_TO_LIVE`. Exercised live: a real run deployed a PROMPT mutation, the GoldenCanary passed, and the gate wrote the live pointer (`.turing/evolved/prompts/current.json`) for the prompt builder to splice in tagged `[evolved]`
- **Centralized Config** — every resilience/circuit-breaker/rate-limiter/tool-limit/concurrency knob is a `pydantic-settings` env var (no hardcoded timeouts/caps in source)
- **Run-Control Safety** — four guards bound every run so a deployed worker can never churn forever: a capability-cap gap-loop break (`CAP_LOOP_BREAK_THRESHOLD`, **on by default** — stops the spawn↔create churn once caps saturate), an opt-in wall-clock timeout (`WORKER_RUN_TIMEOUT_S` → resumable), an opt-in budget hard-stop (`BUDGET_HARD_STOP` — raises instead of silently downgrading onto a cheaper/free-tier provider and fabricating under degradation), and a graceful cancel endpoint (`POST /runs/{id}/cancel`). Exhausted / cancelled / timed-out runs land in terminal `BUDGET_EXHAUSTED` / `CANCELLED` / `TIMEOUT` statuses (acked, not redelivered) and resume from their last checkpoint via `--resume`

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

> **Note (temporary, revert by 2026-07-01):** Anthropic is currently **disabled from routing** — the key is under an account usage cap (returns 400/quota, which does not trip the circuit breaker). Until the cap resets, the COMPLEX/SIMPLE primaries fall through to `deepseek-v4-flash` / `glm-4.7` and anthropic is skipped in every fallback chain. This is config-gated in `src/llm/model_router.py` (`_TEMPORARY_DISABLED_PROVIDERS`) and reverts with a one-line change; the Anthropic `ModelSpec`s and fallback chains remain defined (inert) so the routing topology is unchanged. All other providers above work normally.

See `docs/ARCHITECTURE.md` for the full architectural overview.

---

## Project Structure

```
turing-agent/
├── main.py                     # CLI entry point (Click)
├── requirements.txt
├── .env.example
│
└── src/
    ├── config/                 # pydantic-settings
    │   ├── settings.py         # BaseSettings classes for all config
    │   └── model_registry.py   # Model tier definitions and routing rules
    │
    ├── graph/                  # LangGraph graphs + nodes
    │   ├── task_graph.py       # Main StateGraph
    │   ├── nodes/              # classify, plan, execute, reflect, verify, etc.
    │   ├── prompts.py          # Centralized prompt templates
    │   ├── schemas.py          # Pydantic models for structured LLM output
    │   ├── routers.py          # Conditional edge functions
    │   └── state.py            # AgentState, EvolutionState TypedDicts
    │
    ├── llm/                    # LLM Gateway (litellm)
    │   ├── gateway.py          # LLMGateway class
    │   ├── model_router.py     # Complexity → model mapping
    │   ├── rate_limiter.py     # aiolimiter per-provider
    │   ├── cost_tracker.py     # PostgreSQL cost logging
    │   ├── cache.py            # Redis prompt cache
    │   ├── structured_output.py # JSON mode + json-repair + native JSON-schema (opt-in)
    │   ├── prompt_cache_control.py # Anthropic cache_control breakpoints (opt-in)
    │   ├── thinking_control.py # Per-tier reasoning/thinking params (opt-in)
    │   └── circuit_breaker.py  # Per-provider circuit breaker
    │
    │
    ├── memory/                    # 3-tier memory system
    │   ├── manager.py             # Unified memory interface
    │   ├── hot.py                 # Redis hot memory (ephemeral)
    │   ├── warm.py                # PostgreSQL warm memory
    │   ├── cold.py                # pgvector cold memory (embeddings)
    │   └── embeddings.py          # Embedding generation (litellm + hash fallback)
    │
    ├── tools/                     # Built-in + dynamic tools
    │   ├── registry.py            # Tool registry with @tool decorator
    │   ├── builtin/               # 14 built-in tools (7 core + 7 capability-expansion)
    │   ├── dynamic/               # Runtime tool generation
    │   │   ├── generator.py       # LLM tool generation + validation
    │   │   ├── persister.py       # DB persistence for generated tools
    │   │   └── allowlist.py       # Safe module allowlist + namespace
    │   └── mcp_adapter.py         # fastmcp integration
    │
    ├── agents/                    # Sub-agent system
    │   ├── registry.py            # Sub-agent registry + spawn
    │   ├── persister.py           # DB persistence + rolling metrics
    │   ├── subgraph.py            # Dynamic LangGraph subgraph builder
    │   ├── runner.py              # Sub-agent executor + parallel delegation
    │   └── state.py               # SubAgentState TypedDict
    │
    ├── evolution/                 # Self-evolution engine
    │   ├── engine.py              # SelfEvolutionEngine (4-phase pipeline)
    │   ├── git_tracker.py         # Git-based mutation versioning
    │   ├── report.py              # Evolution reporting
    │   └── templates.py           # Mutation templates
    │
    ├── safety/                    # 7-layer safety pipeline
    │   └── pipeline.py            # All 7 safety layers (consolidated)
    │
    ├── sandbox/                   # Isolated code execution
    │   └── executor.py            # Subprocess/Docker execution sandbox
    │
    ├── observability/             # Logging, metrics, tracing
    │   ├── logging.py             # loguru structured logging
    │   ├── metrics.py             # Prometheus metrics
    │   └── tracing.py             # OpenTelemetry setup
    │
    ├── api/                       # FastAPI web interface
    │   ├── app.py                 # FastAPI application
    │   └── routes/                # health, agent endpoints
    │
    └── db/                        # SQLAlchemy models + Alembic
        ├── engine.py              # Async engine with asyncpg
        ├── models.py              # ORM models
        └── migrations/            # Alembic migrations

tests/                             # Mirror src structure
├── test_graph/
├── test_llm/
├── test_memory/
├── test_tools/
├── test_agents/
├── test_e2e/                     # End-to-end tests (requires OPENAI_API_KEY)
├── test_evolution/
├── test_safety/
└── test_api/

docs/
└── ARCHITECTURE.md               # Architectural narrative with Mermaid diagrams
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

> **⚠️ Never `docker compose down -v`** — the `-v` flag deletes the `pgdata` and
> `redisdata` named volumes, destroying all warm memories, cost ledger rows, and
> evolution history. Use `docker compose down` (no flags) to stop the containers
> while preserving data; `docker compose up -d` resumes on the same volumes.
>
> The host runs against the docker-published ports (`5433` Postgres, `6380`
> Redis — the non-default host ports avoid clashing with a host-local
> Postgres/Redis that other apps may own on `5432`/`6379`; the container ports
> remain the standard `5432`/`6379`). If a host-local Postgres/Redis is also
> listening on those published ports it will silently shadow the container —
> every run logs the resolved database on connect (e.g.
> `Database engine created → localhost:5433/turing_agent`); verify the
> host/database match the container (`self-evolving-agent-postgres-1` / `turing_agent`).

> **⚠️ Do not run the legacy `../web-search/docker-compose.yml`** — that older
> standalone stack (the `ws_*` containers: `ws_searxng`/`ws_meilisearch`/
> `ws_postgres`/`ws_redis` + its own api/worker) is fully superseded by this
> project's `turing-*` services. Bringing it up alongside the app duplicates
> SearXNG/Meilisearch/Postgres/Redis on clashing ports and wastes resources.
> It is kept only as a reference; if it is ever running, stop it with
> `docker compose -f ../web-search/docker-compose.yml down` (plain, no `-v`).

### 2. Configure

```bash
cp .env.example .env
# Edit .env — at minimum set:
#   DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5433/turing_agent
#   REDIS_URL=redis://localhost:6380/0
#   OPENAI_API_KEY=your-key-here
```

All configuration is managed via **pydantic-settings** `BaseSettings` classes. No `os.environ` in application code.

### 3. Run

```bash
# Activate the project environment
source /home/amiagarw/aiml01/bin/activate  # or your virtual environment path

# Run with default provider (from .env)
python main.py --goal "Research the latest developments in LangGraph"

# Specify provider and model
python main.py --provider deepseek --model deepseek-v4-flash --goal "..."
python main.py --provider zai --model glm-5-turbo --goal "..."
python main.py --provider anthropic --model claude-haiku-4-5-20251001 --goal "..."

# Interactive mode
python main.py --interactive

# Skip evolution phase
python main.py --no-evolution --goal "..."

# Stream the final answer token-by-token to stdout
python main.py --stream --goal "..."

# Run the golden eval suite (correctness checks + LLM-judge) and persist results
python main.py --eval

# Inspect the nightly battery capability curve + regression verdict (read-only; no LLM/DB writes)
python main.py --capability-curve --export /tmp/curve.json

# Per-run output organization — writes land under results/<run-id>/; --clean clears it first
python main.py --goal "..." --run-id my-task --clean
python main.py --goal "..." --results-dir results/custom      # override the run's results root

# Cross-process resume — continue a killed/interrupted run from its last checkpoint
python main.py --resume my-task
```

### 4. Run tests

```bash
# Unit + integration (no API key needed)
python -m pytest tests/ -v -k "not e2e"

# Full suite including E2E (requires OPENAI_API_KEY)
OPENAI_API_KEY=sk-... python -m pytest tests/ -v

# Single test
python -m pytest tests/test_graph/test_nodes/test_classify.py -v -k "test_classify_trivial"
```

---

## Available Tools

All tools use the **LangChain `@tool` decorator** with type-annotated parameters. The LLM selects tools via `bind_tools()` and the agent executes parsed `AIMessage.tool_calls`.

| Tool | Description |
|---|---|
| `code_executor` | Run Python in a subprocess (async, timeout-safe) |
| `code_validator` | AST + security check on Python code |
| `terminal_command` | Allowlisted, shell-free terminal command tool |
| `file_reader` | Read any text file |
| `file_writer` | Write files (creates dirs automatically) |
| `list_directory` | List directory entries within a sandboxed root |
| `web_search` | Ground facts/current events: SearXNG (free, primary) + paid fallback chain (see Search & Retrieval) |
| `corpus_search` | Hybrid keyword+vector recall over previously-fetched pages (Meilisearch) |
| `web_scraper` | Fetch a URL and return its main content as clean markdown |
| `http_request` | Make controlled HTTP requests to external APIs/services |
| `document_parser` | Extract text from PDF/DOCX/XLSX/CSV documents |
| `environment_inspect` | Inspect the runtime environment (OS, CPU, disk, RAM, packages) |
| `get_current_time` | Return the current wall-clock timestamp, timezone, and date |
| `self_inspect` | Read the agent's own source code |
| `memory_search` | Query agent's 3-tier memory |

Tools are registered in a dynamic `ToolRegistry`. Evolved tools generated by the evolution engine are added to the same registry at runtime.

### Search & Retrieval

The agent grounds facts and current events via the **free tool path** (`web_search` → `web_scraper` → `corpus_search`), not LLM self-browsing (which costs and hallucinates). `web_search` is **SearXNG-primary** (free, self-hosted in the compose stack) with a **paid fallback chain** tried in order (`SEARCH_FALLBACK_PROVIDERS`), and a **hybrid corpus index** (Meilisearch) so previously-fetched pages recall without re-scraping. Search results are cached in Redis and pages in Meilisearch, so repeat/nearby queries skip the network.

Hardened capabilities:

| Capability | What it does |
|---|---|
| Rate-limit pacing | `WEB_SEARCH_DELAY_MIN/MAX` aiolimiter spaces concurrent/batched queries (no burst-fire) |
| Meilisearch task-await | `corpus.py` polls the index `taskUid` to `succeeded`, so index-then-search in one coroutine sees the docs |
| Tavily depth | `search_depth`/`topic`/`days`/`include_domains`/`exclude_domains`/`score` filtering on the Tavily leg |
| Query operators | `site:`/`filetype:`/`"exact"`/`-exclude` + `multi_query` phrasing variants, merged + deduped |
| Anti-bot fallback | on HTTP 403/429, retries once via `curl_cffi` Chrome-JA3 TLS impersonation (Cloudflare/bot-WAF bypass) |
| Result cache | Redis cache keyed on canonical query+params; **dynamic TTL** — shorter for recency-sensitive queries (`timelimit` or `latest`/`news`/`today`) |
| Pooled client | a batch shares **one** `httpx.AsyncClient` across queries (TCP/TLS reuse) under `SEARCH_BATCH_CONCURRENCY` |

Env knobs: `SEARCH_*`, `WEB_SEARCH_*`, `WEB_SCRAPER_*`, `MEILISEARCH_*`, `TOOL_CACHE_*` (see `.env.example`).

The agent can also **create new tools at runtime** when it detects a capability gap. When the LLM calls a non-existent tool, the reflect node identifies the missing capability, the `tool_create` node generates the tool via LLM, validates it through the 7-layer safety pipeline with a double-barrier security model (static analysis + constrained execution namespace), registers it in the `ToolRegistry` for immediate use, and persists it to PostgreSQL for future runs. Max 3 tools per run. See `docs/ARCHITECTURE.md` for details.

### Sub-Agent Delegation

The agent can also **spawn specialized sub-agents** as isolated LangGraph subgraphs. When the reflect node detects a need for specialized processing, the `agent_spawn` node designs a sub-agent via LLM, validates it through the safety pipeline, persists it to PostgreSQL, and registers it for immediate use. The `delegate` node then routes subtasks to the appropriate sub-agent, tracks performance with rolling metrics (success rate, cost, latency, quality), and auto-deprecates underperformers. Sub-agents are optimized over time by the main agent's evolution engine. Max 3 sub-agents per run. See `docs/ARCHITECTURE.md` for details.

---

## Memory System

3-tier hierarchy with automatic consolidation:

| Tier | Storage | Contents | Latency |
|---|---|---|---|
| **Hot** | Redis | Current task context, ephemeral cache | <1ms |
| **Warm** | PostgreSQL | Skills, procedures, workflows, preferences | ~5ms |
| **Cold** | pgvector | Episodic knowledge, embeddings, long-term patterns | ~20ms |

Background **consolidation** ("dreaming") moves data between tiers: episodic memories in cold storage are periodically analyzed for patterns, which are crystallized into skills in warm storage.

**Autonomous memory folding** keeps long-running tasks within context limits: when the live conversation grows past a token/message threshold, the `reflect` node compresses it in parallel into three structured summaries — episode (key events/decisions), working (current goals/next actions), and tool (usage patterns/rules). Folding genuinely shrinks context (LangGraph `RemoveMessage` deletes the old messages), persists the summaries to warm memory, and recalls them on later runs — bounded by `MEMORY_FOLDING_MAX_FOLDS` per run. See `docs/ARCHITECTURE.md` for the memory-system design.

---

## Self-Evolution

4-phase pipeline running as a background process:

1. **Analyze** — query recent task outcomes, identify failure patterns and improvement opportunities
2. **Generate** — propose mutations (prompt, code, tool, workflow, memory, config)
3. **Test** — sandbox execution, regression suite, A/B comparison with statistical significance
4. **Deploy** — canary deployment (1% → 5% → 25% → 100%) with automatic rollback

**Skill crystallization** extracts successful execution patterns into versioned, reusable skills — progressively reducing costs by 80–95% as the agent learns.

**Objective-success evolution** — evolution fires on any run that reached its objective deliverable, including ones that logged and recovered from transient errors along the way (the richest learning signal), not only pristine clean runs. Genuine failures (no deliverable + low confidence) are still suppressed, so failure is never crystallized.

**Capability-curve + regression gate** — the measured-self-improvement evidence: a pure analytics layer (`CapabilityCurve`) turns nightly `eval_results` scores into a per-night battery trend + a grounded regression verdict (`current < score_floor` AND `(best_prior - current) >= regression_delta` AND `>= min_points` nights — floor+delta both required so a noisy-but-acceptable curve never trips). Inspect it read-only with `python main.py --capability-curve` (per-night table, latest per-goal score, the verdict; `--since`/`--until` window, `--export` JSON/CSV, `--plot` PNG). The nightly scheduler can run a `CurveRegressionGate` (`CAPABILITY_CURVE_GATE_ENABLED`) that, on regression, alerts via telemetry + Prometheus and — when `CAPABILITY_CURVE_AUTO_ROLLBACK` is opted in — reverts a recent PROMPT promotion via the existing `PromotionGate.rollback`. A regression with no active promotion is model/provider drift and is alert-only. Both knobs default off; detection is always-on/read-only when the gate is enabled.

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

All config loaded via `pydantic-settings` from `.env` or environment variables. See `.env.example` for the complete template.

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | — | PostgreSQL connection string (asyncpg driver) |
| `REDIS_URL` | `redis://localhost:6380/0` | Redis connection string |
| `OPENAI_API_KEY` | — | OpenAI provider API key |
| `ANTHROPIC_API_KEY` | — | Anthropic provider API key |
| `DEEPSEEK_API_KEY` | — | DeepSeek provider API key |
| `ZAI_API_KEY` | — | Zhipu AI / GLM provider API key |
| `DEFAULT_LLM_PROVIDER` | `deepseek` | Default LLM provider |
| `DEFAULT_LLM_MODEL` | `deepseek-v4-flash` | Default model |
| `FAST_LLM_PROVIDER` | `openai` | Fast/cheap model provider |
| `FAST_LLM_MODEL` | `gpt-4o-mini-2024-07-18` | Fast model for classification/routing |
| `REASONING_LLM_PROVIDER` | `anthropic` | Reasoning model provider |
| `REASONING_LLM_MODEL` | `claude-sonnet-4-6` | Reasoning model for complex tasks |
| `DAILY_TOKEN_BUDGET` | `500000` | Daily token budget |
| `MAX_ITERATIONS` | `60` | Graph-build recursion-limit basis (`recursion_limit = max(max_iterations*8, 100)`); must be ≥ the largest tier cap below (validated at startup) |
| `MAX_ITERATIONS_TRIVIAL` | `12` | Complexity-aware runtime cap — a TRIVIAL goal stops loop-hunting early. Only applies when no explicit cap is pinned (no `--max-iterations` / worker job / eval spec); an explicit cap always wins |
| `MAX_ITERATIONS_SIMPLE` | `15` | Runtime cap for a SIMPLE goal (default when complexity is unset) |
| `MAX_ITERATIONS_COMPLEX` | `60` | Runtime cap for a COMPLEX goal (keeps full headroom) |
| `MAX_ITERATIONS_CRITICAL` | `60` | Runtime cap for a CRITICAL goal (keeps full headroom) |
| `CONVERGENCE_STABLE_THRESHOLD` | `3` | Convergence early-exit — if verify produces an identical output fingerprint across this many consecutive passes AND the plan is exhausted, accept the partial result via `store_memory` instead of looping to the iteration hard-cap. 3: one transient repeat is common; three unchanged passes is a real plateau. Does not set `is_complete` (stuck, not done) |
| `EVOLUTION_ENABLED` | `true` | Enable self-evolution |
| `EVOLUTION_INTERVAL` | `10` | Evolution trigger interval (tasks) |
| `HITL_ENABLED` | `true` | Enable human-in-the-loop gates |
| `LANGCHAIN_TRACING_V2` | `false` | Enable LangSmith tracing for LangGraph and litellm |
| `LANGSMITH_API_KEY` | — | LangSmith API key (required if tracing enabled) |
| `LANGSMITH_PROJECT` | `turing-agent` | LangSmith project name |
| `MEMORY_FOLDING_ENABLED` | `true` | Enable autonomous mid-run context folding |
| `MEMORY_FOLDING_INTERVAL` | `6` | Cooldown window (iterations between folds); tuned to fit `max_iterations` |
| `MEMORY_FOLDING_TOKEN_THRESHOLD` | `50000` | Live-token trigger for folding |
| `MEMORY_FOLDING_MESSAGE_FLOOR` | `10` | Minimum messages before folding is considered |
| `MEMORY_FOLDING_MESSAGE_THRESHOLD` | `14` | Message-count trigger for folding |
| `MEMORY_FOLDING_MAX_FOLDS` | `3` | Maximum folds per agent run |
| `PROMPT_CACHE_CONTROL_ENABLED` | `false` | Enable Anthropic-native prompt caching (`cache_control` breakpoints on long system prompts); `PROMPT_CACHE_CONTROL_MIN_SYSTEM_TOKENS` gates the cache-write cost |
| `LLM_BATCH_ENABLED` | `false` | Enable concurrent request batching via `LLMGateway.abatch`; `LLM_BATCH_MAX_CONCURRENCY` (default `5`) caps in-flight calls |
| `REASONING_CONTROL_ENABLED` | `false` | Enable per-tier extended thinking (complex/critical on, trivial/simple off, provider-native); `REASONING_CONTROL_*` tunes the effort + Anthropic budgets |
| `NATIVE_STRUCTURED_OUTPUT_ENABLED` | `false` | Enable provider-native JSON-schema `response_format` for structured outputs (OpenAI/DeepSeek strict, Anthropic `output_format`, Gemini `response_schema`) |
| `CAPABILITY_CURVE_GATE_ENABLED` | `false` | Register the nightly curve regression→rollback gate on the scheduler (detection + telemetry + Prometheus always on; rollback is a separate knob) |
| `CAPABILITY_CURVE_CURVE_CRON` | `0 5 * * *` | 5-field crontab for the gate (default 05:00 UTC, after the 02:00 battery so it reads the just-written night) |
| `CAPABILITY_CURVE_TIMEZONE` | `UTC` | IANA zone for the gate cron |
| `CAPABILITY_CURVE_REGRESSION_DELTA` | `0.1` | Minimum drop (`best_prior - current`) to flag a regression |
| `CAPABILITY_CURVE_SCORE_FLOOR` | `0.5` | A regression requires `current` below this floor too (floor + delta both required — a delta-only dip below a held floor is NOT a regression) |
| `CAPABILITY_CURVE_LOOKBACK_DAYS` | `30` | Only PROMPT promotions promoted within this many days are rollback suspects |
| `CAPABILITY_CURVE_MIN_POINTS` | `2` | Fewer nights than this is INCONCLUSIVE (no verdict, no rollback) |
| `CAPABILITY_CURVE_AUTO_ROLLBACK` | `false` | On regression, revert a recent suspect PROMPT promotion via `PromotionGate.rollback` (opt-in; off = alert only) |
| `WORKER_RUN_TIMEOUT_S` | `0` (off) | Run-level wall-clock timeout (seconds); on expiry the run is marked `TIMEOUT` + acked (terminal, resumable via `--resume`). Bounds non-terminating cap-saturation loops. A per-run override is `POST /run {run_timeout_s}` |
| `BUDGET_HARD_STOP` | `false` | On per-run token-cap exhaustion, RAISE `BudgetExhaustedError` (→ `BUDGET_EXHAUSTED`, resumable) instead of silently downgrading onto a cheaper/free-tier provider. Off = current downgrade behavior |
| `PER_TASK_TOKEN_LIMIT` | `200000` | Per-run cumulative token cap (raised 100K→200K so a large COMPLEX run converges before downgrade) |
| `CAP_LOOP_BREAK_THRESHOLD` | `2` | After this many consecutive cap-blocked spawn/create node-visits with no new capability (caps saturated), the routers stop re-routing into spawn/create and route to plan/verify (forces convergence). Default 2 = one fully-saturated spawn+create cycle (3 needs 1.5 cycles a slow model never reaches inside the timeout). On by default — resets to 0 on any real progress |
| `GOVERNANCE_PRUNE_ENABLED` | `false` | Register a periodic capability-governance prune job on the scheduler — re-runs `consolidate.py` retire/redundancy + the tool cap-enforce so a long-lived worker frees cap headroom between restarts |
| `GOVERNANCE_PRUNE_CRON` | `0 4 * * *` | 5-field crontab for the prune (default 04:00 UTC, clear of the 05:00 curve-gate + 02:00 battery) |
| `GOVERNANCE_PRUNE_TIMEZONE` | `UTC` | IANA zone for the prune cron |

---

## Known Limitations & Validation Status

**Validated end-to-end on default routing (2026-06-23, `showcase-vector-db-2`):** a single complex research→analysis→report query exercised the full feature set live — runtime tool creation + reuse (`vector_db_comparator` created, registered, executed), web search (62 SearXNG queries, 5 sources cited), sub-agent delegation, the prompting-technique selector (`step_back` / `generated_knowledge` / `chain-of-thought` / `self_ask` / `first_principles` / `checklist`), 3 autonomous memory folds, and **a live evolution cycle** (a PROMPT mutation deployed to `src/prompts/system_prompt.md`). The primary deliverable was written and verified present. Total ≈ $0.22.

**Resolved — per-query log sink leak (#313, commit `30b5cf6`, local-only).** The "post-success non-termination + goal drift" once attributed to this run was a **misdiagnosis**: the run actually **succeeded and terminated correctly** on its objective (evolution fired ~03:05; the immutable `submitted_goal` anchor kept `verify` on the vector-db objective throughout). The symptom — unrelated q01 `Goal:` lines appearing in the showcase log — was caused by `add_query_log_sink` adding a process-global loguru sink and discarding the returned handler id with no teardown, so the lingering sink captured the *next* run's logs (a separate `bench-battery04_q01-*` benchmark merged into the showcase log). **Fix:** `add_query_log_sink` now returns the handler id and `remove_query_log_sink` (None-safe) tears it down in `execute_run`'s `finally` block and in `_run_single_query`. Short/converging runs always terminated normally.

**Anthropic temporarily disabled** from routing (account usage cap; revert by 2026-07-01) — see the note under *Provider Support* above.

**Run-control hardening — known caveat (budget hard-stop resume).** When `BUDGET_HARD_STOP=true` and a run exhausts its per-run token cap, the gateway raises `BudgetExhaustedError` and the worker marks the run `BUDGET_EXHAUSTED` (acked, terminal, checkpoint persisted). The run is resumable via `--resume <run-id>`, BUT token usage is cumulative across the run_id, so a resumed run re-trips the cap immediately — a resume-window delta (baseline the spent count at resume) is a deferred follow-up. Until then, prefer re-issuing the goal rather than resuming a budget-exhausted run.

---

## Architecture Documentation

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full architectural narrative covering three generations of the evolutionary agent, detailed Mermaid diagrams for tool creation and sub-agent spawning pipelines, key findings from testing, and design decisions.

## License

This project is licensed under the **PolyForm Noncommercial License 1.0.0**. 

**Non-commercial use is permitted.** Any commercial use (including integration into proprietary products, SaaS, or paid services) requires explicit written permission from the copyright holder. 

For commercial licensing inquiries, please contact: agarwalamit081@gmail.com