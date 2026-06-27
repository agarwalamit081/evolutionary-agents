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
- **Typed Correctness Eval Harness** — beyond process metrics, a correctness layer (Structural / Execution-sandbox / Golden-spec / LLM-judge-oracle / **State** checks) scores deliverables and persists results to an `eval_results` table; wired into the verify node behind `EVAL_ENABLED`, runnable standalone via `--eval`. **State** checks assert over live graph state (e.g. `current_goal.complexity`) rather than files — the kind that makes a node's in-state decision (classify) scoreable, so the optimizer's real canary can actually move on a node-prompt candidate. A run with **no** battery GoalSpec still gets a generic **ad-hoc deliverable eval** row (parsed/non-empty per on-disk file) via `EVAL_ADHOC_DELIVERABLES` (default on) — observability-only, never enforced, so fresh/ad-hoc queries are evaluated too
- **Capability Curve + Regression Gate** — nightly `eval_results` scores roll into a per-night battery trend + a grounded regression verdict (floor + delta + min-points conjunction); inspect read-only via `--capability-curve`, and an opt-in scheduler gate (`CAPABILITY_CURVE_GATE_ENABLED`) alerts — or, with `CAPABILITY_CURVE_AUTO_ROLLBACK`, reverts a recent PROMPT promotion — on regression. The measured-self-improvement evidence the thesis needs
- **Metric-Driven Prompt Optimizer (DSPy+GEPA sidecar)** — the *improvement* loop C1's measurement left out: a nightly, cost-bounded, **default-off** job runs DSPy/GEPA to search a better prompt for a node against a cheap proxy metric, then VALIDATES the winner against the real golden canary (full agent runs) before promoting it through the existing canary-gated gate. Refuses to run while the capability curve is regressed/inconclusive. DSPy only (**no torch**); `textgrad` is a deferred backend. The canary selects **node-sensitive** specs first (`GoalSpec.target_node`): for `target_node=classify`, two classify-sensitive canary specs assert `current_goal.complexity` via State checks — without them the canary fed only data-correctness specs whose score is inert to classify prose, so a promotion was *structurally impossible*; now a proxy-win can become a real canary lift
- **Verify Completion Discipline** — the agent refuses to force-complete unless the goal's expected deliverable is present, non-empty, and well-formed (placeholder-leak scan for `.md`/`.txt`, parse-check for `.csv`/`.json`); a missing deliverable triggers a re-plan, never a false success
- **Per-Tool Metrics + Performance Retirement** — each tool invocation records success/empty/latency; governance retires tools below a success-rate floor once they have enough runs, alongside semantic-dedup and cap retirement
- **Semantic/Fact Memory Tier** — durable entity-ish facts (`memory_type="fact"`) extracted during folding and recalled alongside skills/episodes
- **Cross-Process `--resume`** — a killed/interrupted run resumes from its last Postgres checkpoint via `--resume <run-id>`
- **Per-Run Results Subfolders** — writes organize under `results/<run-id>/` (reads fall back to the flat root for backward recall); `--results-dir` / `--clean` CLI flags. This includes **`code_executor` compute deliverables** (`.csv`/`.json` written by LLM-generated Python): the subprocess-bootstrap shim relocates their `open()` writes/reads under `results/<run-id>/`, and a fresh (non-resume) attempt auto-cleans that subdir so a re-enqueued run never inherits a prior attempt's files. The shim also relocates `glob.glob`/`glob.iglob`/`os.listdir`/`os.scandir` for results-prefixed targets (subdir-first + flat-fallback), and the run status (`GET /runs/{id}`) surfaces `results_dir` (the resolved `results/<run-id>/`) so a caller finds the artifacts without guessing
- **Evolution→Live Promotion Gate** — a PROMPT mutation that passes post-deploy verify promotes to a versioned, canary-gated pointer (auto-rollback on regression); opt-in via `EVOLUTION_PROMOTE_TO_LIVE`. Exercised live: a real run deployed a PROMPT mutation, the GoldenCanary passed, and the gate wrote the live pointer (`.turing/evolved/prompts/current.json`) for the prompt builder to splice in tagged `[evolved]`
- **Centralized Config** — every resilience/circuit-breaker/rate-limiter/tool-limit/concurrency knob is a `pydantic-settings` env var (no hardcoded timeouts/caps in source)
- **Run-Control Safety** — four guards bound every run so a deployed worker can never churn forever: a capability-cap gap-loop break (`CAP_LOOP_BREAK_THRESHOLD`, **on by default** — stops the spawn↔create churn once caps saturate), an opt-in wall-clock timeout (`WORKER_RUN_TIMEOUT_S` → resumable), an opt-in budget hard-stop (`BUDGET_HARD_STOP` — raises instead of silently downgrading onto a cheaper/free-tier provider and fabricating under degradation), and a graceful cancel endpoint (`POST /runs/{id}/cancel`). Exhausted / cancelled / timed-out runs land in terminal `BUDGET_EXHAUSTED` / `CANCELLED` / `TIMEOUT` statuses (acked, not redelivered) and resume from their last checkpoint via `--resume`
- **LATS/MCTS Reasoning Search (Phase 5, opt-in)** — on a CRITICAL goal that stalls on a low-confidence retry, a real per-call MCTS tree search (`lats_search` node) imagines and scores candidate next-steps via gateway-only rollouts + an LLM value function, then commits the UCB-best step for single-trajectory execution — no side-effect fan-out, no cost blow-up. Stateless per call (checkpoint/resume-safe); a fail-safe returns the plan's original step unchanged
- **AFlow Workflow-Topology Optimization (Phase 5, opt-in)** — search over the *planning-topology policy* (which prompting techniques wire into each node, per goal-category) — a distinct axis from the prompt-text optimizer and the mutation engine. An offline, DI, cost-bounded optimizer proposes technique-policy candidates and keeps one only if it beats the baseline by the margin against real agent-run fitness; a runtime hook overrides `TechniqueSelector` byte-identically when off
- **Evolution Safety (Phase 5)** — a self-*evolving* agent that rewrites its own graph/prompts needs guardrails before promotion is trustworthy: a **stage-1 graph-invariant verifier** (5 checks on CODE mutations: compiles, a dynamic subprocess import-smoke, AgentState-superset, router-returns-known-nodes, and no self-loops — fail ⇒ rollback; termination/budget are runtime-enforced, not static) and **VCS-tracked, locally-gated prompt promotion** (a promoted prompt lands as a committed tracked artifact gated by a passing local eval/canary before the live pointer flips)
- **Agent Autonomy (Phase 5, opt-in)** — the agent can now **schedule its own future work** (durable `scheduled_tasks` table + `create_scheduled_task` tool + a scheduler consumer that enqueues runs at each cron tick) and **ingest external codebases** (`git_clone` → AST-chunk → embed → pgvector → semantic `code_search`)
- **Neo4j Entity/Relation Graph (Phase 5, opt-in)** — an additive relationship substrate: when `GRAPH_ENABLED`, the memory write hooks mirror **structured** records (skills/procedures/workflows + dependencies, facts-about-entities, sub-agent defs) into Neo4j nodes/edges — relationships the relational + pgvector stores can't express ("which skills depend on X / which sub-agent handles Y"). Default-off, lazy driver, never-raises (a graph outage can't abort a run)

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
    │   ├── builtin/               # 16 built-in tools (core + capability-expansion; see Available Tools)
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
    ├── optimizer/                 # Metric-driven prompt optimizer (DSPy+GEPA sidecar)
    │   ├── engine.py             # PromptOptimizer: GEPA search → canary validate → promote
    │   ├── profiles.py           # Per-node DSPy student/trainset/proxy-metric (classify ships)
    │   ├── server.py             # aiohttp /optimize + /healthz (internal-only)
    │   └── models.py             # OptimizeRequest/OptimizeResponse wire models
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
├── test_optimizer/               # DSPy/GEPA optimizer engine + server + integration
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
# (includes code_executor compute deliverables; a re-enqueued run-id also auto-cleans on a fresh attempt)
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
| `code_executor` | Run Python in a subprocess (async, timeout-safe). Deliverables from `open()` are isolated under `results/<run-id>/` when `RESULTS_PER_RUN_SUBDIR` is on (no flat cross-run leakage); `glob.glob`/`glob.iglob`/`os.listdir`/`os.scandir` are likewise relocated for `results/`-prefixed targets (subdir-first + flat-fallback), so the agent's own `glob('results/*.csv')` finds its deliverables (`pathlib.Path.glob/iterdir/rglob` remain a documented edge). In host mode the subprocess `cwd` is bound to the results subtree and an opt-in `CODE_EXECUTOR_HOST_PATH_GUARD` blocks `open()` of paths resolving outside the workspace (D8); docker/runner modes are already sandboxed |
| `code_validator` | AST + security check on Python code |
| `terminal_command` | Allowlisted, shell-free terminal command tool |
| `file_reader` | Read any text file |
| `file_writer` | Write files (creates dirs automatically) |
| `list_directory` | List directory entries within a sandboxed root |
| `web_search` | Ground facts/current events: SearXNG (free, primary) + paid fallback chain (see Search & Retrieval) |
| `corpus_search` | Hybrid keyword+vector recall over previously-fetched pages (Meilisearch) |
| `web_scraper` | Fetch a URL and return its main content as clean markdown |
| `http_request` | Make controlled HTTP requests to external APIs/services |
| `document_parser` | Extract text from HTML + PDF/DOCX/XLSX/CSV; opt-in pymupdf figure/table extraction from PDFs (D1, D3) |
| `arxiv_search` | Search arXiv papers → title, authors, summary, published, pdf_url (D7) |
| `environment_inspect` | Inspect the runtime environment (OS, CPU, disk, RAM, packages) |
| `get_current_time` | Return the current wall-clock timestamp, timezone, and date |
| `self_inspect` | Read the agent's own source code |
| `memory_search` | Query agent's 3-tier memory |
| `git_clone` | Clone a public git repo into a confined workspace, AST-chunk its code, embed it, and index it into pgvector for semantic `code_search` (SSRF-guarded, size/path-capped) — I2 |
| `create_scheduled_task` | Persist an agent-settable durable cron task (validated cron, capped count) that the scheduler consumer enqueues as a run at each fire — I1 |

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

The agent can also **create new tools at runtime** when it detects a capability gap. When the LLM calls a non-existent tool, the reflect node identifies the missing capability, the `tool_create` node generates the tool via LLM, validates it through the 7-layer safety pipeline with a double-barrier security model (static analysis + constrained execution namespace), registers it in the `ToolRegistry` for immediate use, and persists it to PostgreSQL for future runs. Max 3 tools per run. A generated tool **must ship a `test_code` containing an `assert`** and pass `ruff check --select F,E9` lint — the cheapest-first shared `validate_tool_code` gate (test-assert → pyflakes/syntax lint → 7-layer safety → optional sandbox smoke) that also backs the operator tool-edit API (D9). See `docs/ARCHITECTURE.md` for details.

### Sub-Agent Delegation

The agent can also **spawn specialized sub-agents** as isolated LangGraph subgraphs. When the reflect node detects a need for specialized processing, the `agent_spawn` node designs a sub-agent via LLM, validates it through the safety pipeline, persists it to PostgreSQL, and registers it for immediate use. The `delegate` node then routes subtasks to the appropriate sub-agent, tracks performance with rolling metrics (success rate, cost, latency, quality), and auto-deprecates underperformers. Sub-agents are optimized over time by the main agent's evolution engine. Max 3 sub-agents per run. See `docs/ARCHITECTURE.md` for details.

### Tooling breadth (Phase 3)

A batch closing the tooling gaps documented in `docs/findings.md` P2. All provider-agnostic (no anthropic dependency), validatable live.

| Item | Change |
|---|---|
| **D1** | `document_parser` reads HTML/Markdown via the same trafilatura→markdownify chain `web_scraper` uses (one shared path) |
| **D3** | `document_parser` gains an opt-in `extract_figures` flag → pymupdf (`fitz`) table + figure rendering to PNG under the results root (pypdf stays the default text path) |
| **D5** | `selectolax` + `mdformat` + `mistune` added to deps + the dynamic allowlist (fast HTML parse + markdown render for generated tools; `markitdown` rejected — its `magika` dep pulls `onnxruntime` into every image) |
| **D6** | `matplotlib` installed + allowlisted; `code_executor` bootstrap sets `MPLBACKEND=Agg` so headless `savefig` works |
| **D7** | `arxiv_search` builtin (`arxiv` package) → papers as `{title, authors, summary, published, pdf_url, entry_id}`, so arXiv isn't re-created every run |
| **D8** | host-subprocess `code_executor` binds `cwd` to the per-run results subdir, and an opt-in `CODE_EXECUTOR_HOST_PATH_GUARD` injects an `open()` wrapper rejecting paths outside the workspace tree. docker/runner modes are already confined; host mode is relative-confined / absolute-trusted unless the guard is on |
| **D9** | generated tools must ship a `test_code` with an `assert` and pass `ruff check --select F,E9` — the shared `validate_tool_code` gate |
| **D10** | operator **tool-edit → review → approve** HITL API + `ToolVersion.status` migration (below) |

#### Operator tool-edit API (D10)

A stored generated tool can be edited by an operator, validated against the same bar a runtime-generated tool must clear, staged pending review, then approved/rejected before it reaches the live registry. Endpoints under `/api/v1/tools` (import-wrapped in `app.py`, like the agent router):

| Method | Path | Behavior |
|---|---|---|
| `PATCH` | `/{name}` | body `{handler_code, test_code, description?, input_schema?}` → runs the shared `validate_tool_code` (assert + ruff `F,E9` + 7-layer safety; `sandbox=None` — the stateless API has no Docker, so the functional smoke is deferred to load/run) → on pass, stages a new `pending_review` ToolVersion (`202`); on fail, `422` with the reason |
| `POST` | `/{name}/approve` | promote the latest `pending_review` version to live (`status='approved', is_active=True`; deactivates every other version) |
| `POST` | `/{name}/reject` | dismiss the latest `pending_review` version (`?reason=` optional) |
| `GET` | `` | list generated tools + their latest version/status |
| `GET` | `/{name}` | inspect one tool + its version history |

**`ToolVersion.status` lifecycle** — migration `i1b2c3d4e5f6` adds `status: NOT NULL DEFAULT 'approved'` (values `approved`/`pending_review`/`rejected`) and backfills existing rows to `approved`. A staged edit parks at `pending_review` + `is_active=False` and never touches prior versions, so the live tool keeps running unchanged while the edit is under review. `load_active_tools` now requires `status='approved'` *and* `is_active` (defense-in-depth; the backfill guarantees no stored tool regresses). A pending edit never runs until approved.

### Routing, selection & recall (Phase 4)

The P2 batch from `docs/findings.md` that turns the codebase's dead *recall* pillar into real, measurable behavior — embeddings for tools and sub-agents are now consumed by **selection**, not just governance dedup. All seven items ship **default-off / opt-in** (zero behavior change unless toggled) and are **provider-agnostic** (no anthropic dependency), so they're validatable live against glm-4.7 / deepseek / gemini while anthropic is under quota.

| Item | Change | Knob(s) |
|---|---|---|
| **F2** | Operator JSON env overrides merged on top of the curated `NODE_TIER_MAP` / `COMPLEXITY_TIER_MAP` at `route()` call-time — pin a node (`{"COMPLEX:execute": "glm-4.7"}`) or a whole complexity tier. Empty / unparseable ⇒ identical behavior | `ROUTING_NODE_TIER_OVERRIDES_JSON`, `ROUTING_COMPLEXITY_TIER_OVERRIDES_JSON` |
| **E1** | Retrieval-quality harness — `src/eval/retrieval.py` computes **precision@k + MRR** over a deterministic seeded fixture via a `Retriever` protocol (the real `MemoryManager` recall surface). Inspect with `python main.py --retrieval-eval`; rows written via `EvalStore` (no new table). The A/B backbone that justifies E2/F1 ("did recall improve?") | `--retrieval-eval` (CLI) |
| **E2** | Tool **score-blend**: retrieve top-N (≥k) by cosine, then re-rank by `cosine · (base + weight·success_rate·(1−empty_output_rate))`, take top_k. Reads the per-tool `success_rate` / `empty_output_rate` already maintained atomically in `metrics.py`. `weight=0` ⇒ pure cosine; any error ⇒ full-set fallback | `TOOL_RETRIEVAL_BLEND_SUCCESS` |
| **F1** | Semantic **sub-agent selection**: before the delegate fan-out, embed the subtask and rank the *already-spawned* subset against `sub_agent_definitions.capability_embedding`, running only the top-k. Default-off ⇒ current all-fan-out; fail-safe ⇒ all-spawn | `AGENT_SELECTION_ENABLED`, `AGENT_SELECTION_TOP_K` |
| **F3** | **Tool category tags + MCP hints** (`readOnlyHint` / `destructiveHint` / `idempotentHint` / `openWorldHint`) on every builtin + the dynamic generator schema; a tool whose `destructiveHint` is True routes its invocation through the existing HITL interrupt. In-memory registry lookup — **no DB migration** | `DESTRUCTIVE_TOOL_HITL_ENABLED` |
| **E3** | **structure_analysis LLM-assist**: after the regex pass, when on AND the goal is COMPLEX/CRITICAL AND regex found nothing/ambiguous, a one-shot gateway call (glm-4.7) returns `{tool_gaps, agent_gaps}`, json-repair-salvaged and merged into `pending_*_gaps` (respecting `attempted_*` + the single-shot `structure_analysis_done` guard). Fail-safe ⇒ regex-only | `STRUCTURE_ANALYSIS_LLM_ASSIST_ENABLED` |
| **D2** | **Gateway multimodal/vision**: `LLMGateway.acompletion(images=...)` folds images into the last user message as OpenAI-format text + `image_url` blocks, and the fallback chain is restricted to `ModelSpec.supports_images` entries (original chain kept if none are image-capable). `_estimate_tokens` tolerates list-content. Default-off ⇒ byte-identical text-only | `VISION_ENABLED` |

### Evolution depth, autonomy & reasoning search (Phase 5)

The P3 batch from `docs/findings.md` plus two pulled-forward P4 items — evolution **depth** (invariant verifier + VCS-tracked promotion), agent **autonomy** (durable cron + repo ingestion), real **reasoning search** primitives (LATS + AFlow), and an **entity/relation graph** substrate. All eight ship **default-off / opt-in** (zero behavior change unless toggled) and are **provider-agnostic** (validatable live against glm-4.7 / deepseek / gemini while anthropic is under quota).

| Item | Change | Knob(s) |
|---|---|---|
| **H1** | `hypothesis` + `z3-solver` added to the dynamic-code allowlist and the materializer namespace, so sandboxed generated code (and the G1 termination check) can use a real proof-checker. No behavior change — only newly-allowed imports | — (allowlist) |
| **G2** | **VCS-tracked, locally-gated prompt promotion** — a promoted prompt writes a versioned, *tracked* `prompts/evolved/<node>.<sha>.json` (not a gitignored scratch artifact) AND is gated by a local pre-promotion check (node eval/canary) before the live `current.json` pointer flips. A failed gate ⇒ no pointer change, no commit (a regression stays un-promoted) | `EVOLUTION_PROMOTE_TO_LIVE` |
| **G1** | **Graph-invariant verifier (stage-1)** — `src/evolution/invariants.py` runs 5 checks on CODE mutations: `compiles` (AST), `imports_clean` (a subprocess imports `src/graph/{state,routers,task_graph}` in isolation — catches import-time breakage `compile()` misses, without clobbering the live `sys.modules`), `state_schema_compatible` (AgentState superset of the live baseline), `routers_valid` (router return-literals ⊆ registered nodes + END), and `no_self_loops` (no `add_edge(X,X)`). **Termination/budget are runtime-enforced** (`effective_max_iterations` + the budget hard-cap live in `route_after_*`), not static invariants — a sound static proof is deferred; the runtime sandbox smoke (`post_deploy_verify`) is the dynamic backstop. Always-on for CODE mutations; hooked between `deploy()` and `post_deploy_verify()`; invariant-fail ⇒ existing `rollback_deployment()` | always-on (no knob) |
| **I1** | **Agent-settable durable cron** — `scheduled_tasks` table (Alembic) + `create_scheduled_task` tool (cron validated via `CronTrigger`, capped count) + a scheduler consumer that registers/refreshes APScheduler jobs and enqueues a run via the existing `RunsQueue` at each tick (lease/claim handled by the worker). Resume-safe + idempotent | `AGENT_CRON_ENABLED`, `AGENT_CRON_MAX_TASKS` |
| **I2** | **git-clone builtin + pgvector code index** — `git_clone` SSRF-guards the URL (`assert_public_host`), clones into a confined workspace, walks code files, chunks per function/class (AST chunker), embeds each via `EmbeddingGenerator`, and stores as `ColdMemory` (`memory_type="code"`); `code_search` reuses `cold.search_by_query` for semantic recall. Size/path caps enforced | `GIT_CLONE_ENABLED` |
| **G3a** | **LATS/MCTS tree-search execution primitive** — a real per-call MCTS tree (`lats_search` node): expand N candidate next-steps, gateway-only rollouts, an LLM value function, UCB1 select + backprop, commit the UCB-best root child as the chosen next step for **single-trajectory** execution (no side-effect fan-out). Engages only on CRITICAL + low-confidence retries (`scope="stall"`); stateless per call; a fail-safe returns the plan's original step unchanged | `LATS_ENABLED`, `LATS_SCOPE`, `LATS_MAX_EXPANSIONS`, `LATS_ROLLOUT_DEPTH`, `LATS_MAX_EVALUATIONS`, `LATS_MAX_DEPTH`, `LATS_EXPLORATION` |
| **G3b** | **AFlow/ADAS workflow-topology optimization** — search over the *technique-policy* (which prompting techniques wire into each node, per goal-category — a distinct axis from the prompt-text optimizer). An offline, DI, cost-bounded optimizer (`AFlowOptimizer`) proposes candidate policies, evaluates each against real `execute_run` fitness, and persists a winner only if it beats the baseline by the margin (pre-flighted by the C1 capability curve). A runtime hook (`aflow_techniques_for`) overrides `TechniqueSelector` when on; off ⇒ byte-identical. `python main.py --aflow`; optional scheduler job | `AFLOW_ENABLED`, `AFLOW_TARGET_NODES`, `AFLOW_MAX_CANDIDATES`, `AFLOW_IMPROVEMENT_MARGIN`, `AFLOW_MAX_COST_USD` |
| **I3** | **Neo4j entity/relation graph** — an additive structured-mirror store (`src/memory/graph.py` `Neo4jGraph`): when on, the memory write hooks mirror skills/procedures/workflows (`:Skill`/`:Procedure`/`:Workflow` + `:DEPENDS_ON`), facts (`:Fact :ABOUT :Entity`), and sub-agent defs (`:SubAgent` + tier + tools) into Neo4j nodes/edges. Default-off, lazy driver, never-raises (CostTracker-resilience pattern); profile-gated `neo4j` compose service. Pure structured sync — no LLM extraction | `GRAPH_ENABLED`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` |

All new knobs land in `.env.example`. **HARD GATE:** these are baked into the agent image — mirror the live config, **rebuild the agent image and recreate api/worker** before toggling any on; bring up `--profile graph` for the Neo4j live smoke.

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

**Metric-driven prompt optimizer (DSPy + GEPA sidecar, Phase 2 C2)** — the *improvement* loop the capability curve (measurement) left out. The engine's own PROMPT mutations are one-shot LLM rewrites; the optimizer instead turns the golden canary into an objective and *searches* for a better prompt for a node. It runs in its own container (`src/optimizer`, `Dockerfile.optimizer`, image `self-evolving-agent-optimizer`) so the ML deps (DSPy + GEPA, **no torch**) stay out of the slim api/worker images. Forced by GEPA's real API: GEPA searches candidate instructions against a CHEAP proxy metric over a DSPy student module (bounds cost — each metric call is one cheap LLM call), then the optimized instruction is VALIDATED against the REAL golden canary (full agent runs); if it beats the baseline by the configured margin it promotes through the existing `PromotionGate` (canary final-gate + versioned write + auto-rollback — the eval metric stays the promotion gate). **Safety:** it refuses to run while C1's capability curve is regressed or inconclusive (`OPTIMIZER_REQUIRE_CURVE_CLEAR`), spend is hard-capped at `OPTIMIZER_MAX_COST_USD` (queried before compile AND before promote), and it runs only as a nightly scheduler job (NOT per-run). The scheduler POSTs an empty body; the sidecar resolves node/backend/eval from its own `OptimizerSettings`. Everything defaults off; only `classify` ships a profile in v1 (execute/verify are a `ConfigurationError`, not a stub); `textgrad` (torch) is a deferred backend. **It reuses `eval_results` + `cost_ledger` + `.turing/evolved/prompts/` — no DB migration.** Bring-up: `docker compose --profile optimizer --profile scheduler up -d` (both profiles — the scheduler triggers the sidecar); build with `docker compose --profile optimizer build optimizer`.

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
| `LLM_LITELLM_NUM_RETRIES` | `0` | litellm internal retry count passed on every `litellm.acompletion` call (`0` ⇒ **tenacity is the single retry authority**). Was previously unset, so litellm defaulted to **3 internal retries layered under tenacity's `stop_after_attempt(3)`** → up to 9 HTTP hits per logical call (glm-4.7's 4-model chain → up to 36). That amplifier was the root cause of the `litellm.Timeout "time taken=252s"` → deepseek cascade once miscast as "Z.AI degradation" (252s = 3 retries × 90s `request_timeout`). The fast httpx/openai *transport*-level retry on a transient 429/503 is unaffected and kept by design (sub-second self-heal beats dropping to a fallback model on every hiccup) |
| `LLM_MAX_OUTPUT_CAP` | `16384` | Ceiling on the `max_tokens` reserved per call. Was unset → 128K (the model max) reserved on every node/codegen call, blowing provider TPM windows |
| `RATE_LIMIT_CROSS_PROCESS_ENABLED` | `true` | Coordinate the WHOLE worker fleet against ONE shared provider RPM/TPM quota via a Redis atomic token budget (Lua `GET-check-INCR` on minute-scoped keys `turing:ratelimit:{rpm\|tpm}:{provider}:{minute}`). Without it, N worker processes each apply the full per-process rate against the single quota. Degrades to in-memory-only (`aiolimiter`) when off or when no Redis client is attached. Rate limiting is observability-only — the provider's own 429 + retry/circuit-breaker stack is the hard backstop |
| `RATE_LIMIT_MAX_WAIT_ATTEMPTS` | `5` | Bounded backoff attempts when a shared rate window is full (0.1→2.0s exponential, yielding the event loop so in-flight calls drain); after exhaustion the call proceeds best-effort |
| `GOVERNANCE_PRUNE_ENABLED` | `false` | Register a periodic capability-governance prune job on the scheduler — re-runs `consolidate.py` retire/redundancy + the tool cap-enforce so a long-lived worker frees cap headroom between restarts |
| `GOVERNANCE_PRUNE_CRON` | `0 4 * * *` | 5-field crontab for the prune (default 04:00 UTC, clear of the 05:00 curve-gate + 02:00 battery) |
| `GOVERNANCE_PRUNE_TIMEZONE` | `UTC` | IANA zone for the prune cron |
| `OPTIMIZER_ENABLED` | `false` | Register the nightly metric-driven prompt-optimizer trigger on the scheduler (POSTs `/optimize` to the optimizer sidecar) |
| `OPTIMIZER_BACKEND` | `dspy-gepa` | DSPy teleprompter: `dspy-gepa` (reflective) / `dspy-mipro` / `dspy-copro`; `textgrad` (torch) is deferred |
| `OPTIMIZER_TARGET_NODE` | `classify` | Graph node whose system prompt is optimized (only `classify` ships a profile in v1) |
| `OPTIMIZER_EVAL_SPEC_LIMIT` | `2` | How many golden specs the FINAL canary validates against (cheapest = 1; GEPA's proxy loop is bounded separately) |
| `OPTIMIZER_MAX_CANDIDATES` | `8` | MIPROv2/COPRO candidate-search breadth (unused by GEPA) |
| `OPTIMIZER_MAX_TRIALS` | `1` | GEPA full-eval rounds (`max_full_evals`; 0 lets GEPA pick via `auto="light"`) |
| `OPTIMIZER_MAX_TOKENS` | `1024` | DSPy student + reflection LM call max output tokens |
| `OPTIMIZER_TEMPERATURE` | `0.7` | LM sampling temperature (0–2, validated) |
| `OPTIMIZER_CRON` | `30 3 * * *` | Nightly trigger (default 03:30 UTC — between the 02:00 battery and the 05:00 curve-gate) |
| `OPTIMIZER_TIMEZONE` | `UTC` | IANA zone for the cron |
| `OPTIMIZER_URL` | `http://optimizer:8095` | Scheduler → sidecar connect URL (compose DNS; the bind is `OPTIMIZER_HOST`/`OPTIMIZER_PORT`, internal-only) |
| `OPTIMIZER_REQUIRE_CURVE_CLEAR` | `true` | Refuse to optimize while the capability curve is regressed OR inconclusive (false overrides on a cold curve, accepting the proxy→canary transfer risk) |
| `OPTIMIZER_CANARY_MIN_SCORE` | `None` | Min canary MARGIN (candidate−baseline) to promote; `None` reuses `EVAL_CANARY_MIN_SCORE` as the absolute floor |
| `OPTIMIZER_MAX_COST_USD` | `0.50` | Hard spend cap for one optimization run (queried before compile + before promote); spend lands in `cost_ledger` under `run_id=optimizer-<node>-<ts>` |
| `ROUTING_NODE_TIER_OVERRIDES_JSON` | `{}` | Operator JSON override pinning a specific model onto a `(complexity:node)` route, e.g. `{"COMPLEX:execute": "glm-4.7"}` — merged on top of the curated `NODE_TIER_MAP` at `route()` call-time (F2). Empty/unparseable ⇒ unchanged |
| `ROUTING_COMPLEXITY_TIER_OVERRIDES_JSON` | `{}` | Operator JSON override for a whole complexity tier, e.g. `{"COMPLEX": "glm-4.7"}` — merged on top of `COMPLEXITY_TIER_MAP` (F2). A node-specific override wins; empty/unparseable ⇒ unchanged |
| `TOOL_RETRIEVAL_BLEND_SUCCESS` | `false` | Tool retrieval re-ranks the top-N-by-cosine pool by `cosine · (base + weight·success_rate·(1−empty_output_rate))` (E2). Off ⇒ pure-cosine ranking; on ⇒ a reliable near-match beats a flaky closer one. Any error ⇒ full-set fallback |
| `AGENT_SELECTION_ENABLED` | `false` | Before the delegate fan-out, rank the already-spawned sub-agents against the subtask embedding and run only the top `AGENT_SELECTION_TOP_K` (F1). Off ⇒ all spawned agents run; fail-safe ⇒ all-spawn |
| `AGENT_SELECTION_TOP_K` | `3` | How many of the spawned sub-agents actually run when `AGENT_SELECTION_ENABLED` (F1) is on |
| `STRUCTURE_ANALYSIS_LLM_ASSIST_ENABLED` | `false` | On a COMPLEX/CRITICAL goal where the deterministic regex pass found no gaps, make a one-shot gateway call (glm-4.7) to refine `{tool_gaps, agent_gaps}` (E3). Single-shot (guarded by `structure_analysis_done`); any LLM/parse error ⇒ regex-only |
| `VISION_ENABLED` | `false` | Allow `LLMGateway.acompletion(images=...)` — images fold into the last user message as text + `image_url` blocks and the fallback chain is restricted to image-capable models (`ModelSpec.supports_images`) (D2). Off ⇒ byte-identical text-only; the chain is kept unchanged if no image-capable model is configured |

---

## Known Limitations & Validation Status

**Validated end-to-end on default routing (2026-06-23, `showcase-vector-db-2`):** a single complex research→analysis→report query exercised the full feature set live — runtime tool creation + reuse (`vector_db_comparator` created, registered, executed), web search (62 SearXNG queries, 5 sources cited), sub-agent delegation, the prompting-technique selector (`step_back` / `generated_knowledge` / `chain-of-thought` / `self_ask` / `first_principles` / `checklist`), 3 autonomous memory folds, and **a live evolution cycle** (a PROMPT mutation deployed to `src/prompts/system_prompt.md`). The primary deliverable was written and verified present. Total ≈ $0.22.

**Resolved — per-query log sink leak (#313, commit `30b5cf6`, local-only).** The "post-success non-termination + goal drift" once attributed to this run was a **misdiagnosis**: the run actually **succeeded and terminated correctly** on its objective (evolution fired ~03:05; the immutable `submitted_goal` anchor kept `verify` on the vector-db objective throughout). The symptom — unrelated q01 `Goal:` lines appearing in the showcase log — was caused by `add_query_log_sink` adding a process-global loguru sink and discarding the returned handler id with no teardown, so the lingering sink captured the *next* run's logs (a separate `bench-battery04_q01-*` benchmark merged into the showcase log). **Fix:** `add_query_log_sink` now returns the handler id and `remove_query_log_sink` (None-safe) tears it down in `execute_run`'s `finally` block and in `_run_single_query`. Short/converging runs always terminated normally.

**Anthropic temporarily disabled** from routing (account usage cap; revert by 2026-07-01) — see the note under *Provider Support* above.

**Run-control hardening — known caveat (budget hard-stop resume).** When `BUDGET_HARD_STOP=true` and a run exhausts its per-run token cap, the gateway raises `BudgetExhaustedError` and the worker marks the run `BUDGET_EXHAUSTED` (acked, terminal, checkpoint persisted). The run is resumable via `--resume <run-id>`, BUT token usage is cumulative across the run_id, so a resumed run re-trips the cap immediately — a resume-window delta (baseline the spent count at resume) is a deferred follow-up. Until then, prefer re-issuing the goal rather than resuming a budget-exhausted run.

**Resolved — provider-call amplification (commits local-only, 2026-06-26).** The recurring glm-4.7 `litellm.Timeout "time taken=252.55s"` → deepseek-v4-flash cascade — long treated as "Z.AI degradation" — was **self-inflicted by four codebase mechanisms**, all fixed: (1) `num_retries` was never passed to litellm, so each logical call fired up to 9 HTTP hits (3 internal litellm retries × 3 tenacity attempts; a 4-model chain → up to 36) — the 252s timeout was 3 retries × 90s; now `LLM_LITELLM_NUM_RETRIES=0` makes tenacity the single retry authority. (2) no call site passed `max_tokens`, so 128K was reserved per call, blowing TPM windows; now capped by `LLM_MAX_OUTPUT_CAP=16384`. (3) the rate limiter paced only the original provider on input tokens, leaving fallbacks unpaced; now `_retry_call` acquires per-attempt on the actual fallback provider reserving input + capped output. (4) the worker fleet had no cross-process coordination; now a two-layer `RateLimiterRegistry` (in-memory `aiolimiter` floor + a Redis atomic token budget) coordinates all workers against one shared quota. **Live-validated 2026-06-26** (run `complex-arxiv-zaifix-1`, 26 LLM calls): zero `Timeout`/`RateLimitError`/`RetryError`, glm-4.7 succeeded on the reasoning nodes without degrading, and the shared rate keys stayed sane (`tpm:zai≈17K` ≪ the 100K limit). (Note: deepseek-v4-flash handling the CHEAP execute-tier steps is *intended* per-node routing from `NODE_TIER_MAP`, not the old flat-routing degradation.)

---

## Architecture Documentation

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full architectural narrative covering three generations of the evolutionary agent, detailed Mermaid diagrams for tool creation and sub-agent spawning pipelines, key findings from testing, and design decisions.

## License

This project is licensed under the **PolyForm Noncommercial License 1.0.0**. 

**Non-commercial use is permitted.** Any commercial use (including integration into proprietary products, SaaS, or paid services) requires explicit written permission from the copyright holder. 

For commercial licensing inquiries, please contact: agarwalamit081@gmail.com