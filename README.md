# Turing Agent — Self-Evolving AI Agent

A production-grade, self-evolving AI agent built with **LangGraph** that continuously
improves its reasoning, tooling, memory, and workflow through autonomous mutation,
A/B testing, and skill crystallization. All LLM calls flow through **litellm** (10+
providers) with cost-aware, complexity-based routing; **PostgreSQL + pgvector** is the
sole database; **Redis** holds hot memory and caches; and every knob is a
**pydantic-settings** env var. For the full architectural narrative, see
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md); per-subsystem depth lives in
[`docs/design-docs/`](docs/design-docs/).

---

## Quick Start

### 1. Start the stack

```bash
# Full stack (api/worker/runner + PostgreSQL 18 + pgvector + Redis 7 + SearXNG + Meilisearch)
docker compose up -d

# Run database migrations
alembic upgrade head
```

The compose stack maps **non-default host ports** to avoid clashing with a host-local
Postgres/Redis other apps may own on the canonic ports:

| Service   | Image                     | Host port | Container port |
|-----------|---------------------------|-----------|----------------|
| Postgres  | `pgvector/pgvector:pg18`  | **5433**  | 5432           |
| Redis     | `redis:7-alpine`          | **6380**  | 6379           |

> **Never `docker compose down -v`** — the `-v` flag deletes the `pgdata` / `redisdata`
> named volumes (warm memories, cost ledger, evolution history). Use plain
> `docker compose down` to stop while preserving data; `docker compose up -d` resumes on
> the same volumes. Every run logs the resolved DB on connect (e.g.
> `Database engine created → localhost:5433/turing_agent`) — confirm it points at the
> container (`self-evolving-agent-postgres-1` / `turing_agent`), not a throwaway host DB.

### 2. Configure

```bash
cp .env.example .env
# Edit .env — at minimum set:
#   DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5433/turing_agent
#   REDIS_URL=redis://localhost:6380/0
#   OPENAI_API_KEY=...   (or ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, ZAI_API_KEY, ...)
```

### 3. Run (host alternative to the container stack)

```bash
# Activate the project virtualenv. Do NOT use `uv run` (blocked by an upstream vllm dep).
source /home/amiagarw/aiml01/bin/activate

python main.py --goal "Research the latest developments in LangGraph"
```

Use **provider-specific** keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`,
`ZAI_API_KEY`, …) — there is no generic `LLM_API_KEY`.

---

## Key Commands

| Flag | Purpose |
|------|---------|
| `--goal "…"` / `-g` | Goal for the agent to accomplish |
| `--interactive` / `-i` | Prompt for the goal at runtime |
| `--provider <p>` / `-p` | LLM provider (e.g. `deepseek`, `zai`, `anthropic`) |
| `--model <m>` / `-m` | Specific model (pin a whole run; routing is otherwise complexity-based) |
| `--no-evolution` | Skip the self-evolution phase |
| `--max-iterations N` | Override the runtime iteration cap (else complexity-derived) |
| `--verbose` / `-v` | Debug logging |
| `--run-id <id>` | Unique run identifier (per-query logging + per-run results subfolder) |
| `--clean` | Clear this run's results subfolder first (requires `--run-id`) |
| `--results-dir <path>` | Override `results_root` for this run |
| `--stream` | Stream the final answer token-by-token to stdout |
| `--resume <id>` | Continue a killed/interrupted run from its last Postgres checkpoint |
| `--eval` | Run the Battery-04 golden suite (full agent run + correctness scoring per spec) |
| `--capability-curve` | Print the nightly capability curve + regression verdict (read-only). `--since`/`--until`/`--export`/`--plot` refine it |
| `--retrieval-eval` | Measure memory recall quality (precision@k + MRR). `--retrieval-k` sets top-k |
| `--aflow` | Offline AFlow workflow-topology optimizer (per-category technique-policy search) |

Example: `python main.py --goal "…" --run-id my-task --clean`

---

## Architecture

```
START → classify → plan → retrieve_memory → [disambiguate? → structure_analysis]
  → execute ↔ reflect → agent_spawn? → delegate → tool_create? → plan
  → verify → evolve? → store_memory → hitl? → END
```

**6 layers:** Presentation (`api/`, `main.py`) · Orchestration (`graph/`) · Agent
(`llm/gateway.py`, `llm/model_router.py`) · Evolution (`evolution/`) · Capability
(`tools/`, `memory/`) · Infrastructure (`llm/`, `db/`, `observability/`). See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for Mermaid diagrams and the design
rationale.

### Subsystems (design-doc deep-dives)

| Subsystem | Design doc |
|-----------|------------|
| Model selection & cost tiers | [`02-model-selection.md`](docs/design-docs/02-model-selection.md) |
| LLM gateway (litellm, circuit breaker, rate limiter) | [`13-llm-gateway.md`](docs/design-docs/13-llm-gateway.md) |
| Workflow / StateGraph nodes | [`06-workflow-design.md`](docs/design-docs/06-workflow-design.md) |
| 3-tier memory (Hot/Warm/Cold + folding) | [`08-memory-system.md`](docs/design-docs/08-memory-system.md) |
| Tools (22 built-ins + runtime generation) | [`14-tool-system.md`](docs/design-docs/14-tool-system.md) |
| Sub-agent delegation | [`18-sub-agent-system.md`](docs/design-docs/18-sub-agent-system.md) |
| Self-evolution engine | [`07-self-evolution-engine.md`](docs/design-docs/07-self-evolution-engine.md) |
| Safety guardrails (7-layer) | [`10-safety-guardrails.md`](docs/design-docs/10-safety-guardrails.md) |
| Evaluation & benchmark | [`20-evaluation-benchmark.md`](docs/design-docs/20-evaluation-benchmark.md) |
| Deployment (role-split containers) | [`11-deployment.md`](docs/design-docs/11-deployment.md) |
| Configuration reference | [`03-environment-config.md`](docs/design-docs/03-environment-config.md) |
| State schema | [`12-state-schema.md`](docs/design-docs/12-state-schema.md) |
| Database schema (PostgreSQL + pgvector) | [`04-database-schema.md`](docs/design-docs/04-database-schema.md) |
| Error handling & resilience | [`16-error-handling.md`](docs/design-docs/16-error-handling.md) |

> The 20 docs in `docs/design-docs/` are the **authoritative specification** for this
> project. When code and docs conflict, the docs are correct.

---

## Testing

```bash
# Unit + integration (no API key needed)
python -m pytest tests/ -v -k "not e2e"

# Full suite including E2E (requires a provider API key)
OPENAI_API_KEY=sk-... python -m pytest tests/ -v
```

Tests mirror `src/`; LLM calls are mocked via `litellm.acompletion` for determinism. E2E
tests (`@pytest.mark.e2e`) make real provider calls. Use the `aiml01` virtualenv, not
`uv run`.

---

## Configuration

All configuration is loaded via **pydantic-settings** `BaseSettings` classes from `.env`
(or environment variables) — no `os.environ` in application code. The complete template is
in [`.env.example`](.env.example); every settings class and validator is documented in
[`docs/design-docs/03-environment-config.md`](docs/design-docs/03-environment-config.md).

---

## License

This project is licensed under the **PolyForm Noncommercial License 1.0.0**.
**Non-commercial use is permitted.** Any commercial use (including integration into
proprietary products, SaaS, or paid services) requires explicit written permission from
the copyright holder. For commercial licensing inquiries, contact:
agarwalamit081@gmail.com
