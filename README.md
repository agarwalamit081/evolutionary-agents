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
| `--backfill-embeddings [--table capability\|cold\|all] [--concurrency N] [--dry-run]` | Backfill NULL embedding columns from the live DB (the data half of the capability/cold-memory migrations). `--dry-run` reports the NULL count without calling the embedding API |
| `--score-spec <id> --deliverables <p1> [<p2> …]` | Score on-disk deliverables against a golden `GoalSpec` via the same recomputation engine `--eval` uses. Exit 0=pass, 1=fail, 2=unknown spec |
| `--verify-models [name …]` | Smoke each model over the gateway's real routing; prints pass/fail + tokens, NO secrets. Exit 0=all healthy, 1=any failed |
| `--cost [--cost-run-id X] [--cost-model M] [--cost-by-model] [--cost-today\|--cost-since DATE]` | Read-only spend breakdown from `cost_ledger` (`CostTracker` is write-only). `--cost-today`/`--cost-since` are mutually exclusive |
| `--inspect-mutation [--mutation-id ID] [--inspect-full]` | Print a stored evolution mutation row + whether the promotion gate will parse it (JSON-vs-free-text shape). Latest row if no id; short-prefix or full-UUID match |

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
| Tools (23 built-ins + runtime generation) | [`14-tool-system.md`](docs/design-docs/14-tool-system.md) |
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

### Local quality gate

```bash
scripts/gate.sh                # ruff + pyright + pytest (full pre-merge gate)
scripts/gate.sh --no-pytest    # ruff + pyright only
```

A `.git/hooks/pre-commit` hook (local, not committed) runs `ruff check . && pyright src/`
on every commit — the deterministic, always-green subset (the full pytest suite is gated
manually via `scripts/gate.sh` because ~12 host-env-driven failures would make a commit
gate flaky). Bypass with `git commit --no-verify` for WIP.

---

## Configuration

All configuration is loaded via **pydantic-settings** `BaseSettings` classes from `.env`
(or environment variables) — no `os.environ` in application code. The complete template is
in [`.env.example`](.env.example); every settings class and validator is documented in
[`docs/design-docs/03-environment-config.md`](docs/design-docs/03-environment-config.md).

### Opt-in capability flags

These ship with the shown defaults so production behavior is unchanged until toggled. Most
are **off by default**; the few on-by-default are marked. Set them in `.env`.

| Flag | Default | Purpose |
|------|---------|---------|
| `TOOL_SUCCESS_CONTRACT_ENABLED` | `true` | Record a tool's **real** success (non-empty output, no `ERROR:`-style prefix) in metrics + governance retirement, instead of "didn't raise" (#11). |
| `LLM_CACHE_TOKEN_METRICS_ENABLED` | `true` | Export LLM prompt-cache read/creation token Prometheus counters (#13). |
| `MID_RUN_CAP_ENFORCE_ENABLED` + `MID_RUN_CAP_ENFORCE_INTERVAL` | `false` / `10` | Re-run the nightly capability cap-prune mid-run, after a creation round, so a long-lived worker frees persisted tool/sub-agent headroom instead of saturating mid-run (#4). |
| `EVOLUTION_PROMOTE_CODE_TO_CORE` | `false` | Promote a shadow-verified CODE mutation toward core-`src/` (candidate path only; never live until set) (#8). |
| `EXPERIMENTAL_TECHNIQUES_ENABLED` + 5 `EXPERIMENTAL_TECHNIQUE_*_ENABLED` | `false` | Opt into experimental prompting techniques (self-debugging, Godel-agent, Web-Dreamer, Absolute-Zero, adversarial-debate) behind a master + per-technique gate (#18). |
| `LEAN4_ENABLED` + `LEAN4_TIMEOUT_S` | `false` / `120` | Lean 4 formal-verification builtin (auto-detects `lake`/`elan`); bounded `lean` type-check in a confined dir (#17). |
| `RESEARCH_LOOP_ENABLED` + `RESEARCH_MAX_HOPS` | `false` / `3` | **Phase 5a** — multi-hop research loop node wired as a conditional edge `retrieve_memory → research? → plan`: per hop it queries (`web_search`/`arxiv_search`/`corpus`) → reads top-K → an LLM refine-or-stop decision → accumulates grounding into `state.research_context`. Bounded by `RESEARCH_MAX_HOPS` and a per-hop token budget. Off ⇒ graph topology is byte-identical to the default (single-pass `retrieve_memory`). |
| `VISION_ENABLED` | `false` | **Phase 5c** — image/vision extraction. A goal carrying `images=[...]` (e.g. the `vision_extract` golden goal over `tests/fixtures/vision_sample.png`, which is *not* in the nightly battery) routes through a vision-capable model: the gateway restricts the fallback chain to `ModelSpec.supports_images` and builds text+image content blocks. Accepts `http(s)://` URLs or `data:image/...;base64,` data-URIs (local paths are best-effort converted). |
| `GRAPH_ENABLED` + `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | `false` | **Phase 5b** — Neo4j entity/relation graph mirror (`src/memory/graph.py`, lazy driver, never raises). `MemoryManager` write hooks mirror structured records (skills/procedures/workflows, facts, sub-agent defs) as nodes/edges (`DEPENDS_ON`, etc.) so relationship queries ("which skills depend on X?") the relational + pgvector stores can't express become a Cypher `MATCH`. Compose service `neo4j` lives under the `graph` profile. Apache AGE (Cypher-over-Postgres, no extra service) is deferred — Neo4j satisfies the graph tier. |

Operational scripts added this sweep: `scripts/cve_sweep.py` (non-fatal `pip-audit` CVE sweep
over `requirements*.txt`, JSON to `logs/`, #19) and `scripts/analyze_vector_queries.py`
(read-only `EXPLAIN ANALYZE` over the HNSW-backed tables, #20).

### Cost & convergence knobs (Phase 3.5 Cluster A + B1)

Tuning knobs for token/cost waste and verify-loop non-convergence. All ship with the shown
defaults; production behavior is unchanged until set in `.env`.

| Knob | Default | Purpose |
|------|---------|---------|
| `LLM_REQUEST_TOTAL_TIMEOUT` | `300` | **B1** — master wall-clock deadline wrapping the *whole* LLM fallback chain, so a slow chain can't amplify to `N×attempt_timeout`. `asyncio.TimeoutError` is excluded from tenacity's retriable set, so timeouts are never retried. |
| `LLM_VERIFY_MAX_TOKENS` / `LLM_CLASSIFY_MAX_TOKENS` / `LLM_PLAN_MAX_TOKENS` / `LLM_REFLECT_MAX_TOKENS` | `512` / `256` / `2048` / `1024` | **A3** — per-node output-token caps (execute keeps the model default). Verify pass/fail needs no 4K prose; capping verify at 512 cuts ~180K tokens/run with no quality loss. |
| `VERIFY_MAX_CYCLES` | `12` | **A4** — verify-pass hard cap; terminates via `store_memory` (best-so-far) independent of the global iteration cap. |
| `VERIFY_OSCILLATION_REPEAT` | `3` | **A4** — if the blocking-failure fingerprint is identical for N consecutive verify passes, abort early (stuck on one blocker). |
| `CONVERGENCE_STABLE_THRESHOLD` | `3` | **B3** — if the verify output fingerprint is stable for N passes *and* the plan is exhausted, accept the partial result. |
| `PER_RUN_TOKEN_LIMIT_ABSOLUTE` | `0` (off) | **A5** — cumulative token ceiling *across all attempts* (no baseline subtraction). Mandatory backstop for free/$0 tiers (a $0-model runaway never trips the USD cap). 0 = disabled. |

### Benchmark curves

The battery-04 suite (9 goals, suffix-isolated run_ids) is scored by the eval harness
(`python main.py --eval`) and trended via `src/eval/curve.py`. Score = mean of all eval rows
in each goal's latest attempt, then mean-of-per-goal-means. Results live in `logs/curve_*.json`:

| Curve | Suffix | Battery mean | Ex-q06 | Change |
|-------|--------|--------------|--------|--------|
| #1 | 20260701 | 0.7735 | 0.7785 | baseline |
| #2 (per-step routing) | 20260702 | 0.7362 | 0.7977 | −0.037 (neutral) |
| #3 (Cluster A + B1) | 20260704 | 0.7663 | 0.8482 | +0.030 (neutral recovery; ex-q06 best of three) |
| #4 (Cluster B latency rebalance + A2 cache recorder) | 20260705 | 0.7038 | 0.7446 | −0.063 (neutral; lowest of four) |

The reliability/cost stack is now proven across four consecutive neutral curves (q06
redelivery bounded 40% by the A5 token cap; timeout amplification killed by B1; budget/timeout
backstops fired cleanly — q06 `budget_exhausted` @ $0.985, q08 `timeout` @ 2h). Curve #4 is the
lowest but is **run-to-run variance, not a regression**: goals perfect in #3 (q02/q07/q09=1.0)
dropped, while low-in-#3 goals (q05 0.525→0.863, q06 0.111→0.378) improved. Cluster B was a
speed/cost cycle — its latency-bomb removal (mistral-medium-3-5 946s, nvidia-deepseek-v4-pro
164s) was a **no-op for this battery** (only 3 MODERATE-tier primaries touched; the bombs live
in fallback tails never reached), but valid as forward-looking defensive hardening. The clear
cost win is **A2**: provider prompt-cache hits are now recorded in `cost_ledger.cached_tokens`
(2,107,520 tokens across 428/624 rows in this battery — previously invisible). The self-improvement
pillar is now **qualified-proven** via the generation-over-generation curve below — the last
"unproven" label is retired.

### Generation-over-generation curve (self-improvement)

The config-arm curves above (#1–#4) compare *feature flags* on a clean slate each time; they
cannot show whether a prior generation's crystallized state improves a later one. The G0→G1→G2
generation curve does — each generation **inherits** the prior's tools/skills/facts/cold-memory
+ evolved prompts (no `clean_state` between), same stack (`glm-5.1`, `PER_RUN_COST_LIMIT=1.2`,
Anthropic off, scheduler stopped). Channel-B (prompt promotion) is OFF in G0, ON in G1/G2.
Scored by `scripts/run_metrics.py` + `scripts/generation_compare.py` (terminal-state, suffix-scoped):

| Gen | Suffix | Battery mean | Cost $ | tok_in | span_s |
|-----|--------|--------------|--------|--------|--------|
| G0 | gen0-20260712 | 0.9424 | 4.22 | 5.12M | 4395 |
| G1 | gen1-20260713 | 0.8086 | 4.32 | 5.29M | 5561 |
| G2 | gen2-20260713 | 0.9333 | 3.44 | 4.80M | 4073 |

G1 regressed (−0.134: G0-passer q04 collapsed to 0.167 and G0-failer q06 worsened to 0.111),
then **G2 recovered** (+0.125) back to G0 (G0→G2 −0.009, within run-to-run variance on score).
**G2 is the most efficient generation** — cheapest (−18% vs G0), fewest tokens (−6%), fastest
(−7%) despite inheriting a *regressed* G1. Two real self-improvement signals: (a) efficiency
gen-over-gen, and (b) the q06 failer **beat its G0 baseline** (0.567→0.111→**0.733**) — inherited
error-episodes + tools bought more progress per dollar on the same $1.2 cap. Channel-B fired 8
more live promotions in G2 (history 7→15, all canary 1.0). Not a clean score-up-over-baseline
PROVEN — q04 only partially healed (1.0→0.167→0.667) because the single-goal promotion canary
can't gate multi-goal quality; that canary-coverage gap is the next lever. Verdict:
**RECOVERY + efficiency-PROVEN, qualified-yes** — a prior generation's state improves a later
one on efficiency and on at least one failer's score, and recovers regressions rather than
compounding them.

#### n=3 multi-seed confirmation (2026-07-14, Track-1 stack)

A separate, stronger test of the same thesis on a different stack: `glm-5.2` primary +
Anthropic **on**, the Track-1 attribution/multi-goal-canary apparatus (`probe_analytics_recall`
+ `probe_multi_orchestration`, `passed=False→0.0` strict gate). Three independent seeds, each
G0 (channel-B off) → G2 (channel-B on, inherits G0's channel-A tools/skills), 8-goal matched
primary (q09's `iqr_outlier_detector` retired):

| Seed | G0 | G2 | Δ |
|------|-------|-------|--------|
| seed-2 | 0.7304 | 1.0000 | +0.270 |
| seed-3 | 0.7471 | 0.9019 | +0.155 |
| seed-1 | 0.9038 | 0.8259 | −0.078 |
| **mean Δ** | | | **+0.116** |

2/3 seeds improved under evolution; seed-1 (the strongest baseline) regressed −0.078.
Root-caused: seed-1's regression is **not** a quality counter-example. q06 `budget_exhausted`
in *both* gens — G0 hit the cumulative token cap @1.52M and produced a partial deliverable
(→0.356); G2 hit the per-attempt cost cap @$1.20 @iter 58 and produced nothing (→0.000),
because its evolution/reflection/re-planning overhead (+24 LLM calls, +74% output tokens,
−42% prompt-cache reuse, heavier model mix) hit the $ ceiling at only 1.21M tokens.
Decomposition of seed-1's −0.078: q06 −0.044 (57%, a budget-infeasible never-converges goal)
+ q01/q05 −0.033 (43%, within run-to-run variance). q06 is the highest-variance goal in the
battery (0.0–0.87 across 22 historical runs; it *improved* 0.078→0.733 under evolution in
seed-3) — non-informative in both directions. The verdict rule requires seed-1 G2>G0 → **fails
→ NOT clean/unanimous PROVEN**. The multi-seed test therefore **confirms** the single-seed
Phase-2 verdict (RECOVERY + qualified-yes) rather than upgrading it: self-improvement is
weakly-positive-qualified across seeds, not cleanly proven. (Per-run `$1.2` / `1.5M`-token
caps held fixed; see `logs/n3_baseline_seed2_seed3.md`.)

#### Phase F — 3-seed self-improvement test on the cleaned image (2026-07-17/19)

The decisive test of the thesis on the **cleaned image** `fef50596860f` (post dead-code /
config-drift sweep, Phases A–E): 3 independent seeds × a full G0→G1→G2 curve, `glm-5.2`
primary + Anthropic-on stack, channel-B OFF→ON→ON, no `clean_state` between gens (each
inherits the prior's tools/skills/facts/cold + evolved prompts), clean only at seed
boundaries. Scored terminal-state, suffix-scoped (`scripts/run_metrics.py` +
`scripts/generation_compare.py`). **Thesis: does G2 ≥ G0?**

| Seed | G0 | G1 | G2 | Δ G0→G2 | read |
|------|--------|--------|--------|---------|------|
| seed-1 | 0.8398 | 0.8398 | 0.8389 | −0.0009 | flat (PASS, within noise) |
| seed-2 | 0.9554 | 0.8344 | 0.8669 | −0.0885 | G1 collapsed, G2 only partly recovered (FAIL) |
| seed-3 | 0.8477 | 0.8775 | 0.9574 | **+0.1097** | monotonic up, zero regressions (PASS — strongest) |
| **mean Δ** | | | | **+0.007** | **2/3 pass** |

**Verdict — qualified-positive, not clean-proven.** 2/3 seeds recover G2≥G0; the tiebreaker
(seed-3) shifted the picture from lean-negative to lean-positive, but seed-2 fails and the
cross-seed variance is high (+0.110 vs −0.089 from the same protocol and frozen image). At
n=3 the paired Wilcoxon is underpowered (p-floor, CI straddling zero) → descriptive, not
confirmatory; n≥5–10 seeds would be needed to call it statistically.

**The robust signal is efficiency, not quality.** Every seed's G2 was cheaper/faster than
its G0 — seed-3: −23% cost, −20% tokens, −166 LLM calls, −26% wall-time; seed-2: −103
calls; seed-1: −9% cost. Crucially this is *not* a cache artifact: the hit-rate *fell* in
seed-2/3 yet cost still dropped, so G2 makes fewer, more effective calls (accumulated
tools/skills doing less redundant work). This self-improvement fires reliably **3/3**.

**The quality signal is dominated by two budget-lottery goals.** 6 of 9 goals (q02/q03/
q05/q07/q09, sometimes q04) are pinned at the 1.0 ceiling and cannot improve; only q01/q06/
q08 have headroom, and they are the three most budget-cap-prone goals. The standout: **q06
is a mirror image** — seed-2's q06 went 0.867→0.356 (−0.511) while seed-3's went
0.356→0.867 (+0.511), the *same two outcomes in opposite generations*. q06 is bimodal
(≈0.87 when it converges before the `$1.2`/attempt cap, ≈0.36 when it doesn't), so seed-3's
"win" and seed-2's "loss" are substantially the same q06 coin-flip read in opposite
directions. Stripping q06 (score the 8 feasible goals) keeps the 2/3 pattern but roughly
halves the magnitude (seed-3 +0.060, seed-2 −0.036, seed-1 −0.001).

**Channel-B (prompt promotion) fires without a quality gradient.** It promoted on every
seed (seed-1: 5→11 execute promotions) and every promotion passed the canary — yet produced
no consistent quality lift. The canary gates on "converges on a proxy goal," never "scores
higher on the real battery," so evolved prompts accumulate without a fitness→quality link.
The evolution machinery is sound; the *fitness function* is the weak link.

**Bottlenecks & resolutions.** (1) The `$1.2`/attempt cap terminates hard goals (q06
especially) before convergence → bimodal, dominates the mean — *do not raise the cap to
rescue q06* (it would confound cost); instead report q06 separately as an apparatus limit
and score the 8 feasible goals. (2) Channel-B needs a task-quality A/B gate so only
fitness-improving prompts adopt (the highest-leverage fix). (3) Verify-loop churn on q01/q08
(re-verify without a score-improvement gate) burns budget and sometimes degrades — this is
#600 (eval-score-gated verify convergence), still deferred. (4) High variance + low n — more
seeds (n≥5–10) and harder, non-ceiling battery goals would give headroom to show gains.
Full analysis: `docs/findings-001.md` §C3 + the `phasef-3seed-verdict` memory; per-seed
detail in `logs/`. Spend ~$54 of a `$60` pool.

---

## License

This project is licensed under the **PolyForm Noncommercial License 1.0.0**.
**Non-commercial use is permitted.** Any commercial use (including integration into
proprietary products, SaaS, or paid services) requires explicit written permission from
the copyright holder. For commercial licensing inquiries, contact:
agarwalamit081@gmail.com
