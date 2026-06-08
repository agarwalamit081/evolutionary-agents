# Claude Code Skills

This directory contains reusable, enterprise-grade patterns and utilities.

## Usage Protocol for Claude Code
1. **Always read `SKILL.md`** when a task matches the skill's triggers.
2. **Read `reference.md`** ONLY when you need to verify deep technical constraints, performance rules, or architectural boundaries.
3. **Read `examples.md`** ONLY when generating new code, to ensure output matches established project patterns.
4. **Execute `scripts/*`** to validate, scaffold, or analyze code. Do not read the script source unless debugging the script itself.

## Skill Inventory

### Requirements & Specification
- `requirements-and-specification` — BRD, PRD, FRD, SRS, TRD generation

### Architecture & Design
- `system-architecture` — High-level architecture, distributed systems, scalability, C4 model
- `code-quality-and-patterns` — SOLID, DRY, GoF design patterns

### Frontend
- `frontend-architecture` — Component design, UI/UX, state management, JS/TS patterns
- `fullstack-sync` — Frontend-backend type/API consistency validation

### Backend & Database
- `backend-patterns` — Controllers, services, auth, background jobs, rate limiting, caching
- `backend-and-db-patterns` — Schema design, migrations, PostgreSQL queries and optimization

### Testing & QA
- `testing-and-qa` — Unit, integration, E2E testing, code review checklists
- `playwright-automation` — Playwright E2E testing with POM, mocking, visual regression
- `update-tests` — Systematic workflow for updating tests after code changes

### DevOps & Infrastructure
- `ci-cd-pipelines` — GitHub Actions, GitLab CI, Docker optimization, deployment strategies
- `parallel-feature-development` — Parallel feature dispatch in git worktrees

### AI/LLM
- `pgvector-and-rag-architecture` — RAG pipelines, pgvector, hybrid search, re-ranking
- `structured-outputs-and-guardrails` — JSON schema enforcement, retry logic, hallucination detection
- `llm-observability-and-evals` — Tracing, golden datasets, LLM-as-a-judge, eval pipelines
- `agentic-memory-and-resilience` — Agent memory layers, token management, loop detection
- `langgraph-patterns` — LangGraph state management, HITL, multi-agent patterns
- `agent-tooling` — LLM tool definitions, JSON schema, error handling, idempotency
- `multi-agent-orchestration` — Supervisor-worker, handoffs, conflict resolution
- `prompt-engineering` — System prompts, CoT, few-shot, XML tagging, versioning
- `api-integration` — Streaming (SSE), provider abstraction, rate limits
- `api-cost-optimization` — Token tracking, model routing, caching, budget management
- `context-window-management` — Context window conservation, token-efficient workflows
- `llms-txt` — Fetch and generate llms.txt documentation files for packages

### Code Quality & Operations
- `code-quality-check` — Comprehensive quality checklist (syntax, imports, errors, security, performance)
- `safe-refactor` — Systematic refactoring workflow (find callers → assess → update → validate)
- `import-validator` — Python/TS import hygiene (unused, missing, grouping, circular)
- `circular-dependency-check` — Circular import detection and resolution strategies
- `library-usage` — Preferred library selection guide, dependency verification
- `check-docs` — API documentation verification before using unfamiliar features
- `security-audit` — Comprehensive security vulnerability scanning
- `resource-check` — Resource leak detection (connections, listeners, file handles)

### Python
- `python-patterns` — Asyncio, type hints, dataclasses, decorators, generators

### Browser Extensions
- `chrome-extension` — Manifest V3, React, typed storage/messaging, Shadow DOM

### Utilities
- `summarize-changes` — Git diff summarization, risk flagging, commit message generation
