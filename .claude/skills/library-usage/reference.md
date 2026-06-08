---
description: Deep technical reference for preferred libraries, verification protocol, and anti-patterns across Python and JavaScript/TypeScript projects.
---

# Library Usage Reference

## Python Preferred Libraries

| Category | Use This | NEVER Use |
|---|---|---|
| Logging | `loguru` | `logging` (stdlib) |
| Path Operations | `pathlib.Path` | `os.path` |
| Data Validation | `pydantic` | Manual validation |
| Environment Config | `pydantic-settings` | `os.environ` directly |
| HTTP Client | `httpx` (async) | `urllib` |
| Database ORM | `SQLAlchemy` + `Alembic` | Raw SQL strings |
| Async DB Driver | `asyncpg` | `psycopg2` for async |
| JSON Parsing | `msgspec` (perf), `pydantic` (validation) | Manual JSON |
| JSON Repair | `json-repair` | Custom fixers |
| Linting | `ruff` | `flake8`, `pylint` |
| Type Checking | `mypy`, `pyright` | - |
| Testing | `pytest` + `pytest-asyncio` | `unittest` |
| Document Parsing | `docling`, `pymupdf4llm`, `pdfplumber` | Manual text extraction |
| Web Scraping | `crawl4ai`, `scrapy`, `beautifulsoup4` | - |
| Unstructured Data | `unstructured` | - |
| JSON (dirty) | `dirtyjson` | - |
| Privacy | `presidio` | Manual PII redaction |
| Retry/Resilience | `tenacity`, `circuitbreaker` | Manual retry loops |
| File I/O (async) | `aiofiles` | sync I/O in async code |
| Env Vars | `python-dotenv` | Manual .env parsing |
| Token Counting | `tiktoken` | Char/4 estimate |
| Templating | `jinja2` | f-strings for complex templates |
| Serialization | `msgspec` | `pickle` |
| LLM Framework | `langchain`, `langgraph` | - |
| Checkpointing | `langgraph-checkpoint-postgres` | - |
| Observability | `langsmith` | - |
| Evaluation | `ragas`, `deepeval` | Manual eval |
| MCP Servers | `fastmcp` | - |
| API Framework | `fastapi` + `uvicorn` | Flask for async |
| Vector DB | `pgvector` (via SQLAlchemy) | Raw SQL for vectors |
| Streaming Audio | `livekit`, `elevenlabs` | - |
| E2E Testing | `playwright` | Selenium |

## JavaScript/TypeScript Preferred Libraries

| Category | Use This | NEVER Use |
|---|---|---|
| HTTP Client | `fetch` or `axios` | `xmlhttprequest` |
| State Management | `zustand`, `jotai` | Direct global mutations |
| Validation | `zod` | Manual validation |
| Testing | `vitest`, `jest` | - |
| Linting | `eslint` + `prettier` | - |
| Type Checking | `typescript`, `pyright` | - |
| Bundling/Repo | `repomix` | - |

## Verification Protocol

Follow these 5 steps whenever you call a library API you have not used recently:

1. **Check the import path** — confirm the module name and sub-module path against the installed version. Many libraries reorganize between major versions.
2. **Read the function signature** — open the library docs or source for the exact parameter names, types, and default values. Do not guess.
3. **Confirm async/sync mode** — some libraries expose separate async and sync clients (e.g., `httpx.Client` vs `httpx.AsyncClient`). Match the calling context.
4. **Check deprecation status** — look for deprecation warnings in the changelog. Prefer the newer API even if the old one still works.
5. **Write a smoke test** — add a minimal test that exercises the exact call you plan to make. This catches version mismatches before runtime.

## Anti-Patterns

- **Internal APIs**: Never call `_`-prefixed attributes or methods. They are undocumented and can change without notice.
- **Undocumented behavior**: Do not rely on observed behavior that is not in the official docs (e.g., dict ordering before Python 3.7).
- **Version mixing**: Never install two major versions of the same library in one environment (e.g., `pydantic` v1 and v2).
- **Unpinned dependencies**: Every dependency must have a version constraint. Floating deps cause reproducibility failures.
- **Vendor forks without reason**: Prefer the canonical package over a fork unless the fork is the de-facto standard.

## Cross-Reference

For deep API verification (parameter types, return shapes, edge cases), use the **`check-docs`** skill pattern: fetch the library's official docs page and read the method signature directly rather than relying on memory or inferred names.
