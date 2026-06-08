# Python Code Standards

## Logging
- ALWAYS use `loguru` for all logging. NEVER use the standard `logging` module.
- Configure `loguru` at application entry with appropriate level, format, and rotation.
- ALL logs MUST be written to the `./logs/` directory. NEVER write logs to `/tmp/`.
- NEVER log or echo sensitive information (API keys, passwords, tokens, PII).

## Path Operations
- ALWAYS prefer `pathlib.Path` over `os.path` for file and directory path operations.
- NEVER hardcode absolute paths. Use relative paths from the project root.

## Data Validation
- ALWAYS use `pydantic` for structured data validation. NEVER write manual validation logic.
- Define explicit Pydantic models for all API request/response schemas, configuration objects, and structured data.

## Database Operations
- ALWAYS use SQLAlchemy ORM. NEVER write raw SQL strings.
- ALWAYS use parameterized queries when raw SQL is absolutely unavoidable.
- ALWAYS generate Alembic migration files when modifying ORM schemas.
- For pgvector setup: `PGPASSWORD=$DB_PASSWORD psql -h localhost -U postgres -d $DB_NAME -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>&1` then verify with `SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';`

## Error Handling
- ALL async operations MUST have `try/except` blocks.
- ALL HTTP client calls MUST handle error status codes.
- ALWAYS check for `None` before accessing attributes. Use `Optional` types with proper unwrapping.
- NEVER swallow exceptions silently. At minimum, log them via `loguru`.
- NEVER leave `except:` without specifying the exception type.

## Code Quality
- NEVER create stubs, placeholders, or dummy implementations.
- NEVER leave unused imports in any file.
- ALL functions must be called somewhere (production code or tests). Dead code must be removed.
- ALL called functions must be defined somewhere in the codebase.
- ALWAYS initialize variables before use.
- ALWAYS verify library version and documentation before using APIs. NEVER invent methods or properties that do not exist.

## File Naming and Organization
- NEVER use curly braces `{}` in folder or file names.
- ALL folder names must be lowercase with hyphens or underscores.
- ALWAYS update `__init__.py` when adding modules to a package.

## Package Management
- ALWAYS pin dependency versions in `pyproject.toml` or `requirements.txt`.
- NEVER use deprecated libraries or APIs without verifying the current recommended alternative.
- Run `ruff check .` for linting and `mypy .` for type checking before committing.

## Package Management & Environment
- ALWAYS use `uv pip` instead of bare `pip`. Use `uv pip show <pkg>` not `pip show <pkg>`.
- ALWAYS use `uv run python` instead of bare `python` for running scripts or tests.
- NEVER use bare `pip install`, `python script.py`, or `python -m pytest` directly. Always prefix with `uv run`.
- NEVER run `pip install` or `uv pip install` automatically. ALWAYS ask the user for confirmation before installing packages. Inform the user of the exact command: `uv pip install <package>`.
- Before using an unfamiliar method on a third-party library, verify it exists: `uv run python .claude/skills/check-docs/scripts/verify_api.py --package <pkg> --method <method>`.

## Environment Verification
- NEVER assume `python` or `python3` points to the project's virtual environment. ALWAYS use `uv run python`.
- To verify the active environment: `uv run python -c "import sys; print(sys.executable)"`.

## Configuration
- ALWAYS use `pydantic-settings` (`BaseSettings`) for all configuration. NEVER use `os.environ` directly in application code.
- Load environment variables through a single `Settings` class at application startup. Pass settings via dependency injection.

## High-Performance JSON
- Use `msgspec` for high-performance JSON parsing and serialization, especially for LLM output processing and large batch operations.
- Use `pydantic` for validation-heavy schemas where JSON Schema generation is needed (API models, config).

## Resilience
- Use `tenacity` for retry with exponential backoff + jitter on all external API calls.
- Use `circuitbreaker` for provider outage protection with configurable failure thresholds.
- Use `backoff` as an alternative retry library when `tenacity` doesn't fit the use case.

## Async Patterns
- Use `asyncio.Semaphore` for concurrency control — NEVER allow unlimited concurrent external API calls.
- Use `asyncio.gather(*tasks, return_exceptions=True)` for parallel operations with graceful error handling.
- Use `aiofiles` for async file I/O. NEVER use synchronous `open()` in async functions.

## Type Hints
- ALWAYS add `from __future__ import annotations` at the top of Python files for modern type hint syntax.
- NEVER use bare `dict`, `list`, or `tuple` without type parameters in public APIs.
- ALL function parameters MUST have type annotations. No bare `def foo(x, y):` signatures.
- ALL function return types MUST be explicitly declared. No implicit `-> None` returns.
- Run `uv run python -m mypy --strict` (or at minimum `--warn-return-any --disallow-untyped-defs`) before committing.
