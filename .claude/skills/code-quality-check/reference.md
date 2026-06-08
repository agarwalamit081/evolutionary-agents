# Code Quality Check — Reference

Detailed checklist items and language-specific rules for the quality gate.

## Syntax / Structure

- [ ] All brackets, braces, and parentheses are balanced.
- [ ] Indentation is consistent (4 spaces for Python, 2 spaces for TS/JS).
- [ ] No trailing whitespace or mixed line endings.
- [ ] Files under 500 lines — warn if exceeding; split large modules.
- [ ] No commented-out code blocks left behind.

## Import Hygiene

- [ ] No unused imports (ruff `F401`, eslint `no-unused-vars`).
- [ ] No missing imports — all referenced symbols are imported.
- [ ] Imports grouped: stdlib, third-party, local; separated by blank lines.
- [ ] No circular imports (run `scripts/import_validator.py` from `code-quality-and-patterns`).
- [ ] Avoid `from foo import *` — use explicit named imports.

## Function Integrity

- [ ] No stub bodies: `pass`, `NotImplementedError`, `TODO`, `FIXME`, bare `...`.
- [ ] No dead code: unreachable branches, unused private functions.
- [ ] Function signatures match their callers — type-check before merge.
- [ ] Return types are explicit; avoid implicit `Any` leakage.

## Error Handling

- [ ] Every `async` function that performs I/O has a `try/except` block.
- [ ] HTTP handlers return appropriate status codes (400, 404, 422, 500).
- [ ] No bare `except:` or `except Exception: pass` — at minimum log the error.
- [ ] Exceptions are specific: catch `ValueError`, not `Exception`.
- [ ] Cleanup resources with `finally` or context managers.

## Security

- [ ] No hardcoded secrets, tokens, or API keys (see `security-audit` skill).
- [ ] No wildcard `Access-Control-Allow-Origin: *` in production configs.
- [ ] No raw SQL string interpolation — use parameterized queries.
- [ ] User inputs are validated and sanitized before processing.
- [ ] Dependencies pinned with hashes where possible.

## State Management (React/TypeScript)

- [ ] `useEffect` dependency arrays include all referenced values.
- [ ] No direct state mutation — use setters or immutable patterns.
- [ ] Derived values use `useMemo` only when computation is expensive.
- [ ] Cleanup functions present for subscriptions and timers.

## Testing

- [ ] Coverage >= 80% for new code; no deleted tests without replacement.
- [ ] Tests are deterministic — no reliance on execution order or timing.
- [ ] Mocks target external boundaries, not internal modules.
- [ ] Edge cases covered: empty input, null, large payloads, auth failures.

## Performance

- [ ] List endpoints implement pagination (cursor or offset).
- [ ] No N+1 queries — use `select_related` / `prefetch_related` or DataLoader.
- [ ] Read files line-by-line or in chunks; avoid loading entire files into memory.
- [ ] Database queries use indexes — verify with `EXPLAIN ANALYZE`.

## Python-Specific Rules

- **Ruff**: `ruff check .` for linting, `ruff format --check .` for formatting.
- **Mypy**: `mypy --strict src/` — address every `Any` and missing type.
- **Docstrings**: public functions have Google-style docstrings.
- **F-strings**: prefer over `.format()` and `%` for readability.
- **Dataclasses / Pydantic**: use for structured data over raw dicts.

## TypeScript-Specific Rules

- **ESLint**: `eslint . --ext .ts,.tsx` with strict rule set.
- **tsc**: `tsc --noEmit --strict` — zero errors required.
- **No `any`**: use `unknown` and narrow with type guards.
- **Nullish coalescing**: prefer `??` over `||` for falsy-zero cases.
- **Strict null checks**: enabled in `tsconfig.json`.

## Common Code Generation Artifacts

These patterns frequently appear in AI-generated code and must be cleaned up:

1. Placeholder comments like `# Your code here` or `// implement this`.
2. Overly verbose variable names (`theResultOfTheCalculation`).
3. Redundant type annotations the compiler infers (`: string` on string literals).
4. Unnecessary imports pulled from training data but not used.
5. Exception handlers that log and re-raise without adding context.
6. Test files with `# Arrange / # Act / # Assert` comments on every block.
