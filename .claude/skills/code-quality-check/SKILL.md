---
name: code-quality-check
description: Comprehensive code quality verification checklist. Run after code generation to catch common issues before they reach production.
---

# Code Quality Check

Run this skill after generating or modifying code to verify quality across syntax, imports, error handling, security, and performance before committing.

## When to Use

- After generating new functions, modules, or files — before presenting them to the user.
- After applying a diff or refactoring to verify no regressions slipped in.
- As a pre-commit gate before squashing or merging a feature branch.
- When auditing an unfamiliar codebase for quick quality wins.

## Core Categories

1. **Syntax/Structure** — bracket matching, indentation consistency, file size limits.
2. **Import Hygiene** — unused, missing, grouped correctly, no circular dependencies.
3. **Function Integrity** — no stubs/placeholders, no dead code, consistent signatures.
4. **Error Handling** — async try/except coverage, HTTP error codes, no silent swallowing.
5. **Security** — no hardcoded secrets, no wildcard CORS, no raw SQL (see `security-audit`).
6. **State Management** — React useEffect dependency arrays, no direct state mutation.
7. **Testing** — coverage thresholds, no deleted tests, deterministic assertions.
8. **Performance** — pagination on list endpoints, no N+1 queries, file reading discipline.

## References

- `reference.md` — detailed checklist items per category, language-specific rules.
- `examples.md` — runnable snippets for ruff, mypy, eslint, tsc, and AST-based checks.

## Scripts

- `scripts/quality_check.py` — automated quality gate for Python and TypeScript files. Run with `--path <target>` and optional `--report json`.

> For operational refactoring workflow, see `safe-refactor`. For testing methodology, see `testing-and-qa`.
