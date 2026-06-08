---
name: update-tests
description: Systematic workflow for updating tests when changing function signatures, adding features, or modifying business logic. Use after any code change that affects existing tests.
---

# Update Tests

## When to Use

- You changed a function signature (parameters, return type, exceptions).
- You added, removed, or renamed a public method or class.
- You modified business logic that existing tests depend on.
- You fixed a bug and need a regression test to prevent recurrence.

## Core Workflow

1. **Baseline** — Run the full test suite and capture the passing state before making changes.
2. **Identify** — Use `scripts/find_affected_tests.py` (or `rg`) to locate every test, fixture, and conftest that references the changed symbol.
3. **Update call sites** — Fix imports, argument lists, fixture definitions, and mock data to match the new signature or behavior.
4. **Add new tests** — Cover new parameters, edge cases, error paths, and integration scenarios introduced by the change.
5. **Run & fix** — Execute the full suite. If tests fail, fix the application code — never delete or bypass a failing test.

## Critical Rules

- **NEVER** delete, comment out, `@pytest.skip`, `xit`, or otherwise bypass a failing test.
- If a test fails after your change, the **application code** is wrong, not the test.
- Fix the application code to pass the test. If the test itself was incorrect *before* your change, fix it and document why in the commit message.
- Every bug fix MUST include a regression test that would have caught the original bug.

## References

- `reference.md` — Detailed workflow steps, quality standards, and anti-patterns.
- `examples.md` — Concrete snippets for finding tests, updating fixtures, and adding regression tests.

## Scripts

- `scripts/find_affected_tests.py --function <name> [--path <root>]` — Locate all test files, fixtures, and imports that reference a changed function or class.
