# Surgical Edit and Refactoring Rules

## File Modification Discipline
- NEVER replace an entire file to make a minor modification. ALWAYS use targeted, surgical search-and-replace edits.
- When a file is under 50 lines, full replacement is acceptable. Above 50 lines, always use targeted edits.
- NEVER output the entire file content when only a subset of lines needs changing.

## No Stubs or Placeholders
- NEVER create placeholder functions, TODO stubs, empty code generators, or dummy implementations.
- Write out the COMPLETE implementation or do not write it at all.
- Every function must have a real implementation body. No `pass`, `# TODO`, `raise NotImplementedError`, or empty function bodies.

## Import and Dependency Hygiene
- ALWAYS verify that all imports used in a file are actually installed and available in the project environment.
- Remove ALL unused imports before finalizing any file edit.
- When adding, removing, or renaming a module, or changing its public symbols, ALWAYS update the corresponding `__init__.py`. Run `uv run python .claude/skills/import-validator/scripts/validate_imports.py --path <dir>` to verify.
- NEVER create circular import dependencies. If A imports B, B must not import A (directly or transitively).

## Function Signature Changes
- When changing a function signature, you MUST use `grep` or `ripgrep` to find ALL callers across the codebase.
- Update ALL dependent callers simultaneously. Leaving even one outdated caller creates runtime errors.
- Update all related tests when changing function signatures.

## Git Discipline
- Before declaring any task finished, review the `git diff` to ensure no custom logic or hand-crafted core functions were accidentally deleted or overwritten.
- NEVER truncate or accidentally delete large blocks of code because of a file-level replacement.
- If you need to make changes to a file that has uncommitted manual changes by the user, always use surgical edits rather than full file replacement.

## Complexity Control
- NEVER create complex abstractions, design patterns, or inheritance hierarchies for simple requests that only need a few lines of code.
- Default to the simplest solution. If a 3-line fix solves the bug, do not build a complex wrapper pattern.
- NEVER build overly complex nested wrappers around an edge-case bug. Instead, refactor the single line of fundamentally flawed foundational logic causing it.

## Cross-File Consistency
- When updating a frontend component, you MUST identify and propagate matching changes to backend API routes and database schema.
- NEVER update UI beautifully while completely forgetting the underlying data layer.
- ALWAYS maintain sync between TypeScript types, Zod schemas, and Pydantic models that represent the same data.

## Database Schema Refactoring
- When modifying ORM models, ALWAYS generate the corresponding Alembic migration file. NEVER apply schema changes without a migration.
- Test both `upgrade` and `downgrade` migration paths before committing.
- Use the `safe-refactor` skill workflow for systematic refactoring: find callers → assess impact → update synchronously → validate.
