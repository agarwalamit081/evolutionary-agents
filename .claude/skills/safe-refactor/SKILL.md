---
name: safe-refactor
description: Enforce safe refactoring practices when modifying function signatures, moving code, or restructuring modules. Use after any refactoring operation to verify consistency.
---

# Safe Refactor

## When to Use

- Renaming a function, class, method, or variable across the codebase
- Changing a function signature (adding/removing/reordering parameters)
- Moving a module, file, or directory to a new location
- Extracting or inlining methods during code cleanup

## Core Workflow

1. **Find All Callers** — Use `scripts/find_callers.py` or ripgrep to locate every reference across the entire codebase. Never rely on IDE hints alone.
2. **Assess Impact** — Identify all affected files: call sites, test files, type stubs, barrel exports (`__init__.py`, `index.ts`), and documentation. Count the blast radius before changing anything.
3. **Update Synchronously** — Apply changes in order: definition first, then every caller, then every test, then barrel exports. Never leave the codebase in a half-updated state.
4. **Validate** — Run linter + type checker + relevant tests. Run `git diff --stat` to confirm the change scope matches expectations. No orphaned references should remain.

## References

- `reference.md` — Deep refactoring protocols, barrel file update rules, circular dependency avoidance
- `examples.md` — Concrete refactoring patterns for Python and TypeScript

## Scripts

- `scripts/find_callers.py` — CLI tool to find all callers of a function across the codebase, distinguishing definitions from call sites and tracking imports.
