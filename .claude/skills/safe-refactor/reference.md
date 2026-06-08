# Safe Refactor — Reference

## 4-Step Refactoring Workflow (In Depth)

### Step 1: Find All Callers

Use `scripts/find_callers.py --function NAME --path ROOT --type python|ts|all` to discover every reference. Supplement with manual `rg` for edge cases (string references, decorators, dynamic dispatch). Always search the full project root — not just `src/`.

### Step 2: Assess Impact

Catalog every affected file into categories:
- **Call sites** — Direct invocations of the symbol
- **Tests** — Unit tests, integration tests, fixtures referencing the symbol
- **Type definitions** — Interfaces, type aliases, protocol classes
- **Barrel exports** — `__init__.py` / `index.ts` re-exports
- **Documentation** — Docstrings, README, inline comments referencing the symbol

If the blast radius exceeds 10 files, consider whether the refactor is worth it or if a deprecation path (old name wrapping new name) is safer.

### Step 3: Update Synchronously

Apply changes atomically and in strict order:
1. Update the definition (function signature, class location, module path)
2. Update every call site — no partial updates
3. Update every test that references the old form
4. Update barrel exports so downstream consumers are not broken
5. Update type stubs and interfaces

**Never commit a half-refactored state.** If the change is large, use a feature branch and complete all updates before merging.

### Step 4: Validate

Run in sequence:
1. Linter (`ruff check .`, `eslint .`)
2. Type checker (`mypy .`, `tsc --noEmit`)
3. Tests (`pytest`, `jest`) — focus on affected test files first
4. `git diff --stat` — verify the file list matches the impact assessment

## Barrel File Update Protocol

When moving or renaming an exported symbol:
- Remove the old name from `__init__.py` / `index.ts` `__all__` or re-export list
- Add the new name in the same location
- If the symbol is part of a public API, add a deprecation alias pointing to the new name
- Verify no wildcard imports (`from package import *`) are affected

## Linter / Type Checker Integration

- Run linter before and after refactoring to catch new warnings
- Type checkers catch missing positional arguments, wrong argument types, and stale imports
- If the type checker passes, you have high confidence no call site was missed

## Git Diff Review

- `git diff --stat` — Confirm file list matches expectation
- `git diff` — Review each hunk for correctness
- `git diff --check` — Catch whitespace errors introduced during refactor

## Module Restructuring Without Circular Dependencies

- Before moving a module, draw its dependency graph (imports in, imports out)
- Never move a module into a package it already imports from (creates cycles)
- Use forward references or `TYPE_CHECKING` guards to break unavoidable cycles
- After restructuring, run a cycle-detection pass: `python -c "import package"` or a dedicated tool

## When to ASK Before Refactoring

- **Ambiguous business logic** — If the function's behavior is not fully understood, ask before changing its signature
- **Public API changes** — If the symbol is consumed by external packages or services, confirm the deprecation strategy
- **Performance-critical paths** — Restructuring hot loops may change performance; ask before proceeding
- **Shared ownership** — If another team owns the module, coordinate before refactoring

## Cross-References

- **`code-quality-and-patterns`** — Architectural patterns and design principles (this skill = operational refactoring, that = architectural guidance)
- **`import-validator`** — Post-restructuring import validation
- **`update-tests`** — Test update protocols after signature changes
