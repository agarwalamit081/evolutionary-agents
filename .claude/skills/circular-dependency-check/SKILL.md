---
name: circular-dependency-check
description: Detect and resolve circular import dependencies in Python and TypeScript/JavaScript projects. Use when adding new modules, restructuring code, or after refactoring.
---

# Circular Dependency Check

## When to Use

- After adding new import statements or cross-module references
- When restructuring a codebase (moving files, splitting modules)
- Encountering `ImportError` / `cannot read property of undefined` at runtime
- Before merging a branch that modifies module boundaries or shared utilities

## Core Workflow

1. **Detect** — Run `scripts/detect_cycles.py` against the project root to surface cycles.
2. **Map** — Review the reported cycle chains to understand which modules form the loop.
3. **Resolve** — Apply one of the four resolution strategies from `reference.md`:
   - Extract shared code into a third module
   - Use dependency injection (pass the dependency as a parameter)
   - Use lazy imports (import inside function body)
   - Extract interfaces/types into a separate types module
4. **Prevent** — Re-run detection; add the script to CI to catch regressions.

## References

- `reference.md` — Python import mechanics, TypeScript `madge` usage, resolution strategies in depth, ImportError trace reading.
- `examples.md` — Six runnable examples covering Python and TypeScript detection and resolution patterns.

## Scripts

- `scripts/detect_cycles.py` — CLI tool that scans a directory for `.py` or `.ts` files, builds an import graph, and reports cycles via DFS.
  - Usage: `python scripts/detect_cycles.py --path <dir> --language python|ts [--verbose]`
