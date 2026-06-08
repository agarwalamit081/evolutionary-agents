---
name: import-validator
description: Validate import hygiene after code generation or refactoring. Use after writing or modifying Python and TypeScript/JavaScript files to ensure clean import structure.
---

# Import Validator

## When to Use

- After generating or modifying Python files with new imports
- After generating or modifying TypeScript/JavaScript files with new imports
- After refactoring that moves, renames, or deletes modules
- Before committing changes to ensure no import-related lint errors
- When adding new modules to a package that requires `__init__.py` or `index.ts` updates

## Core Principles

1. **No Unused Imports**: Every import statement must reference a name that is actually used in the module. Unused imports add noise and can mask real issues.
2. **No Missing Imports**: Every name referenced in a module must be imported or defined locally. Missing imports cause `NameError` at runtime.
3. **Proper Grouping**: Imports must be organized into three groups separated by blank lines: stdlib, third-party, and local imports. Use `isort` or `ruff format` conventions.
4. **Package Exports**: When creating, deleting, or renaming a module, or changing its public symbols, you MUST immediately update the parent `__init__.py` (Python) or `index.ts` (TypeScript) to add or remove the corresponding `from .module import` line. Run `uv run python scripts/validate_imports.py --path <package>` to verify.
5. **No Circular Dependencies**: Verify the import graph contains no cycles. Circular imports cause `ImportError` at runtime and indicate architectural coupling issues.

## References

- **reference.md** — Detailed rules for Python (F401, F821, F811) and TypeScript import validation, `__init__.py` and barrel file protocols, tool integration.
- **examples.md** — Code snippets for ruff, eslint, `__init__.py` updates, barrel files, import grouping, and `import type` usage.

## Scripts

- **scripts/validate_imports.py** — CLI tool to validate import hygiene. Usage: `python scripts/validate_imports.py --path <file_or_dir> [--fix] [--language python|ts|auto]`
