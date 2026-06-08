# Import Validator — Reference

## Python Import Validation

### Unused Imports (F401)
An import that is never referenced in the module. Ruff flags these with `F401`.
Common causes: leftover imports after refactoring, IDE auto-imports, or imports
added "just in case." Every import must have at least one usage in the module body.

### Missing Imports (F821)
A name is used but never imported or defined. Ruff flags these with `F821`.
These cause `NameError` at runtime. Always verify that generated code includes
all required imports.

### Redefined Imports (F811)
An import name is defined multiple times in the same module. Ruff flags with `F811`.
This usually indicates a merge conflict or copy-paste error. The later definition
shadows the earlier one.

### Tool Integration
```bash
# Check for unused and redefined imports
ruff check --select F401,F811 .

# Auto-fix unused and redefined imports
ruff check --select F401,F811 --fix .
```

## TypeScript/JavaScript Import Validation

### Unused Imports
ESLint rule `@typescript-eslint/no-unused-vars` catches unused imports when
configured to check import bindings. Alternatively, the `no-unused-vars` rule
with TypeScript-aware settings.

### Missing Imports
TypeScript's compiler (`tsc --noEmit`) catches missing imports as part of type
checking. Ensure `tsconfig.json` has `noUnusedLocals` and `noUnusedParameters`
enabled for stricter validation.

### Type-Only Imports
Use `import type { ... }` for imports that are only used as types. This ensures
they are erased at compile time and do not appear in the JavaScript output.
```typescript
import type { User } from "./types";    // type-only — erased at compile
import { formatUser } from "./utils";    // value import — kept at compile
```

## Import Grouping Conventions

### Python (isort / ruff format rules)
Three groups, separated by blank lines:
1. **Standard library** — `os`, `sys`, `json`, `pathlib`, etc.
2. **Third-party** — `numpy`, `requests`, `fastapi`, etc.
3. **First-party / local** — modules from the current project

Within each group, imports are sorted alphabetically. Relative imports (`from .`)
belong in the local group. Use `ruff format` or `isort` to enforce automatically.

### TypeScript
Similar three-group convention:
1. **Node / platform built-ins** — `fs`, `path`, `http`
2. **Third-party packages** — `react`, `express`, `lodash`
3. **Local modules** — relative imports (`./`, `../`)

## `__init__.py` Maintenance Protocol

When a new Python module is added to a package:
1. Determine if the module exposes public symbols that external consumers need.
2. Add explicit re-exports to `__init__.py`:
   ```python
   from .new_module import NewClass, new_function
   ```
3. Avoid wildcard imports (`from .new_module import *`) — they hide the public API.
4. Run `ruff check --select F401` on the `__init__.py` to ensure re-exports are used.
5. If the module is internal-only (prefixed with `_`), skip the re-export.

## Barrel File (`index.ts`) Maintenance Protocol

When a new TypeScript module is added to a package directory:
1. Determine if the module should be part of the public API.
2. Add a re-export to `index.ts`:
   ```typescript
   export { NewComponent } from "./NewComponent";
   export type { NewComponentProps } from "./NewComponent";
   ```
3. Use `export type` for type-only re-exports.
4. Do not re-export internal utilities (prefixed with `_` or in `internal/`).
5. Verify with `tsc --noEmit` that no imports are missing.

## Circular Dependency Detection

Circular imports occur when two or more modules import from each other, directly
or transitively. They cause `ImportError` (Python) or `undefined` references
(TypeScript) at runtime.

Prevention strategies:
- **Dependency injection**: Pass collaborators rather than importing them.
- **Extract shared types**: Move shared definitions to a separate module.
- **Use late imports**: Import inside functions instead of at module level (Python).
- **Event-driven patterns**: Decouple modules via events/callbacks instead of imports.

For automated detection, see the `circular-dependency-check` skill.

## Cross-References

- **circular-dependency-check** skill — dedicated tooling for detecting and
  resolving circular import cycles in the dependency graph.
- **python-patterns** skill — Python module structure conventions.
- **code-quality-and-patterns** skill — general linting and formatting standards.
