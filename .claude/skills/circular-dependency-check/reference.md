# Circular Dependency Detection — Reference

## Python Circular Import Mechanics

Python executes a module's top-level code on first import. If module A imports B and B imports A (directly or transitively), one of the two modules will see a partially-initialized version of the other. The interpreter raises `ImportError` or `AttributeError` depending on where the cycle is broken.

### Module-Level vs Function-Level Imports

- **Module-level**: `import foo` or `from foo import Bar` at the top of the file. Executed once when the module is loaded. Safe unless a cycle exists.
- **Function-level** (lazy): `import foo` inside a function body. Executed each time the function is called, after both modules are fully loaded. Breaks most cycles.

### Reading ImportError Traces

1. Look for `ImportError: cannot import name 'X'` or `AttributeError: module 'Y' has no attribute 'Z'`.
2. The traceback shows the import chain — read it bottom-to-top to trace the cycle.
3. The module that raises is the one that was imported *second*; it sees the first module before the first module finished executing.
4. A typical pattern: `a.py` → `b.py` → `c.py` → `a.py`. The error fires in `a.py` when `c.py` tries to `from a import Something` and `a.py` hasn't finished loading yet.

## TypeScript/JavaScript Circular Dependencies

ES module semantics differ from Python: bindings are live references, not copies. A circular dependency does not throw at load time, but imported bindings may be `undefined` when accessed before the exporting module finishes evaluating.

### Detection with `madge`

```bash
npx madge --circular src/
```

Outputs a list of circular paths. Requires `madge` as a dev dependency or npx access. The `detect_cycles.py` script falls back to regex-based import parsing when `madge` is unavailable.

## Resolution Strategies

### 1. Extract Shared Code into a Third Module

Move the shared constants, utilities, or base classes that both modules depend on into a new module with no dependencies on either consumer. This is the cleanest solution and should be preferred.

**When to use**: Both modules depend on the same data or behavior, and that shared code has no reason to live in either module.

### 2. Dependency Injection

Instead of importing the dependency at module level, accept it as a function or constructor parameter. The caller is responsible for providing the concrete implementation.

**When to use**: The dependency is only needed in specific function calls, not at module initialization time. Works well with frameworks that manage wiring (FastAPI, NestJS).

### 3. Lazy Imports

Move the `import` statement inside the function that needs it. The import executes after all modules have finished loading, so the partially-initialized problem is avoided.

**When to use**: Quick fix for simple cycles. Be aware of minor performance overhead on first call and reduced readability. Avoid stacking many lazy imports in hot paths.

### 4. Interface / Type Extraction

Create a separate module that defines only interfaces, types, or abstract base classes. Both consumer modules import from the types module, which has no logic and therefore no transitive dependencies.

**When to use**: The cycle is caused by type annotations or protocol definitions. Particularly common in TypeScript projects.

## Prevention Rules

- **Layering**: Enforce a unidirectional dependency graph (e.g., `presentation → domain → infrastructure`). No upward imports.
- **CI guard**: Run `detect_cycles.py` (or `madge`) as a CI step that fails on any detected cycle.
- **Pre-commit hook**: Add a pre-commit hook that checks changed files for new cycles.
- **Module ownership**: Assign clear ownership boundaries so cross-team imports are explicit and reviewed.

## Code Organization Patterns

```
project/
├── domain/           # Core business logic, no external deps
│   ├── models.py
│   └── interfaces.py # Abstract base classes / protocols
├── infrastructure/   # DB, HTTP clients — depends on domain
│   └── repos.py
└── presentation/     # API handlers — depends on domain (not infra)
    └── routes.py
```

Dependency direction: `presentation → domain ← infrastructure`. Infra and presentation never import each other directly.

## Cross-Reference

- See `import-validator` skill for broader import hygiene checks (unused imports, missing imports, alias consistency).
