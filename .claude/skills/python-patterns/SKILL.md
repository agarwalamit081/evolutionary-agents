---
name: python-patterns
description: Python best practices — asyncio, type hinting, dataclasses, decorators, context managers, and idiomatic Python patterns.
---

**When to Use**
- Writing Python code with async/await patterns.
- Adding type hints or designing typed interfaces.
- Creating data models with dataclasses or Pydantic.
- Implementing decorators, context managers, or generators.
- Optimizing Python performance or writing idiomatic code.

**Core Principles**
1. **Type Everything**: Use type hints for all function signatures. Prefer `from __future__ import annotations`.
2. **Async by Default**: Use asyncio for I/O-bound work. `asyncio.gather` for concurrent ops. Never block the event loop.
3. **Data Modeling**: Use dataclasses or Pydantic for structured data. Avoid raw dicts for complex types.
4. **Decorators for Cross-Cutting Concerns**: Use decorators for logging, retries, caching, rate limiting.
5. **Context Managers**: Use `with` for resource management. Create custom ones with `@contextmanager`.
6. **Environment Awareness**: Always use `uv run python` for executing Python scripts. Verify the environment with `uv run python -c 'import sys; print(sys.executable)'` at session start.

**References**
- Load `reference.md` for asyncio patterns, type hinting, data modeling, decorators, and performance tips.
- Load `examples.md` for ready-to-use code patterns.

**Scripts**
- `scripts/type_check_stub.py`: Scaffold type-checked function stubs from a spec. Run with: `uv run python scripts/type_check_stub.py`
