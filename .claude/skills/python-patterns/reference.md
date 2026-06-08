---
description: Python Patterns Reference
---

## Asyncio Patterns

- **Event loop**: `asyncio.run(main())` as entry point.
- **Concurrent execution**: `asyncio.gather(*coros)` for parallel, `TaskGroup` (3.11+) for structured concurrency.
- **Semaphore limiting**: `asyncio.Semaphore(n)` to cap concurrent I/O operations.
- **Never block**: Use `asyncio.to_thread()` for CPU-bound or blocking I/O calls.

## Type Hinting

- **Generics**: `list[str]`, `dict[str, int]`, `Callable[[str], bool]`.
- **Protocol**: Structural subtyping (duck typing with type safety).
- **TypeVar**: Generic functions `T = TypeVar("T")`.
- **overload**: Multiple signatures for the same function.
- **Forward references**: `from __future__ import annotations` at module top.

## Data Modeling

### Dataclasses
- `@dataclass(frozen=True)` for immutable value objects.
- `@dataclass(slots=True)` for memory-efficient instances (3.10+).
- `field(default_factory=list)` for mutable defaults.
- `__post_init__` for validation after initialization.

### Pydantic
- `BaseModel` for validated data with automatic serialization.
- `field_validator` for custom validation logic.
- `model_config = ConfigDict(frozen=True)` for immutable models.
- `model_dump()` and `model_dump_json()` for serialization.

## Decorators

- Always use `@functools.wraps(func)` to preserve metadata.
- Parameterized decorators: outer function takes params, returns decorator.
- Class decorators: use `__call__` or `__init__` + `__call__`.

## Context Managers

- `@contextlib.contextmanager`: yield the resource, cleanup in finally.
- Async: `@contextlib.asynccontextmanager` with `async with`.

## Iterators & Generators

- Generator expressions: `(x*2 for x in range(100))` — lazy evaluation.
- `yield` for producing values lazily.
- `itertools`: `chain`, `islice`, `groupby`, `product`.

## Error Handling

- Custom exception hierarchy: `class AppError(Exception): ...` with specific subclasses.
- Exception groups (3.11+): `except*` for handling multiple exceptions from TaskGroup.
- Always use specific exception types, never bare `except:`.

## Performance

- List comprehensions over `map`/`filter` for readability.
- `functools.lru_cache` for memoization of pure functions.
- `__slots__` for memory-efficient classes with many instances.
- Generators over lists for large datasets (streaming).
