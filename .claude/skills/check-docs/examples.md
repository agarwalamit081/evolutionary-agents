# check-docs — Examples

## Example 1: Get installed version before using an API

```python
# Use pip show for a quick CLI check
# $ pip show langgraph
# Name: langgraph
# Version: 0.2.28
# ...

# Use importlib.metadata from within Python
import importlib.metadata

version = importlib.metadata.version("langgraph")
print(f"langgraph version: {version}")  # langgraph version: 0.2.28
```

## Example 2: Verify a LangGraph `interrupt()` API exists in installed version

```python
import importlib.metadata
from packaging.version import Version

installed = Version(importlib.metadata.version("langgraph"))
# interrupt() was added in langgraph 0.2.5 as a top-level export
if installed >= Version("0.2.5"):
    from langgraph.types import interrupt
    value = interrupt("Provide input:")
else:
    raise RuntimeError(f"interrupt() requires langgraph>=0.2.5, got {installed}")
```

## Example 3: Check pgvector extension version before using specific operators

```python
import psycopg2

with psycopg2.connect(DSN) as conn:
    with conn.cursor() as cur:
        # Check the installed extension version
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("pgvector extension not installed")
        ext_version = row[0]
        print(f"pgvector extension version: {ext_version}")

        # Halfvec support was added in pgvector 0.5.0
        # Verify before using halfvec columns or operators
```

## Example 4: Deprecation check pattern with Python `warnings` module

```python
import warnings

# Catch deprecation warnings in tests
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", DeprecationWarning)

    import some_library
    result = some_library.old_method()

    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    if deprecations:
        for w in deprecations:
            print(f"DEPRECATED: {w.message} (in {w.filename}:{w.lineno})")
```

## Example 5: Documenting API usage with version comments

```python
from anthropic import Anthropic

client = Anthropic()

# requires anthropic>=0.39.0 for cached system messages
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "You are a helpful assistant.",
            "cache_control": {"type": "ephemeral"},  # requires anthropic>=0.30.0
        }
    ],
    messages=[{"role": "user", "content": "Hello"}],
)
```

## Example 6: Checking FastAPI route decorator signature across versions

```python
import inspect
import importlib.metadata
from packaging.version import Version

import fastapi

installed = Version(importlib.metadata.version("fastapi"))

# FastAPI changed the @app.route decorator signature in 0.100+
# Verify the parameters your code uses are still valid
sig = inspect.signature(fastapi.FastAPI.get)
params = list(sig.parameters.keys())
print(f"FastAPI {installed}: app.get parameters = {params}")

if "openapi_extra" not in params:
    print("WARN: openapi_extra not available in this version")
```
