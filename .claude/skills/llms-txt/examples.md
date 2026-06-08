# llms.txt Examples

## Fetching from a known provider

```bash
# FastAPI provides llms-full.txt
uv run python .claude/skills/llms-txt/scripts/fetch_llms_txt.py --package fastapi

# This caches to .claude/llms-cache/fastapi.txt
```

## Fetching when provider is unknown

```bash
# Script tries common URL patterns, then falls back to docstring extraction
uv run python .claude/skills/llms-txt/scripts/fetch_llms_txt.py --package httpx
```

## Force regenerating the cache

```bash
# After upgrading a package, force regenerate the cache
uv run python .claude/skills/llms-txt/scripts/fetch_llms_txt.py --package pydantic --force
```

## Using the cached docs in code generation

After fetching, the cached `llms.txt` can be read to inform code generation:

```python
# The cached file at .claude/llms-cache/fastapi.txt contains:
# - Quick start guide
# - Key API reference (FastAPI, Depends, HTTPException, etc.)
# - Common patterns and examples
```

## Verifying a specific API from llms.txt

```bash
# After reading the cached docs, verify a specific method exists
uv run python .claude/skills/check-docs/scripts/verify_api.py \
  --package fastapi \
  --method FastAPI.add_api_route
```

## Generating from docstrings (fallback)

When a package has no public `llms.txt`:

```bash
# The script extracts public API from installed package docstrings
uv run python .claude/skills/llms-txt/scripts/fetch_llms_txt.py --package sqlalchemy
# Output: .claude/llms-cache/sqlalchemy.txt with extracted API reference
```

## Programmatic usage in agents

```python
from pathlib import Path

cache_dir = Path(".claude/llms-cache")
package_docs = cache_dir / "fastapi.txt"

if package_docs.exists():
    content = package_docs.read_text()
    # Use content to inform code generation decisions
else:
    # Fetch first
    import subprocess
    subprocess.run([
        "uv", "run", "python",
        ".claude/skills/llms-txt/scripts/fetch_llms_txt.py",
        "--package", "fastapi"
    ])
```
