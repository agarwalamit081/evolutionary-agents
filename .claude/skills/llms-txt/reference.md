# llms.txt Reference

## What is llms.txt?

`llms.txt` is a emerging standard for providing LLM-optimized documentation from project websites. It is a markdown file served at well-known URLs that gives AI agents structured, concise access to a project's API surface without needing to crawl the full documentation site.

## URL Patterns

| URL | Content |
|---|---|
| `https://<docs-site>/llms.txt` | Summary — brief overview, quick-start, key API list |
| `https://<docs-site>/llms-full.txt` | Full content — complete API reference, all examples |

## Known Providers

These packages are known to serve `llms.txt` files:

| Package | Documentation URL |
|---|---|
| FastAPI | `https://fastapi.tiangolo.com/llms-full.txt` |
| LangChain | `https://python.langchain.com/llms.txt` |
| SQLAlchemy | `https://docs.sqlalchemy.org/llms.txt` |
| Pydantic | `https://docs.pydantic.dev/llms.txt` |
| httpx | `https://www.python-httpx.org/llms.txt` |
| DuckDB | `https://duckdb.org/llms.txt` |
| LiteLLM | `https://docs.litellm.ai/llms.txt` |
| Django | `https://docs.djangoproject.com/llms.txt` |

The list is growing rapidly. Always attempt the standard URL patterns before falling back.

## Fallback Strategy

When `llms.txt` is not available for a package:

1. **Read README.md** from the installed package directory or GitHub repository
2. **Extract docstrings** using Python's `inspect` and `pydoc` modules on the installed package
3. **Parse type stubs** (`.pyi` files) if available in the package or in `typeshed`
4. **Check `__all__` exports** to identify the public API surface
5. **Generate from source** using `ast` module to walk the package tree

## Cache Management

- Cache directory: `.claude/llms-cache/`
- Cache filename: `<package-name>.txt`
- Cache validity: Regenerate when `uv pip show <package>` reports a different version than the cached file's header
- Force refresh: `uv run python scripts/fetch_llms_txt.py --package <name> --force`

## Integration with verify_api.py

After fetching or generating an `llms.txt`, cross-reference specific methods with:
```bash
uv run python .claude/skills/check-docs/scripts/verify_api.py --package <pkg> --method <method>
```

This catches cases where the `llms.txt` documents an API from a newer version than what is installed.
