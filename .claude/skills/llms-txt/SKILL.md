---
name: llms-txt
description: Fetch and generate llms.txt documentation files for packages. Use when working with unfamiliar libraries, verifying API availability, or building LLM-readable docs.
---

# llms.txt Skill

## When to Use

- Before using an unfamiliar third-party library in code generation
- When needing to verify that a specific method, class, or function exists in a package
- When setting up project documentation optimized for LLM consumption
- After adding a new dependency to the project

## Core Principles

1. **Fetch before coding**: Before using an unfamiliar library, attempt to fetch its `llms.txt` from known documentation URLs. Many popular packages now serve LLM-optimized docs at `/llms.txt` or `/llms-full.txt`.
2. **Generate if missing**: If a package does not have a public `llms.txt`, generate one locally by extracting docstrings, type stubs, and public API surface into a structured markdown file.
3. **Cache locally**: Store fetched/generated `llms.txt` files in `.claude/llms-cache/<package>.txt` to avoid re-fetching and to keep context efficient.
4. **Verify against reality**: Cross-reference the `llms.txt` content against `verify_api.py` to ensure the documented APIs actually exist in the installed version.
5. **Respect versioning**: If the installed package version differs from the cached `llms.txt`, regenerate the cache.

## Scripts

- **`scripts/fetch_llms_txt.py`** — Fetches or generates llms.txt for a given package. Usage: `uv run python scripts/fetch_llms_txt.py --package <name> [--force] [--output <path>]`

## References

- **`reference.md`** — llms.txt standard, URL patterns, known providers, fallback strategies
- **`examples.md`** — Fetching and generating examples for common packages
