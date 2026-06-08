---
name: library-usage
description: Enforce correct library selection and usage patterns for Python and JavaScript/TypeScript projects. Use when choosing libraries, verifying API compatibility, or preventing hallucinated library methods.
---

# Library Usage Skill

## When to Use

- Choosing a library for a new feature or replacing an existing dependency
- Writing imports or calling APIs — verify the method exists and signature matches the installed version
- Reviewing dependency lists (`pyproject.toml`, `package.json`, `requirements.txt`) for policy compliance
- Debugging import errors, `AttributeError`, or deprecation warnings from third-party packages
- Generating code that calls any external package — prevents hallucinated methods and incorrect parameter names

## Core Principles

1. **Prefer established libraries** — use the tables in `reference.md` as the authoritative guide. If a category is listed, use the preferred option unless the project already uses a documented alternative.
2. **Verify before writing** — when calling an unfamiliar API, confirm the method signature against the library's actual docs or source rather than guessing from the name.
3. **Pin all dependencies** — every package must declare a version constraint (`>=`, `==`, or compatible range). Unpinned deps are a FAIL in `check_deps.py`.
4. **Avoid anti-patterns** — never rely on internal/private APIs (`_`-prefixed), undocumented behavior, or mixing major versions of the same library in one project.
5. **Validate LLM output** — when parsing JSON from LLM responses, always use `json-repair` or `dirtyjson` before schema validation to handle malformed output gracefully.
6. **Verify with verify_api.py** — before using any unfamiliar method, confirm it exists: `uv run python .claude/skills/check-docs/scripts/verify_api.py --package <pkg> --method <attr>`.

## References

- **`reference.md`** — full preferred-library tables (Python and JS/TS), verification protocol, and anti-pattern catalog
- **`examples.md`** — idiomatic code snippets for the most commonly misused libraries

## Scripts

- **`scripts/check_deps.py`** — validates `pyproject.toml`, `requirements.txt`, or `package.json` for pinned versions and preferred-library compliance. Run it after any dependency change.
