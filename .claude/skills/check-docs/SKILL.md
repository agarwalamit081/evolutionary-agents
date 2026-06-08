---
name: check-docs
description: Verify library documentation before using new features or APIs. Use when introducing a new package, upgrading a dependency, or using an unfamiliar API.
---

# check-docs

Verify library documentation and API availability before writing code that depends on them.

## When to Use

- Introducing a new third-party package into the project
- Upgrading a dependency to a new major or minor version
- Using an unfamiliar method, class, or decorator from an existing dependency
- Encountering an `ImportError`, `AttributeError`, or deprecation warning at runtime
- Writing code that relies on library-specific constants, enums, or type hints

## Core Workflow

1. **Identify library + installed version** — Run `pip show <package>` or use `importlib.metadata.version("<package>")` to confirm what is actually installed.
2. **Check official documentation** — Look up the exact method signature, parameters, and return type in the library's docs for the installed version. Never rely on memory.
3. **Verify the method/property exists** — Use `scripts/verify_api.py --package <pkg> --method <attr>` to confirm the attribute is present on the installed version.
4. **Check for deprecation warnings** — Run `python -W all` or inspect the changelog for any deprecation notices that affect the API you plan to use.
5. **Document usage with version comments** — Add an inline comment noting the minimum version required, e.g., `# requires langgraph>=0.2.0`.

## References

- Library changelogs: always read the changelog between your last-known version and the installed version before using new features.
- Migration guides: check for dedicated migration docs when upgrading across major versions.
- Deprecation policy: respect the library's deprecation timeline; avoid deprecated APIs in new code.

## Scripts

- `scripts/verify_api.py` — CLI tool that checks whether a package can be imported, whether a specific method/attribute exists, and whether the installed version meets a minimum requirement. Run with `--package`, `--method`, and/or `--min-version`.
