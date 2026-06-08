# check-docs — Reference

## Version Pinning and Verification Protocol

- Always pin dependencies in `requirements.txt` or `pyproject.toml` with `>=x.y.z,<x.(y+1).0` semantics (or exact pins for critical libraries).
- Before using any API, confirm the installed version matches what the documentation describes.
- Use `pip show <package>` for a quick check or `importlib.metadata.version("<package>")` from within Python.
- When a `pyproject.toml` specifies a minimum version, treat that as the baseline; verify the actual installed version is at least that high.

## Libraries with Frequent API Changes

The following libraries are known for breaking or evolving their public APIs across minor versions. Always verify before use:

| Library | Risk Areas |
|---|---|
| **LangChain** | Chain constructors, callback managers, output parsers, retrieval patterns |
| **LangGraph** | `interrupt()`, `Command`, state schema, checkpointer APIs |
| **Anthropic SDK** | Tool use format, message parameters, streaming API, caching headers |
| **OpenAI SDK** | Response format, function calling schema, embedding models, assistants API |
| **pgvector** | Distance operators (`<=>`, `<->`), index types, extension version vs PostgreSQL version |
| **Playwright** | Locator API, auto-waiting behavior, browser context options |
| **FastAPI** | Route decorator signatures, dependency injection, `Response` model changes |

## Deprecation Detection Strategies

1. **Python `warnings` module** — Run your test suite with `python -W error::DeprecationWarning` to catch deprecation warnings as errors.
2. **Changelog parsing** — Read the library's `CHANGELOG.md` or GitHub Releases page. Search for "deprecat" to find relevant entries.
3. **`DeprecationWarning` in docstrings** — Many libraries annotate deprecated methods in their docstrings; inspect the source or rendered docs.
4. **Type stubs and `@deprecated` decorators** — Some libraries use `typing_extensions.deprecated` or similar markers that IDEs and type checkers can surface.

## Reading Changelogs Effectively

- Start from your current installed version and read forward to the target version.
- Focus on sections labeled "Breaking Changes", "Deprecations", and "Migration Guide".
- For GitHub-hosted projects, the Releases page often summarizes changes better than the commit log.
- Pay attention to versioning scheme: semver libraries may break APIs on major bumps; date-versioned libraries may break at any time.

## Anti-Patterns

- **Assuming methods exist** — Never call a method because you remember it from a blog post or a different version. Verify against the installed version.
- **Relying on code completion** — IDE completion may surface methods from a newer stub version than what is installed at runtime.
- **Using features from memory** — API signatures change. Always look up the current documentation for the exact version you are using.
- **Copying code from tutorials** — Tutorials often target a specific version that may differ from yours. Cross-reference with official docs.
- **Skipping the changelog on upgrades** — A minor version bump can deprecate or remove APIs you depend on. Always read the changelog.

## Cross-Reference

- For preferred library selection and recommended versions, see the `library-usage` skill (if available).
- For testing patterns that validate API compatibility, see the `testing-and-qa` skill.
