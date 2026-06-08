# Fullstack Sync — Reference

## Frontend Change Propagation

When a frontend component is modified, the propagation path is:

1. **Component** — Identify which props/state fields changed.
2. **API call** — Trace the data source to the `fetch()`, `axios`, or API client call.
3. **Backend route** — Map the API call URL to the corresponding FastAPI route handler.
4. **DTO / schema** — Check the Pydantic model or response schema the route returns.
5. **Database column** — Verify the underlying SQLAlchemy column still exists and has the correct type.

If the frontend expects a field the backend no longer provides, the check fails at the component level. If the backend adds a field the frontend ignores, no error occurs but the UI is stale — flag these as warnings.

## Backend Change Propagation

When a backend route or Pydantic model is modified:

1. **Route handler** — Update path parameters, query parameters, or request/response schemas.
2. **Pydantic DTO** — Ensure `response_model` on the route matches the updated schema.
3. **TypeScript interface** — Update the matching frontend interface to reflect added/removed/renamed fields.
4. **Frontend components** — Update any component that consumes the changed fields.
5. **OpenAPI spec** — If the project auto-generates TypeScript types from the OpenAPI spec, re-run generation after the backend change.

## Database Schema Change Propagation

When an Alembic migration alters a table:

1. **Migration script** — Write the upgrade (and downgrade) in Alembic.
2. **SQLAlchemy ORM model** — Update the model class to match the new column definitions.
3. **Pydantic DTO** — Update any Pydantic model that maps to the altered table, including field types and optional/required status.
4. **API response** — Ensure the route's `response_model` reflects the DTO change.
5. **Frontend type** — Update the TypeScript interface and any component rendering the changed data.

Required rules:
- Every new nullable column must become an `Optional` field in the Pydantic DTO and a `field?: type` in TypeScript.
- A column rename must be propagated as a field rename through every layer — never leave the old name in any layer.
- A column type change (e.g., `String` to `Integer`) must update the Pydantic type annotation and the TypeScript type.

## Shared Type Synchronization

### Pydantic to TypeScript mapping

| Pydantic / Python type | TypeScript type |
|---|---|
| `str` | `string` |
| `int` | `number` |
| `float` | `number` |
| `bool` | `boolean` |
| `datetime` | `string` (ISO 8601) |
| `Optional[T]` | `T \| null` |
| `list[T]` | `T[]` |
| `dict[str, T]` | `Record<string, T>` |
| `Enum` class | union of literal strings or numbers |

### Zod schemas

When the frontend uses Zod for runtime validation, the Zod schema should mirror the Pydantic model field-for-field. Use `z.infer<typeof Schema>` to derive the TypeScript type from the Zod schema, keeping a single source of truth on the frontend side.

## Enum and Constant Propagation

Enums and constants that exist in both Python and TypeScript must be kept in sync manually or via code generation:

- **Python enum** — Define in a shared `enums.py` or `constants.py` module.
- **TypeScript enum or union** — Define in a corresponding `enums.ts` or `constants.ts` file.
- **Naming convention** — Use identical names and values across both languages. Prefer `SCREAMING_SNAKE` for constants and `PascalCase` for enum members.
- **Validation** — Run `scripts/check_sync.py --check types` after any enum change to detect mismatches.

## API Contract Testing Strategies

- **Schema-based tests** — Use Pydantic's `model_json_schema()` to generate the JSON Schema and compare it against the TypeScript interface definition.
- **Integration tests** — Hit the FastAPI endpoint with `httpx.TestClient` and assert the response shape matches what the frontend expects.
- **OpenAPI diff** — Export the OpenAPI spec before and after a change; diff the YAML/JSON to catch unintended breaking changes.
- **Snapshot testing** — Store a snapshot of each endpoint's response schema and fail the test suite if the snapshot changes without an explicit update.

## Cross-Skill References

- **`backend-and-db-patterns`** — For deeper patterns on SQLAlchemy models, Alembic migrations, and Pydantic DTO design.
- **`frontend-architecture`** — For React component patterns, state management, and TypeScript interface conventions.
- **`testing-and-qa`** — For contract-testing patterns and integration test scaffolding.
