---
name: fullstack-sync
description: Ensure frontend-backend consistency when making changes. Use after modifying UI components, API routes, database schemas, or shared types to prevent disconnects.
---

# Fullstack Sync

## When to Use

- After adding, renaming, or removing fields in a Pydantic model, SQLAlchemy table, or TypeScript interface.
- After creating or modifying API endpoints (FastAPI routes) and their corresponding frontend fetch/axios calls.
- After running database migrations that alter column names, types, or constraints.
- After changing shared enums, constants, or configuration values that span both frontend and backend.

## Core Workflow

**Frontend changes:** When a React component is updated, trace the change backward through its props/interface to the API call, then to the backend route, DTO, and database column. Ensure every field consumed in the UI is supplied by the backend.

**Backend changes:** When a FastAPI route or Pydantic schema changes, propagate forward: update the response DTO, verify the TypeScript interface matches, and update any frontend components that consume the changed fields.

**Database changes:** When an Alembic migration alters a table, update the SQLAlchemy ORM model, then the Pydantic DTO, then the TypeScript type, then any frontend components rendering that data.

**Type changes:** When a shared type (enum, constant, interface) is modified, use `scripts/check_sync.py` to verify the change is reflected across Python, TypeScript, and any generated OpenAPI artifacts.

## References

- See `reference.md` for detailed propagation rules, contract-testing strategies, and cross-skill references.
- See `examples.md` for end-to-end sync patterns covering common change scenarios.

## Scripts

- **`scripts/check_sync.py`** — Scans backend Python files and frontend TypeScript files to detect type mismatches, missing fields, and orphaned API endpoints. Run after any cross-stack change.
  - `--backend-dir` — path to the backend source directory
  - `--frontend-dir` — path to the frontend source directory
  - `--check types|endpoints|all` — scope of the check (default: `all`)
