---
name: backend-and-db-patterns
description: Database schema design, migration patterns, and PostgreSQL best practices — schema standards, zero-downtime migrations, query optimization, and connection management.
---

**When to Use**
- Designing new database tables or modifying existing ones.
- Reviewing ORM models (SQLAlchemy, Prisma, TypeORM).
- Creating, reviewing, or modifying database migration files.
- Writing raw SQL queries, functions, or views.
- Optimizing slow queries or designing database interactions.
- Discussing zero-downtime deployments or schema evolution.

**Core Principles**
1. **Standard Columns**: Every table MUST have `id` (UUID or BigInt), `created_at` (TIMESTAMPTZ), and `updated_at` (TIMESTAMPTZ).
2. **Naming**: `snake_case` for tables/columns. Pluralize table names (e.g., `user_profiles`).
3. **Constraints**: Enforce integrity at the database level (NOT NULL, UNIQUE, CHECK, FK), not just in app logic.
4. **Soft Deletes**: Use `deleted_at` (TIMESTAMPTZ, nullable) for critical business entities.
5. **Backward Compatibility**: Migrations must not break the currently running application.
6. **Non-Blocking**: Avoid exclusive locks on large tables. Use `CONCURRENTLY` for indexes.
7. **Reversible**: Every `up` must have a safe `down`.
8. **Idempotent**: Use `IF NOT EXISTS` — safe to run multiple times.
9. **No `SELECT *`**: Explicitly list columns to prevent breaking changes and reduce payload.
10. **Parameterized Queries**: ALWAYS prevent SQL injection. Never concatenate SQL values.

**References**
- Load `reference.md` for data types, indexing strategy, migration patterns, query optimization, and connection pooling.
- Load `examples.md` for schema designs, migrations, and query patterns.

**Scripts**
- `scripts/validate_schema.py`: Validate SQL files for standard columns and anti-patterns.
- `scripts/generate_migration_scaffold.sh`: Generate timestamped migration files with safety headers.
- `scripts/explain_analyze_wrapper.py`: Conceptual scaffold for EXPLAIN ANALYZE.
