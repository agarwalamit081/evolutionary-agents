---
description: Backend and Database Patterns Reference
---

## Data Types

- **Identifiers**: `UUID` (preferred for distributed systems) or `BIGSERIAL` (strict sequential ordering).
- **Timestamps**: Always `TIMESTAMPTZ` (PostgreSQL) to avoid timezone ambiguity.
- **Text**: Use `VARCHAR(n)` only for strict known limits (country codes). Otherwise `TEXT`.
- **Money**: `NUMERIC` or `DECIMAL`. NEVER `FLOAT` or `REAL`.

## Indexing Strategy

- Default to B-Tree for equality and range queries.
- Use GIN indexes for `JSONB` columns or full-text search.
- Create composite indexes for frequent multi-column `WHERE` clauses, ordering by cardinality (highest selectivity first).
- Create partial indexes for common filtered queries (e.g., `WHERE deleted_at IS NULL`).

## Audit & Compliance

- Consider `created_by` and `updated_by` (UUID) for strict audit trails.
- Mark PII columns in comments for future encryption/masking routines.

## Zero-Downtime Migration: Expand and Contract Pattern

When renaming a column or changing a type, never do it in one step:
1. **Expand**: Add the new column. Deploy app code that writes to both old and new columns.
2. **Backfill**: Background script copies data from old to new in batches.
3. **Switch**: Deploy app code that reads/writes only the new column.
4. **Contract**: Remove old column in a subsequent migration.

## PostgreSQL Specifics

- Use `CREATE INDEX CONCURRENTLY` to avoid locking the table for writes.
- Adding `NOT NULL` column to existing table: (1) Add nullable, (2) Backfill in batches, (3) `SET NOT NULL`, (4) Add default.
- Use `EXISTS` instead of `IN` for subqueries checking row existence.
- Use `ILIKE` carefully — prevents B-Tree usage. Consider `pg_trgm` + GIN for case-insensitive search.
- Keyset pagination (`WHERE id > last_id`) over `OFFSET` for large result sets.
- `JSONB` over `JSON` for binary storage and indexing.
- Assume PgBouncer for connection pooling. Avoid long transactions and `PREPARE`.

## Query Optimization

- Use `EXPLAIN ANALYZE` for new complex queries.
- Use `RETURNING` clause for inserts/updates to avoid follow-up SELECTs.
- CTEs (`WITH`) for readability; CTEs over subqueries.
