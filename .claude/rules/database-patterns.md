# Database Patterns (PostgreSQL, SQLAlchemy, Alembic, Redis)

## SQLAlchemy ORM
- ALWAYS use declarative models with `Base = declarative_base()` or `DeclarativeBase`. NEVER write raw SQL strings.
- Use `Mapped` type annotations (SQLAlchemy 2.0 style) for all column definitions.
- For async: use `create_async_engine` with `asyncpg` driver and `async_sessionmaker`. NEVER use sync engine in async code.
- Always configure connection pooling: set `pool_size`, `max_overflow`, and `pool_timeout` appropriate to your deployment.

## Alembic Migrations
- ALWAYS run `alembic revision --autogenerate -m "descriptive_name"` then REVIEW the generated migration before applying.
- Name migrations descriptively: `<verb>_<target>_<detail>` (e.g., `add_embedding_dimension_column`).
- ALWAYS test both `upgrade` and `downgrade` paths before deploying.
- NEVER edit a migration that has already been applied to production — create a new migration instead.

## pgvector
- ALWAYS enable the extension: `CREATE EXTENSION IF NOT EXISTS vector;` and verify with `SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';`.
- Use HNSW indexes for most workloads: `CREATE INDEX ON table USING hnsw (column vector_cosine_ops);`.
- Tune HNSW parameters: `m` (connections per node, default 16) and `ef_construction` (build-time accuracy, default 64) based on dataset size.
- Use `EXPLAIN ANALYZE` to verify vector queries use the index. NEVER deploy without checking query plans.

## Query Optimization
- Use eager loading (`selectinload`, `joinedload`) to prevent N+1 queries. NEVER rely on lazy loading in API endpoints.
- Use `EXPLAIN ANALYZE` for any query that runs slower than 100ms.
- Implement pagination for all list endpoints. NEVER return unbounded result sets.

## Redis
- Use connection pooling with `redis.ConnectionPool`. NEVER create a new connection per request.
- ALWAYS set TTL on cache entries. NEVER allow unbounded cache growth.
- Implement cache invalidation on data mutations. Use pattern-based invalidation for related caches.
- Use Redis for: semantic caching (with TTL), rate limiting, session storage, and pub/sub for real-time updates.

## Transaction Management
- Always use context managers for sessions: `async with AsyncSessionLocal() as session:`.
- Commit at the end of the business operation. Rollback on any exception.
- NEVER hold transactions open across external API calls — they tie up connection pool resources.
