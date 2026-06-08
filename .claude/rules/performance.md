# Performance and Token Conservation Rules

## Model Selection Policy
- Use **Sonnet** for code generation, implementation, and routine editing tasks.
- Use **Opus** ONLY for complex architectural planning, system design, and high-level analysis.
- Use **Haiku** for simple classification, labeling, and trivial transformations.
- NEVER default to expensive models (Claude Opus, GPT-5) for tasks that Sonnet or Haiku can handle.
- NEVER exceed the token budget without explicit user approval.

## Token Conservation
- NEVER read massive build logs, database dumps, or raw minified vendor files repeatedly. This exhausts the context window.
- When debugging, use targeted `grep`/`ripgrep` searches instead of reading entire log files.
- ALWAYS exclude `**/*.log`, `**/*.min.js`, `**/node_modules/**`, `**/dist/**`, `**/package-lock.json`, `**/db_dump.sql` from the context window.
- NEVER read the same file multiple times without a clear reason. Cache your understanding.
- NEVER make the same shell command or test call repeatedly expecting different results without modifying underlying code.
- Monitor context window usage proactively. Run `/compact` at 60% usage. NEVER wait until context is critically full (80%+).
- If context exceeds 70%, immediately stop reading new files and compact before proceeding.

## Edit Efficiency
- NEVER replace an entire file when only a few lines need to change. Use surgical, targeted edits.
- When making changes, identify the exact lines to modify and change only those lines.
- NEVER create monolithic files that exceed single-file edit token limits. Keep files under 500 lines.

## Runtime Performance
- Implement proper pagination for large datasets. Never load everything into memory at once.
- Clean up event listeners in component unmount lifecycle methods.
- Use connection pooling for database connections.
- Implement proper caching strategies (Redis, in-memory) for frequently accessed data.
- Use batch processing for bulk database operations instead of individual queries.
- NEVER create N+1 query patterns. Use eager loading or batch queries.

## Redis Caching
- Use Redis connection pooling. NEVER create a new connection per request.
- ALWAYS set TTL on cache entries. NEVER allow unbounded cache growth.
- Implement cache invalidation on data mutations. Use pattern-based invalidation for related caches.
- Use Redis for semantic caching of LLM responses with embedding-based similarity matching.

## asyncio Performance
- Use `asyncio.Semaphore` to limit concurrent external API calls and prevent overwhelming downstream services.
- Implement backpressure patterns: if the queue is full, reject new requests rather than accumulating unbounded work.
- Use `asyncio.gather` for parallel independent operations to minimize total latency.

## pgvector Query Performance
- Use HNSW indexes for vector similarity queries. Tune `m` and `ef_construction` parameters based on dataset size.
- ALWAYS run `EXPLAIN ANALYZE` on vector queries to verify index usage. NEVER deploy without checking query plans.
- Consider IVFFlat indexes for very large datasets (>1M rows) where HNSW build time is prohibitive.

## FastAPI Performance
- ALWAYS use async route handlers. Sync handlers block the event loop.
- Use dependency caching for expensive-to-compute values (DB sessions, config objects).
- Minimize middleware overhead — each middleware adds latency to every request.

## AI-Specific Performance
- Implement semantic caching (e.g., Redis for embeddings) for repeated similar queries.
- Use request batching when making multiple LLM API calls.
- Implement dynamic model routing: cheap models for classification, expensive models for generation.
- Set up request timeouts and circuit breakers for all external API calls.
- Monitor token throughput (tokens/sec) and set up alerts for cost spikes.
- Implement response truncation to prevent excessive token consumption from verbose LLM outputs.
