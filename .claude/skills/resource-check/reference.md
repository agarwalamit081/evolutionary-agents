# Resource Check — Reference

## Event Listener and Subscription Lifecycle

- Every `addEventListener` must have a corresponding `removeEventListener` with the same reference (not an inline arrow function).
- Use `{ once: true }` for one-shot listeners when appropriate.
- In pub/sub systems, store the subscription object and call `.unsubscribe()` or `.off()` in cleanup.
- EventEmitter listeners in Node.js should be removed on component teardown or scope exit to prevent reference retention.

## File and I/O Resource Management

- **Python**: Always use `with` statements or `contextlib.closing()` for file handles, sockets, and subprocesses.
- **Async Python**: Use `async with` with `aiofiles` or `anyio` path objects; avoid bare `open()` in async code.
- **TypeScript/Node.js**: Use `fs.promises` with try/finally or stream pipelines (`stream.pipeline()`). Destroy streams on error.

## Network Connection Management

- Reuse HTTP client instances across requests (e.g., `httpx.Client`, `requests.Session`, `axios` instances) to leverage connection pooling.
- Always close clients in a `finally` block or context manager.
- WebSocket connections must have explicit close logic in error and success paths.
- Set reasonable pool sizes and timeouts to prevent connection exhaustion under load.

## Memory Management

- Avoid unbounded caches; use LRU/LFU eviction (`functools.lru_cache`, `lru-cache` npm package) with size limits.
- Do not accumulate data in lists/maps inside long-running loops or event handlers without bounds.
- Closures that capture large objects (e.g., request bodies, file buffers) prevent GC; null out references when done.
- Weak references (`WeakRef`, `weakref`) are appropriate for caches that should not prevent GC.

## React-Specific

- Every `useEffect` that creates a subscription, timer, or listener must return a cleanup function.
- Use `AbortController` for fetch requests; abort on unmount to prevent state updates on unmounted components.
- Guard against stale closures by using refs (`useRef`) for values read inside long-lived callbacks.
- `setInterval` inside `useEffect` must be cleared with `clearInterval` in the cleanup return.

## Python-Specific

- Implement context managers via `contextlib.contextmanager` or `__enter__`/`__exit__` for resources.
- Cancel asyncio tasks on shutdown: gather pending tasks, cancel them, and await cancellation with a timeout.
- Shut down `ThreadPoolExecutor` and `ProcessPoolExecutor` explicitly; use them as context managers when possible.
- Use `contextlib.AsyncExitStack` for managing multiple async resources in a single scope.

## LLM API Client Reuse

- Instantiate LLM clients (OpenAI, Anthropic, etc.) once at the application level, not per request.
- Pass the client instance via dependency injection rather than creating new connections.
- Close/teardown the client during application shutdown hooks.

## Redis Connection Management

- Use `redis-py` connection pooling (`redis.ConnectionPool`) rather than creating new connections per operation.
- Set `max_connections` appropriate to your workload to prevent pool exhaustion.
- Close the pool during application shutdown: `pool.disconnect()` or let the client context manager handle it.
- In async Redis (`redis.asyncio`), use `await client.aclose()` in shutdown hooks.

## SQLAlchemy Session Lifecycle

- Use scoped sessions or session factories; do not create a new `Session()` per operation without closing.
- Always close sessions in a `finally` block: `session.close()` or use a context manager (`with Session() as session:`).
- Expire or refresh long-lived sessions to prevent stale data and unbounded identity map growth.
- In async SQLAlchemy, use `async with AsyncSession()` and ensure `await session.close()` runs on all exit paths.
