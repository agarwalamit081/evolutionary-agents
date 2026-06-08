---
name: resource-check
description: Check for resource management issues including memory leaks, unclosed connections, and uncleaned listeners. Use during code review or after writing async/concurrent code.
---

# Resource Check Skill

## When to Use

- After writing async/concurrent code that opens connections, files, or subscribes to events.
- During code review to catch resource leaks before they reach production.
- When debugging unbounded memory growth or "too many open files" errors.
- Before merging PRs that touch I/O, networking, event emitters, or subscription logic.

## Core Categories

1. **Event Listeners** — `addEventListener`/`removeEventListener` pairing, emitter subscription teardown, pub/sub unsubscribe calls.
2. **File I/O** — context managers (`with` statements), `aiofiles`, ensuring file handles are closed on error paths.
3. **Network** — connection pooling, socket reuse, HTTP client shutdown, WebSocket close on disconnect.
4. **Memory** — unbounded caches, accumulating arrays/maps without eviction, closures capturing large objects.
5. **React-specific** — `useEffect` cleanup returns, `AbortController` abort on unmount, stale closure guards.
6. **Python-specific** — `__enter__`/`__exit__` and `contextlib`, asyncio task cancellation, thread/process pool shutdown.

## References

See `reference.md` for detailed patterns, anti-patterns, and language-specific rules. See `examples.md` for idiomatic code snippets.

## Scripts

- **`scripts/detect_leaks.py`** — Static analysis tool that scans Python and TypeScript files for common resource leak patterns. Run with `python scripts/detect_leaks.py --path <file-or-dir> [--language python|ts|auto]`.
