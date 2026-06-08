# Resource Check — Examples

## Example 1: Python Context Manager for Database Connections (SQLAlchemy)

```python
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine("postgresql://user:pass@localhost/db")
SessionLocal = sessionmaker(bind=engine)

@contextmanager
def get_db_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

# Usage
with get_db_session() as session:
    session.query(User).all()
```

## Example 2: Proper useEffect Cleanup for Event Listeners (React)

```tsx
useEffect(() => {
    const handleClick = (event: MouseEvent) => {
        console.log("Clicked:", event.target);
    };

    window.addEventListener("click", handleClick);

    // Cleanup: remove the SAME function reference
    return () => {
        window.removeEventListener("click", handleClick);
    };
}, []); // Empty deps = mount/unmount only
```

## Example 3: AbortController Pattern for Fetch Requests (React)

```tsx
useEffect(() => {
    const controller = new AbortController();

    async function fetchData() {
        try {
            const res = await fetch("/api/data", {
                signal: controller.signal,
            });
            const data = await res.json();
            setData(data);
        } catch (err) {
            if (err instanceof DOMException && err.name === "AbortError") {
                return; // Expected on unmount
            }
            console.error(err);
        }
    }

    fetchData();

    return () => {
        controller.abort(); // Cancel in-flight request on unmount
    };
}, [query]);
```

## Example 4: Redis Connection Pooling with Proper Shutdown

```python
import redis

# Create a shared pool at module/app level
pool = redis.ConnectionPool(
    host="localhost",
    port=6379,
    db=0,
    max_connections=10,
)
client = redis.Redis(connection_pool=pool)

def get_user_cache(key: str) -> str | None:
    return client.get(key)

# Application shutdown hook
def on_shutdown():
    pool.disconnect()
```

## Example 5: asyncio Task Cancellation on Shutdown

```python
import asyncio

async def background_worker():
    try:
        while True:
            await asyncio.sleep(1)
            print("working...")
    except asyncio.CancelledError:
        print("Worker cancelled, cleaning up...")
        raise

async def main():
    task = asyncio.create_task(background_worker())
    await asyncio.sleep(5)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass  # Clean shutdown

asyncio.run(main())
```

## Example 6: Using aiofiles with Context Managers for Async File Operations

```python
import aiofiles
import asyncio

async def write_log(path: str, message: str) -> None:
    async with aiofiles.open(path, mode="a") as f:
        await f.write(f"{message}\n")
    # File is automatically closed when exiting the async with block

async def read_config(path: str) -> str:
    async with aiofiles.open(path, mode="r") as f:
        return await f.read()
```
