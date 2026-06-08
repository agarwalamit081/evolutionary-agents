---
description: Python Patterns Examples
---

**Example 1: Async Batch Processing with Semaphore**

```python
import asyncio
from typing import Any

async def process_batch(
    items: list[str],
    handler: callable,
    max_concurrent: int = 10,
) -> list[Any]:
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited(item: str) -> Any:
        async with semaphore:
            return await handler(item)

    results = await asyncio.gather(*[limited(item) for item in items])
    return results

# Usage
async def fetch_url(url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.json()

results = await process_batch(urls, fetch_url, max_concurrent=5)
```

---

**Example 2: Generic Typed Repository Pattern**

```python
from typing import Generic, TypeVar, Protocol, Sequence
from dataclasses import dataclass

T = TypeVar("T")

class Repository(Protocol[T]):
    async def get(self, id: str) -> T | None: ...
    async def list(self, limit: int = 50) -> Sequence[T]: ...
    async def save(self, entity: T) -> T: ...
    async def delete(self, id: str) -> bool: ...

@dataclass
class User:
    id: str
    name: str
    email: str

class UserRepository:
    def __init__(self, db):
        self.db = db

    async def get(self, id: str) -> User | None:
        row = await self.db.fetchone("SELECT * FROM users WHERE id = $1", id)
        return User(**row) if row else None

    async def save(self, user: User) -> User:
        await self.db.execute(
            "INSERT INTO users (id, name, email) VALUES ($1, $2, $3) "
            "ON CONFLICT (id) DO UPDATE SET name=$2, email=$3",
            user.id, user.name, user.email
        )
        return user
```

---

**Example 3: Dataclass with Validation via __post_init__**

```python
from dataclasses import dataclass, field
import re

@dataclass(frozen=True, slots=True)
class EmailAddress:
    value: str
    _pattern = re.compile(r'^[\w.-]+@[\w.-]+\.\w+$')

    def __post_init__(self):
        if not self._pattern.match(self.value):
            raise ValueError(f"Invalid email: {self.value}")

    @property
    def domain(self) -> str:
        return self.value.split("@")[1]

# Usage
email = EmailAddress("user@example.com")  # OK
email = EmailAddress("invalid")           # Raises ValueError
```

---

**Example 4: Parameterized Retry Decorator**

```python
import functools
import asyncio
import random
from typing import Type, tuple

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        jitter = random.uniform(0, 0.5)
                        await asyncio.sleep(delay * (2 ** attempt) + jitter)
            raise last_error
        return wrapper
    return decorator

# Usage
@retry(max_attempts=3, delay=1.0, exceptions=(ConnectionError, TimeoutError))
async def fetch_data(url: str) -> dict:
    ...
```

---

**Example 5: Async Context Manager for Database Connections**

```python
from contextlib import asynccontextmanager
from typing import AsyncGenerator

@asynccontextmanager
async def database_transaction(pool) -> AsyncGenerator:
    async with pool.acquire() as conn:
        async with conn.transaction():
            yield conn

# Usage
async with database_transaction(db_pool) as conn:
    await conn.execute("INSERT INTO logs (message) VALUES ($1)", "Hello")
    result = await conn.fetch("SELECT * FROM logs")
```

---

**Example 6: Protocol-Based Dependency Injection**

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class CacheBackend(Protocol):
    async def get(self, key: str) -> str | None: ...
    async def set(self, key: str, value: str, ttl: int = 300) -> None: ...

class RedisCache:
    async def get(self, key: str) -> str | None:
        # Redis implementation
        ...
    async def set(self, key: str, value: str, ttl: int = 300) -> None:
        # Redis implementation
        ...

class UserService:
    def __init__(self, cache: CacheBackend):
        self.cache = cache  # Works with any CacheBackend

    async def get_user(self, user_id: str) -> dict:
        cached = await self.cache.get(f"user:{user_id}")
        if cached:
            return json.loads(cached)
        # Fetch from DB...
```

---

**Example 7: Pydantic Model with Validators**

```python
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

class CreateOrderRequest(BaseModel):
    customer_id: str = Field(pattern=r"^cust_[a-z0-9]+$")
    items: list[OrderItem] = Field(min_length=1)
    shipping_address: Address

    @field_validator("items")
    @classmethod
    def check_total_quantity(cls, items: list[OrderItem]) -> list[OrderItem]:
        total = sum(item.quantity for item in items)
        if total > 100:
            raise ValueError(f"Total quantity {total} exceeds limit of 100")
        return items

    model_config = {"frozen": True}
```

---

**Example 8: Generator Pipeline for Data Processing**

```python
import csv
from typing import Generator

def read_csv(filepath: str) -> Generator[dict, None, None]:
    with open(filepath) as f:
        reader = csv.DictReader(f)
        yield from reader

def filter_active(records: Generator[dict, None, None]) -> Generator[dict, None, None]:
    for record in records:
        if record["status"] == "active":
            yield record

def transform(records: Generator[dict, None, None]) -> Generator[dict, None, None]:
    for record in records:
        yield {
            "email": record["email"].lower().strip(),
            "name": record["name"].title(),
            "signup_date": record["created_at"][:10],
        }

# Compose pipeline (lazy — processes one record at a time)
pipeline = transform(filter_active(read_csv("users.csv")))

for user in pipeline:
    save_user(user)
```
