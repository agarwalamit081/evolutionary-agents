---
description: Backend Patterns Examples
---

**Example 1: TypeScript/Express Controller-Service-Repository**

```typescript
// controller.ts
import { Request, Response } from 'express';
import { UserService } from './service';

export class UserController {
  constructor(private service: UserService) {}

  async createUser(req: Request, res: Response) {
    const { email, name } = req.body;
    if (!email || !name) return res.status(400).json({ type: 'validation_error', detail: 'email and name required' });

    const user = await this.service.createUser({ email, name });
    res.status(201).json(user);
  }
}

// service.ts
import { UserRepository } from './repository';

export class UserService {
  constructor(private repo: UserRepository) {}

  async createUser(data: { email: string; name: string }) {
    const existing = await this.repo.findByEmail(data.email);
    if (existing) throw new ConflictError('Email already registered');
    return this.repo.save(data);
  }
}
```

---

**Example 2: Python/FastAPI with Dependency Injection**

```python
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr

class CreateUserRequest(BaseModel):
    email: EmailStr
    name: str

class UserResponse(BaseModel):
    id: str
    email: str
    name: str

router = APIRouter()

@router.post("/users", response_model=UserResponse, status_code=201)
async def create_user(request: CreateUserRequest, service: UserService = Depends()):
    existing = await service.get_by_email(request.email)
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")
    user = await service.create_user(request.email, request.name)
    return user
```

---

**Example 3: Global Error Handler Middleware (RFC 7807)**

```python
from fastapi import Request
from fastapi.responses import JSONResponse
import uuid

async def error_handler(request: Request, call_next):
    trace_id = str(uuid.uuid4())
    request.state.trace_id = trace_id

    try:
        return await call_next(request)
    except ValidationError as e:
        return JSONResponse(status_code=400, content={
            "type": "validation_error", "title": "Validation Error",
            "status": 400, "detail": str(e), "trace_id": trace_id,
        })
    except NotFoundError as e:
        return JSONResponse(status_code=404, content={
            "type": "not_found", "title": "Resource Not Found",
            "status": 404, "detail": str(e), "trace_id": trace_id,
        })
    except Exception as e:
        logger.error(f"Unhandled error: {e}", trace_id=trace_id)
        return JSONResponse(status_code=500, content={
            "type": "internal_error", "title": "Internal Server Error",
            "status": 500, "trace_id": trace_id,
        })
```

---

**Example 4: Background Job with Redis Queue**

```python
import redis
import json
from datetime import datetime

r = redis.Redis()

def enqueue_job(queue: str, task: str, payload: dict, idempotency_key: str = None):
    if idempotency_key and r.exists(f"job:{idempotency_key}"):
        return r.get(f"job:{idempotency_key}")

    job = {
        "task": task, "payload": payload,
        "idempotency_key": idempotency_key,
        "enqueued_at": datetime.now().isoformat(),
        "attempts": 0, "max_attempts": 3,
    }
    r.rpush(f"queue:{queue}", json.dumps(job))

async def process_job(queue: str):
    _, raw = r.blpop(f"queue:{queue}", timeout=30)
    job = json.loads(raw)
    try:
        result = await execute_task(job["task"], job["payload"])
        if job["idempotency_key"]:
            r.setex(f"job:{job['idempotency_key']}", 3600, json.dumps(result))
    except Exception:
        job["attempts"] += 1
        if job["attempts"] < job["max_attempts"]:
            r.rpush(f"queue:{queue}", json.dumps(job))
        else:
            r.rpush("queue:dead_letter", json.dumps(job))
```

---

**Example 5: Rate Limiter Middleware**

```python
import time
from collections import defaultdict

class RateLimiter:
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> tuple[bool, int]:
        now = time.time()
        self.requests[key] = [t for t in self.requests[key] if now - t < self.window]
        if len(self.requests[key]) >= self.max_requests:
            retry_after = int(self.window - (now - self.requests[key][0])) + 1
            return False, retry_after
        self.requests[key].append(now)
        return True, 0
```

---

**Example 6: Event Publishing Pattern**

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DomainEvent:
    event_type: str
    aggregate_id: str
    payload: dict
    timestamp: datetime

class EventBus:
    def __init__(self):
        self.handlers: dict[str, list] = {}

    def subscribe(self, event_type: str, handler):
        self.handlers.setdefault(event_type, []).append(handler)

    async def publish(self, event: DomainEvent):
        for handler in self.handlers.get(event.event_type, []):
            await handler(event)

# Usage
bus = EventBus()
bus.subscribe("order.created", lambda e: send_confirmation_email(e.payload))
bus.subscribe("order.created", lambda e: update_analytics(e.payload))
```

---

**Example 7: Repository Pattern with Type-Safe Queries**

```python
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Sequence

T = TypeVar("T")

class Repository(ABC, Generic[T]):
    @abstractmethod
    async def get_by_id(self, id: str) -> T | None: ...
    @abstractmethod
    async def list(self, limit: int = 50, offset: int = 0) -> Sequence[T]: ...
    @abstractmethod
    async def save(self, entity: T) -> T: ...
    @abstractmethod
    async def delete(self, id: str) -> bool: ...

class PostgresUserRepository(Repository[User]):
    def __init__(self, db):
        self.db = db

    async def get_by_id(self, id: str) -> User | None:
        row = await self.db.fetchone("SELECT * FROM users WHERE id = $1", id)
        return User(**row) if row else None

    async def save(self, user: User) -> User:
        await self.db.execute(
            "INSERT INTO users (id, email, name) VALUES ($1, $2, $3) "
            "ON CONFLICT (id) DO UPDATE SET email=$2, name=$3",
            user.id, user.email, user.name,
        )
        return user
```
