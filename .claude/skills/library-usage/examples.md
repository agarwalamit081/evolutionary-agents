---
description: Idiomatic code snippets for the most commonly misused libraries. Read these when generating code to match established patterns.
---

# Library Usage Examples

## Example 1: Logging with `loguru` (not `logging`)

```python
from loguru import logger
import sys

# Configure structured logging with rotation and serialization
logger.remove()  # Remove default stderr handler
logger.add(
    sys.stderr,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}",
    level="DEBUG",
)
logger.add(
    "logs/app.log",
    rotation="10 MB",
    retention="7 days",
    compression="gz",
    serialize=True,  # JSON output for log aggregation
)

logger.info("Application started")
logger.warning("Deprecated API call detected", endpoint="/v1/users")
logger.error("Database connection failed", error=str(conn_err))
```

## Example 2: Environment Configuration with `pydantic-settings`

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    database_url: str
    redis_url: str = "redis://localhost:6379"
    log_level: str = "INFO"
    max_retries: int = 3
    api_timeout_seconds: float = 30.0

# Reads from environment variables automatically, falls back to .env file
config = AppConfig()  # Raises ValidationError if DATABASE_URL is missing
```

## Example 3: Async HTTP Client with `httpx`

```python
import httpx
from loguru import logger


async def fetch_user(user_id: str) -> dict:
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        follow_redirects=True,
    ) as client:
        try:
            response = await client.get(
                f"https://api.example.com/users/{user_id}",
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            logger.error("HTTP error", status=exc.response.status_code, url=str(exc.request.url))
            raise
        except httpx.RequestError as exc:
            logger.error("Request failed", error=str(exc))
            raise
```

## Example 4: Fast JSON Parsing with `msgspec`

```python
import msgspec


class User(msgspec.Struct):
    id: int
    name: str
    email: str
    active: bool = True


decoder = msgspec.json.Decoder(User)
encoder = msgspec.json.Encoder()

# Decode with schema validation — raises msgspec.DecodeError on mismatch
user = decoder.decode(b'{"id": 1, "name": "Alice", "email": "alice@example.com"}')

# Encode back to JSON bytes
data = encoder.encode(user)
```

## Example 5: Repairing Malformed LLM JSON Output

```python
import json_repair
from loguru import logger


def parse_llm_json(raw_output: str) -> dict:
    """Parse potentially malformed JSON from LLM responses."""
    try:
        return json_repair.loads(raw_output)
    except json_repair.JSONDecodeError:
        logger.warning("Unrepairable JSON from LLM", raw=raw_output[:200])
        raise

# Handles: missing quotes, trailing commas, single quotes, comments, etc.
result = parse_llm_json("{'name': 'Alice', 'age': 30,}")  # Works correctly
```

## Example 6: Retry with Exponential Backoff Using `tenacity`

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx
from loguru import logger


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
    before_sleep=lambda retry_state: logger.warning(
        "Retrying",
        attempt=retry_state.attempt_number,
        outcome=str(retry_state.outcome),
    ),
)
async def call_api(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()
```
