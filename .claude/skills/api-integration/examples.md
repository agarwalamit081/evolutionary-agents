---
description: API Integration Examples
---

**Example 1: SSE Stream Parser**

```python
import json
from typing import AsyncIterator

async def parse_sse_stream(response: AsyncIterator[bytes]) -> AsyncIterator[dict]:
    """Parse Server-Sent Events from an HTTP streaming response."""
    buffer = ""
    async for chunk in response:
        buffer += chunk.decode("utf-8")
        while "\n\n" in buffer:
            event, buffer = buffer.split("\n\n", 1)
            for line in event.split("\n"):
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        return
                    yield json.loads(data)
```

---

**Example 2: Provider-Abstracted LLM Client Interface**

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator

class LLMClient(ABC):
    @abstractmethod
    async def chat(self, messages: list[dict], model: str = None, **kwargs) -> str: ...

    @abstractmethod
    async def stream(self, messages: list[dict], model: str = None, **kwargs) -> AsyncIterator[str]: ...

class OpenAIClient(LLMClient):
    def __init__(self, api_key: str):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=api_key)

    async def chat(self, messages, model="gpt-4o-mini", **kwargs):
        response = await self.client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
        return response.choices[0].message.content

    async def stream(self, messages, model="gpt-4o-mini", **kwargs):
        stream = await self.client.chat.completions.create(
            model=model, messages=messages, stream=True, **kwargs
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

class AnthropicClient(LLMClient):
    def __init__(self, api_key: str):
        from anthropic import AsyncAnthropic
        self.client = AsyncAnthropic(api_key=api_key)

    async def chat(self, messages, model="claude-sonnet-4-6", **kwargs):
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        user_messages = [m for m in messages if m["role"] != "system"]
        response = await self.client.messages.create(
            model=model, max_tokens=4096, system=system, messages=user_messages, **kwargs
        )
        return response.content[0].text

    async def stream(self, messages, model="claude-sonnet-4-6", **kwargs):
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        user_messages = [m for m in messages if m["role"] != "system"]
        async with self.client.messages.stream(
            model=model, max_tokens=4096, system=system, messages=user_messages, **kwargs
        ) as stream:
            async for text in stream.text_stream:
                yield text
```

---

**Example 3: OpenAI Streaming with Cancellation**

```python
import asyncio
from openai import AsyncOpenAI

async def stream_with_timeout(prompt: str, timeout: float = 30.0):
    client = AsyncOpenAI()
    try:
        stream = await asyncio.wait_for(
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            ),
            timeout=timeout,
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except asyncio.TimeoutError:
        yield "[Response timed out]"
```

---

**Example 4: Anthropic Streaming with tool_use Events**

```python
from anthropic import AsyncAnthropic

async def stream_with_tools(query: str):
    client = AsyncAnthropic()
    async with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        tools=[search_tool],
        messages=[{"role": "user", "content": query}],
    ) as stream:
        async for event in stream:
            if event.type == "content_block_delta":
                if event.delta.type == "text_delta":
                    yield event.delta.text
            elif event.type == "content_block_start":
                if event.content_block.type == "tool_use":
                    print(f"Tool call: {event.content_block.name}")
```

---

**Example 5: Rate Limit Tracker and Backoff**

```python
import asyncio
import time

class RateLimitTracker:
    def __init__(self, requests_per_minute: int = 60):
        self.rpm = requests_per_minute
        self.timestamps: list[float] = []

    async def acquire(self):
        now = time.time()
        self.timestamps = [t for t in self.timestamps if now - t < 60]

        if len(self.timestamps) >= self.rpm:
            wait = 60 - (now - self.timestamps[0]) + 0.1
            await asyncio.sleep(wait)

        self.timestamps.append(time.time())

# Usage
tracker = RateLimitTracker(requests_per_minute=50)
async for query in queries:
    await tracker.acquire()
    result = await call_llm(query)
```

---

**Example 6: Error Handling Wrapper with Retry**

```python
import asyncio
import random
from functools import wraps

RETRYABLE_ERRORS = {429, 500, 502, 503, 504}

def with_retry(max_retries: int = 3, base_delay: float = 1.0):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    status = getattr(e, "status_code", None)
                    if status and status not in RETRYABLE_ERRORS:
                        raise
                    last_error = e
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                    await asyncio.sleep(delay)
            raise last_error
        return wrapper
    return decorator
```

---

**Example 7: Async Batch Request Handler**

```python
import asyncio

async def batch_requests(
    items: list[str], handler, batch_size: int = 5, delay_between: float = 0.5
) -> list:
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        batch_results = await asyncio.gather(
            *[handler(item) for item in batch], return_exceptions=True
        )
        results.extend(batch_results)
        if i + batch_size < len(items):
            await asyncio.sleep(delay_between)
    return results
```

---

**Example 8: Streaming Response to HTTP (FastAPI)**

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/chat/stream")
async def stream_chat(request: ChatRequest):
    async def generate():
        async for token in llm_client.stream(
            messages=[{"role": "user", "content": request.message}],
            model=request.model,
        ):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```
