---
description: API Integration Reference
---

## SSE (Server-Sent Events) Format

```
data: {"id":"msg_001","choices":[{"delta":{"content":"Hello"}}]}\n\n
data: {"id":"msg_001","choices":[{"delta":{"content":" world"}}]}\n\n
data: [DONE]\n\n
```

- Each event prefixed with `data: `, terminated with `\n\n`.
- Parse with: split on `\n\n`, strip `data: `, parse JSON.
- `data: [DONE]` signals stream end.

## Provider Streaming Formats

| Provider | Event Format | Content Path |
|---|---|---|
| OpenAI | `choices[0].delta.content` | `delta.content` |
| Anthropic | `content_block.delta.text` (type: text) or `content_block_delta` | `delta.text` |
| Google Gemini | `candidates[0].content.parts[0].text` | Nested parts |

## Provider Abstraction Layer

```python
class LLMProvider(Protocol):
    async def chat(self, messages: list[dict], **kwargs) -> str: ...
    async def stream(self, messages: list[dict], **kwargs) -> AsyncIterator[str]: ...
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
```

## Rate Limit Handling

- Track from response headers: `x-ratelimit-remaining`, `retry-after`.
- When `remaining` < 5: preemptively slow down or queue.
- On 429: parse `retry-after` header, sleep that duration + jitter.
- Implement per-provider rate limit counters for multi-provider setups.

## Error Code Mapping

| Provider Error | Common Code | Response |
|---|---|---|
| Rate limit | 429 | Backoff + retry |
| Context length exceeded | 400 | Truncate or summarize input |
| Auth failure | 401/403 | Alert, do not retry |
| Server error | 500/502/503 | Retry with backoff |
| Timeout | N/A | Retry once, then fallback |

## Connection Pooling

- Reuse HTTP client instances (httpx.AsyncClient, openai.AsyncOpenAI).
- Set connection pool limits: `max_connections=100`, `max_keepalive=20`.
- Close clients on shutdown (use async context manager).
