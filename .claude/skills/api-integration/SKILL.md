---
name: api-integration
description: LLM API integration patterns — streaming responses (SSE), REST patterns, rate limiting, error handling, and provider-agnostic abstraction.
---

**When to Use**
- Integrating with LLM APIs (OpenAI, Anthropic, Google).
- Implementing streaming responses (SSE).
- Building provider-agnostic LLM client wrappers.
- Handling rate limits, retries, and error mapping.

**Core Principles**
1. **Always Stream for User-Facing**: Use streaming (SSE) for user-facing output to reduce perceived latency.
2. **Provider Abstraction**: Design a common interface (chat, stream, embed) wrapping provider-specific SDKs.
3. **Rate Limit Awareness**: Track limits from response headers. Implement backoff before hitting limits.
4. **Structured Errors**: Map provider-specific errors to a common error taxonomy.
5. **Timeout Management**: 30s default for non-streaming, 120s for streaming.

**References**
- Load `reference.md` for SSE format, provider APIs, rate limit handling, and error mapping.
- Load `examples.md` for streaming parsers, provider clients, and FastAPI integration.

**Scripts**
- `scripts/test_streaming.py`: Quick test for streaming LLM API responses.
