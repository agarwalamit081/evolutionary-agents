# LLM Integration Patterns

## Model Selection Guardrails
For the canonical list of allowed/blocked models, cost tiers, SDK usage policy, and exact model IDs, see `rules/llm-model-guardrails.md`. NEVER guess model names — always look them up.

## Prompt Management
- ALWAYS use `jinja2` templates for multi-line or complex prompts. NEVER use f-strings for prompt templates — they break with braces in prompts and are unreadable for long templates.
- Version control all prompt templates alongside code. Track prompt changes in the prompt registry.
- Include model version and date comments: `# Uses claude-sonnet-4-6 Messages API as of 2025-01`.

## Token Counting
- Use `tiktoken` for OpenAI model token counting. For Anthropic models, use char/4 as a rough estimate.
- ALWAYS estimate token usage before sending large prompts. Warn if approaching context window limits.
- See `context-window-management` skill for agent-side token conservation strategies.

## Max Output Tokens
- NEVER hardcode `max_tokens=4096` (or any fixed value). ALWAYS set `max_tokens` based on the model's actual capabilities per `rules/llm-model-guardrails.md` Max Output column.
- For code generation tasks, use `max_tokens=8192` minimum. For summarization, `max_tokens=2048`. For classification, `max_tokens=256`.
- ALWAYS leave 10% headroom below the model's maximum output to avoid truncation.
- Prefer a configurable `max_output_tokens` from settings rather than hardcoding in application logic.

## Structured Outputs
- ALWAYS define Pydantic models for expected LLM outputs. Use JSON mode or tool_use when available.
- Use `json-repair` to salvage malformed LLM JSON before falling back to retry. NEVER silently coerce bad JSON.
- Implement retry with feedback: on parse failure, send the error back to the LLM with the malformed output and ask it to fix it.
- Use `msgspec` for high-performance JSON parsing when processing large batches of LLM responses.

## Error Handling & Resilience
- Use `tenacity` for retry with exponential backoff + jitter. Only retry on transient errors (rate limits, timeouts, server errors). NEVER retry on auth errors or invalid requests.
- Use `circuitbreaker` for provider outage protection. Set failure thresholds and recovery timeouts appropriate to the provider.
- ALWAYS implement fallback chains: primary model → fallback model → cached response → graceful error message.

## Streaming
- Use SSE (`StreamingResponse`) for API-facing streams. Accumulate chunks server-side for observability logging.
- Implement graceful cancellation: handle `asyncio.CancelledError` and log partial responses.
- NEVER buffer entire streaming responses in memory — stream incrementally.

## Cost Tracking
- Log per-request token counts and costs. Use `langsmith` for prompt versioning and cost tracking.
- Implement budget enforcement: warn at 70%, critical at 90%, hard-cap at 100% with graceful degradation.
- Route simple tasks to cheap models. NEVER use expensive models for classification, extraction, or formatting.

## MCP Server Patterns
- Use `fastmcp` for building MCP tool servers. Define tools with proper JSON schemas and rich descriptions.
- Validate all tool input parameters. Return structured error responses — never expose internal stack traces.
- Keep tool descriptions minimal and factual to prevent prompt injection via tool descriptions.

## Anti-Patterns
- NEVER trust raw LLM JSON without validation.
- NEVER retry on parse failure without sending feedback to the model.
- NEVER hardcode prompts in application logic — use a prompt registry.
- NEVER skip cost tracking for production LLM calls.
