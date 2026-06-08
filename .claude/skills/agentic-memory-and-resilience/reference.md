---
description: Agentic Memory and Resilience Reference
---

## Memory Architecture

### Short-Term Memory
- Current conversation messages (last N turns or token window).
- Fastest access, always in-context.
- Cleared on session end.

### Working Memory
- Rolling summary of conversation, compressed every K turns.
- Preserves key facts and decisions while saving tokens.
- Updated incrementally, not rebuilt from scratch.

### Long-Term Memory
- User preferences, facts, interaction history stored in DB/vector store.
- Update strategy: extract facts after each conversation, merge into user profile.
- Retrieval: fetch relevant memories via semantic search before each turn.
- Schema considerations: entity type, confidence, last_updated, source_conversation_id.

## Token Management

### Counting Strategies
- OpenAI: tiktoken (exact count per model family).
- Anthropic: anthropic tokenizer or character-based estimate (~4 chars/token).
- Always count before each LLM call, not after.

### Truncation Strategy (when approaching limit)
1. Summarize oldest messages into compact "context so far" block.
2. Remove low-relevance messages (keep: system prompt + recent + retrieved context + tool results).
3. As last resort, inform user of context reset.
4. **Never silently truncate** — always summarize first.

### Budget Allocation
- System prompt: 10-15% of context window.
- Retrieved context: 20-30%.
- Conversation history: 40-50%.
- Output buffer: 20-25%.

## Loop Prevention

### Detection Signals
- Same tool called with same args ≥ 3 times → break loop, ask user.
- Agent hasn't made progress after ≥ 10 tool calls → force summarization and replan.
- Cycling between 2-3 states → stuck detector triggers.

### Intervention Strategies
1. Inject a "you appear stuck" message into conversation.
2. Force the agent to summarize progress and replan.
3. Offer to escalate to user with current state.
4. As last resort, terminate with partial results.

## Error Recovery Patterns

| Failure | Strategy |
|---|---|
| Tool timeout | Retry once with backoff, fall back to simpler approach |
| API rate limit | Exponential backoff with jitter, queue for retry |
| Hallucinated tool name | Correct and retry, log for prompt improvement |
| Invalid tool output | Validate, retry with error feedback (max 2 retries) |
| Full pipeline failure | Return partial results with explanation |

## Idempotency Patterns

- Use idempotency keys for any mutation (write operations).
- Check-before-write: verify state before applying changes.
- Design tools to return same result on repeated calls (read-only where possible).
