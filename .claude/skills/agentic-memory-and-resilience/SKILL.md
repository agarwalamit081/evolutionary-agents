---
name: agentic-memory-and-resilience
description: Robust agent state management and error recovery — short-term memory, long-term memory, token-limit truncation, infinite loop prevention, and graceful degradation.
---

**When to Use**
- Building agents with persistent or multi-turn memory.
- Managing conversation context length and token budgets.
- Handling agent failures, timeouts, or infinite tool-calling loops.
- Implementing user profile updates from conversations.

**Core Principles**
1. **Layered Memory**: Short-term (current conversation), working (summarized context), long-term (persistent user data in DB/vector store).
2. **Token Budget Awareness**: Track token usage. Truncate/summarize before hitting limits, not after.
3. **Graceful Degradation**: When tools fail, provide fallback behavior rather than crashing.
4. **Loop Detection**: Detect and break infinite tool-calling loops (max retries, duplicate action detection, progress checks).
5. **Idempotent Recovery**: Design agent actions to be safely retryable without side effects.

**References**
- Load `reference.md` for memory architecture, token management, loop prevention, and error recovery patterns.
- Load `examples.md` for memory managers, loop detectors, and fallback implementations.

**Scripts**
- `scripts/token_counter.py`: Count tokens for message lists (tiktoken + anthropic).
- `scripts/summarize_context.py`: Compress conversation history into summaries.
