# LangChain & LangGraph Patterns

## Graph State Design
- ALWAYS use `TypedDict` with `Annotated` reducers for graph state. NEVER use raw `dict` or `dict[str, Any]`.
- Design state to be minimal — only include fields that nodes actually read or write. Bloated state wastes tokens and obscures intent.
- Use `operator.add` reducer for list accumulation fields (messages, artifacts).

## Node Design
- Nodes MUST be single-responsibility, pure functions that return partial state updates. NEVER mutate state in-place.
- Every node must have an explicit return type annotation matching the state TypedDict.
- Keep node logic testable: nodes should not directly call external APIs — inject clients via closure or config.

## Persistence
- Use `PostgresSaver` (via `langgraph-checkpoint-postgres`) for production state persistence. NEVER use `MemorySaver` in production.
- Configure TTL on checkpoints to prevent unbounded state growth in long-running workflows.
- Always test checkpoint serialization — Pydantic models and custom types must be JSON-serializable.

## Human-in-the-Loop
- Use `interrupt()` for approval checkpoints. Use `Command(resume=...)` for resumption after human input.
- NEVER allow autonomous execution of high-stakes actions (data deletion, financial transactions, email sending) without HITL.

## Tool Definition
- Use `@tool` decorator with proper type-annotated parameters. The docstring is the tool description the LLM sees — write it carefully.
- Every tool must have a docstring that explains: what it does, what each parameter means, and what it returns.

## Streaming
- Use `astream_events` with `version="v2"` for production streaming. NEVER use deprecated v1.
- Always handle `asyncio.CancelledError` in streaming handlers and clean up resources.

## Error Handling
- Add conditional edges to error correction nodes. NEVER allow errors to silently terminate a graph execution.
- Implement max-iteration guards on every iterative node. NEVER allow agents to loop indefinitely.
- Use `tenacity` for retry with exponential backoff on external API calls within nodes.

## Chain Patterns
- Use LCEL (`|` pipe operator) for composable chains. Keep chains readable — decompose complex chains into named runnables.
- NEVER use global mutable state in chains. Pass all configuration explicitly.

## Anti-Patterns
- NEVER allow unbounded state growth — implement cleanup nodes or TTL.
- NEVER skip error edges — every node that can fail must have a defined error path.
- NEVER hardcode model names or prompts in node functions — use configuration and prompt registries.
