---
description: LangGraph Patterns Reference
---

## Graph Types

| Type | When to Use |
|---|---|
| `StateGraph` | Custom state schema, fine-grained control |
| `MessageGraph` | Chat-focused, messages as state (auto-handles `add_messages`) |
| `create_react_agent` | Quick tool-calling agent without custom graph structure |

## State Design Patterns

- **TypedDict**: `class AgentState(TypedDict): messages: Annotated[list, add_messages]` — most common.
- **Pydantic BaseModel**: `class AgentState(BaseModel): ...` — adds runtime validation.
- **Reducers**: `Annotated[list, operator.add]` for append, `add_messages` for chat history.
- **Keep state minimal**: Store references/IDs, not full documents. State is serialized on every checkpoint.

## Edge Types

- **Normal edge**: `graph.add_edge("node_a", "node_b")` — unconditional transition.
- **Conditional edge**: `graph.add_conditional_edges("node_a", router_fn, {"path_1": "node_b", "path_2": "node_c"})` — route based on state.
- **Entry point**: `graph.set_entry_point("first_node")`.

## Streaming

| Method | Granularity | Use When |
|---|---|---|
| `astream` | Node-by-node | Need intermediate state snapshots |
| `astream_events` | Token-level | Real-time UI updates, SSE responses |
| `astream_log` | Detailed with metadata | Debugging and tracing |

```python
# Token-level streaming
async for event in app.astream_events(input, version="v2"):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="", flush=True)
```

## Persistence

| Checkpointer | Use Case | Setup |
|---|---|---|
| `MemorySaver` | Development, testing | `app = graph.compile(checkpointer=MemorySaver())` |
| `SqliteSaver` | Single-instance prod | Requires `aiosqlite` |
| `PostgresSaver` | Production, multi-instance | Requires `asyncpg` + pgvector setup |

**Thread IDs**: Must be unique per user session or workflow instance. Use `config={"configurable": {"thread_id": "user-123"}}`.

## Human-in-the-Loop

```python
# Pause execution for human approval
from langgraph.types import interrupt, Command

def approval_node(state):
    decision = interrupt("Approve this action?")
    if decision == "yes":
        return {"status": "approved"}
    return {"status": "rejected"}

# Resume with user decision
app.invoke(Command(resume="yes"), config=config)
```

## Testing LangGraph Apps

- **Mock LLM calls**: Replace real LLM with deterministic fake for unit tests.
- **Test state transitions**: Invoke graph, assert on intermediate state at each node.
- **Test conditional routing**: Provide inputs that trigger each branch, verify correct path.
- **Test tool execution**: Mock tool responses, verify tool is called with correct args.

## Anti-Patterns

- Mutable state in nodes (modify and return new dict, don't mutate input).
- Storing large payloads in state (use references/IDs).
- Missing error edges (every node should have a failure path).
- Unbounded retries (cap at 3, escalate to human).
