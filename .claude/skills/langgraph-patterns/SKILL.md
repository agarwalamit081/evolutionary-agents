---
name: langgraph-patterns
description: LangGraph patterns for multi-agent systems and stateful LLM workflows — graph design, state management, streaming, human-in-the-loop, and tool-calling agents.
---

**When to Use**
- Building multi-agent systems or stateful LLM workflows with LangGraph.
- Refactoring LangChain chains into graph-based workflows.
- Implementing human-in-the-loop approval, error recovery, or conditional routing.
- Designing RAG pipelines with retrieval grading and query rewriting.

**Core Principles**
1. **Explicit State**: Define graph state with `TypedDict` or Pydantic `BaseModel`. Never untyped dicts.
2. **Modular Nodes**: Each node = single responsibility. Separate LLM calls from state logic.
3. **Human-in-the-Loop**: Use `interrupt()` for critical actions (email, payment, DB write).
4. **Error Recovery**: Conditional edges route to retry/correction nodes on failures.
5. **Streaming First**: Use `astream_events` for token-level streaming in production.
6. **Subgraph Composition**: Break complex graphs into reusable subgraphs.

**Workflow**
1. Define the state schema first (TypedDict with Annotated reducers).
2. Identify nodes (each = one tool/LLM call or state transformation).
3. Define edges (normal + conditional routing).
4. Add persistence (MemorySaver for dev, PostgresSaver for prod).
5. Test state transitions with mock LLM calls.
6. Add streaming for production.

**References**
- Load `reference.md` for graph types, state design, streaming, persistence, and testing patterns.
- Load `examples.md` for react agents, RAG pipelines, HITL, and supervisor patterns.

**Scripts**
- `scripts/visualize_graph.py`: Generate Mermaid diagram from a compiled graph.
