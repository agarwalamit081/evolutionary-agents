---
description: LangGraph Patterns Examples
---

**Example 1: State Definition with TypedDict and Graph Construction**

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import operator

class AgentState(TypedDict):
    query: str
    research_results: Annotated[list, operator.add]
    next_step: str

def research_node(state: AgentState):
    results = ["Fact 1", "Fact 2"]
    return {"research_results": results, "next_step": "analyze"}

def analyze_node(state: AgentState):
    analysis = f"Found {len(state['research_results'])} results"
    return {"next_step": "done"}

workflow = StateGraph(AgentState)
workflow.add_node("research", research_node)
workflow.add_node("analyze", analyze_node)
workflow.set_entry_point("research")
workflow.add_edge("research", "analyze")
workflow.add_edge("analyze", END)

app = workflow.compile(checkpointer=MemorySaver())
result = app.invoke({"query": "What is RAG?"}, config={"configurable": {"thread_id": "test-1"}})
```

---

**Example 2: Conditional Routing for Error Recovery**

```python
def route_after_tool(state: AgentState) -> str:
    if state.get("error"):
        return "handle_error"
    if state["next_step"] == "done":
        return END
    return "continue"

workflow.add_conditional_edges("tool_execution", route_after_tool, {
    "handle_error": "error_correction_node",
    "continue": "next_node",
    END: END,
})

def error_correction_node(state: AgentState):
    return {"error": None, "next_step": "retry"}
```

---

**Example 3: Tool-Calling Agent with create_react_agent**

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# Define tools
from langchain_core.tools import tool

@tool
def search_documents(query: str) -> str:
    """Search internal documents."""
    return f"Results for: {query}"

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

model = ChatOpenAI(model="gpt-4o-mini")
app = create_react_agent(model, [search_documents, calculate])

result = app.invoke({"messages": [("user", "Search for refund policy and calculate 15% of 200")]})
for msg in result["messages"]:
    print(f"{msg.type}: {msg.content[:100]}")
```

---

**Example 4: RAG Pipeline with Retrieval Grading**

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END

class RAGState(TypedDict):
    query: str
    documents: list[dict]
    graded_documents: list[dict]
    answer: str
    needs_rewrite: bool

def retrieve(state: RAGState):
    docs = vector_store.search(state["query"], k=5)
    return {"documents": docs}

def grade_documents(state: RAGState):
    graded = [d for d in state["documents"] if d["relevance_score"] > 0.7]
    return {
        "graded_documents": graded,
        "needs_rewrite": len(graded) == 0,
    }

def generate(state: RAGState):
    context = "\n".join(d["text"] for d in state["graded_documents"])
    answer = llm.invoke(f"Context: {context}\n\nQuestion: {state['query']}")
    return {"answer": answer}

def rewrite_query(state: RAGState):
    improved = llm.invoke(f"Rewrite for better retrieval: {state['query']}")
    return {"query": improved, "needs_rewrite": False}

def route_after_grading(state: RAGState):
    return "rewrite" if state["needs_rewrite"] else "generate"

graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve)
graph.add_node("grade", grade_documents)
graph.add_node("generate", generate)
graph.add_node("rewrite", rewrite_query)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "grade")
graph.add_conditional_edges("grade", route_after_grading)
graph.add_edge("rewrite", "retrieve")
graph.add_edge("generate", END)
```

---

**Example 5: Human-in-the-Loop with interrupt()**

```python
from langgraph.types import interrupt, Command

class EmailState(TypedDict):
    recipient: str
    subject: str
    body: str
    approved: bool

def compose_email(state: EmailState):
    return {"body": f"Draft email about {state['subject']}"}

def approval_node(state: EmailState):
    decision = interrupt(
        f"Approve sending email to {state['recipient']}?\n"
        f"Subject: {state['subject']}\n"
        f"Body: {state['body']}"
    )
    return {"approved": decision == "approve"}

def send_email(state: EmailState):
    if state["approved"]:
        # send_email_logic()
        return {"status": "sent"}
    return {"status": "cancelled"}

graph = StateGraph(EmailState)
graph.add_node("compose", compose_email)
graph.add_node("approve", approval_node)
graph.add_node("send", send_email)
graph.add_edge("compose", "approve")
graph.add_conditional_edges("approve", lambda s: "send" if s["approved"] else END)
graph.add_edge("send", END)

app = graph.compile(checkpointer=MemorySaver(), interrupt_before=["approve"])

# Usage: starts, pauses at approval
result = app.invoke(
    {"recipient": "user@example.com", "subject": "Update"},
    config={"configurable": {"thread_id": "email-1"}}
)
# Resume with approval
result = app.invoke(Command(resume="approve"), config={"configurable": {"thread_id": "email-1"}})
```

---

**Example 6: Multi-Agent Supervisor Pattern**

```python
from langgraph.graph import StateGraph, END, MessagesState

def supervisor(state: MessagesState):
    last_message = state["messages"][-1].content.lower()
    if "research" in last_message:
        return "researcher"
    elif "code" in last_message or "implement" in last_message:
        return "coder"
    elif "review" in last_message:
        return "reviewer"
    return END

def researcher(state: MessagesState):
    response = llm.invoke(f"You are a researcher. Answer: {state['messages'][-1].content}")
    return {"messages": [response]}

def coder(state: MessagesState):
    response = llm.invoke(f"You are a coder. Answer: {state['messages'][-1].content}")
    return {"messages": [response]}

def reviewer(state: MessagesState):
    response = llm.invoke(f"You are a reviewer. Answer: {state['messages'][-1].content}")
    return {"messages": [response]}

graph = StateGraph(MessagesState)
graph.add_node("supervisor", lambda s: None)  # Router only
graph.add_node("researcher", researcher)
graph.add_node("coder", coder)
graph.add_node("reviewer", reviewer)

graph.set_entry_point("supervisor")
graph.add_conditional_edges("supervisor", supervisor)
for agent in ["researcher", "coder", "reviewer"]:
    graph.add_edge(agent, END)

app = graph.compile()
```

---

**Example 7: Streaming Token-by-Token**

```python
async def stream_response(query: str):
    async for event in app.astream_events(
        {"messages": [("user", query)]},
        version="v2",
        config={"configurable": {"thread_id": "stream-1"}},
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            token = event["data"]["chunk"].content
            if token:
                yield token
        elif kind == "on_tool_start":
            print(f"[Calling tool: {event['name']}]")

# FastAPI endpoint
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(
        stream_response(request.message),
        media_type="text/event-stream",
    )
```
