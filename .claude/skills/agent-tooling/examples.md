---
description: Agent Tooling Examples
---

**Example 1: Complete Tool Definition (Anthropic Format)**

```python
search_tool = {
    "name": "search_documents",
    "description": "Search internal documents by query. Returns top-K matching chunks with relevance scores. Use when the user asks about policies, procedures, or factual information from internal sources.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query"
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (1-20)",
                "default": 5,
                "minimum": 1,
                "maximum": 20
            },
            "category": {
                "type": "string",
                "description": "Filter by document category",
                "enum": ["policy", "technical", "hr", "finance", "all"],
                "default": "all"
            }
        },
        "required": ["query"]
    }
}
```

---

**Example 2: Python Tool with Input Validation and Error Handling**

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class ToolResult:
    status: Literal["success", "error", "partial"]
    data: dict | None = None
    error: str | None = None
    retry_hint: str | None = None

async def search_documents(query: str, top_k: int = 5, category: str = "all") -> ToolResult:
    # Input validation
    if not query or len(query.strip()) < 3:
        return ToolResult(
            status="error",
            error="Query must be at least 3 characters",
            retry_hint="Provide a more specific search query."
        )

    if top_k < 1 or top_k > 20:
        return ToolResult(status="error", error="top_k must be between 1 and 20")

    try:
        results = await vector_store.search(query, k=top_k, filter={"category": category})
        return ToolResult(status="success", data={"results": results, "count": len(results)})
    except ConnectionError:
        return ToolResult(status="error", error="Search service unavailable", retry_hint="Retry in 5 seconds.")
    except Exception as e:
        return ToolResult(status="error", error=f"Unexpected error: {type(e).__name__}")
```

---

**Example 3: Idempotent Write Tool with Key**

```python
import hashlib

executed_operations: dict[str, dict] = {}

async def create_order(order_data: dict, idempotency_key: str) -> ToolResult:
    # Check if already executed
    if idempotency_key in executed_operations:
        return ToolResult(status="success", data=executed_operations[idempotency_key])

    # Validate
    if not order_data.get("items"):
        return ToolResult(status="error", error="Order must have at least one item")

    # Execute
    order = await db.create_order(order_data)

    # Cache result for idempotency
    executed_operations[idempotency_key] = order
    return ToolResult(status="success", data=order)
```

---

**Example 4: Sequential Tool Chain**

```python
async def research_pipeline(query: str, client):
    # Step 1: Search for relevant documents
    search_result = await search_documents(query)
    if search_result.status != "success":
        return {"error": "Search failed"}

    # Step 2: Analyze found documents
    analysis = await analyze_documents(
        documents=search_result.data["results"],
        question=query
    )

    # Step 3: Generate summary
    summary = await generate_summary(analysis.data)

    return {"sources": search_result.data, "analysis": analysis.data, "summary": summary}
```

---

**Example 5: Parallel Tool Execution**

```python
import asyncio

async def gather_customer_context(customer_id: str) -> dict:
    """Call multiple independent tools in parallel."""
    results = await asyncio.gather(
        get_customer_profile(customer_id),
        get_recent_orders(customer_id),
        get_support_tickets(customer_id),
        return_exceptions=True,
    )

    profile, orders, tickets = results
    return {
        "profile": profile if not isinstance(profile, Exception) else None,
        "orders": orders if not isinstance(orders, Exception) else [],
        "tickets": tickets if not isinstance(tickets, Exception) else [],
    }
```

---

**Example 6: Error Response Format**

```python
def make_error_response(error_type: str, message: str, retry_hint: str = None) -> dict:
    return {
        "status": "error",
        "error": {
            "type": error_type,
            "message": message,
            "retry_hint": retry_hint,
        }
    }

# Usage
return make_error_response("validation_error", "Email format invalid", "Provide email as user@domain.com")
return make_error_response("not_found", "User 12345 not found", "Verify the user ID and retry")
return make_error_response("rate_limit_error", "Too many requests", "Wait 60 seconds before retrying")
```

---

**Example 7: LangChain @tool Decorator Pattern**

```python
from langchain_core.tools import tool

@tool
def search_knowledge_base(query: str, category: str = "all") -> str:
    """Search the internal knowledge base for information.
    Use when users ask about company policies, procedures, or technical documentation.

    Args:
        query: Natural language search query
        category: Optional category filter (policy, technical, hr, finance, all)
    """
    results = vector_store.search(query, category=category)
    return format_results(results)

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely.
    Use for any numerical computation or calculation.

    Args:
        expression: Mathematical expression (e.g., "2 * (15 + 30)")
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {"abs": abs, "round": round, "min": min, "max": max})
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"
```

---

**Example 8: Tool Result Caching Decorator**

```python
import functools
import hashlib
import json

_tool_cache: dict[str, tuple] = {}

def cache_tool_result(ttl_seconds: int = 300):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = hashlib.sha256(
                json.dumps({"func": func.__name__, "args": str(args), "kwargs": str(kwargs)}).encode()
            ).hexdigest()

            if cache_key in _tool_cache:
                result, timestamp = _tool_cache[cache_key]
                if time.time() - timestamp < ttl_seconds:
                    return result

            result = await func(*args, **kwargs)
            _tool_cache[cache_key] = (result, time.time())
            return result
        return wrapper
    return decorator
```
