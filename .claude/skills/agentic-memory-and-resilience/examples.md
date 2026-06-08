---
description: Agentic Memory and Resilience Examples
---

**Example 1: Short-Term Memory Manager with Token Counting**

```python
import tiktoken

class ConversationMemory:
    def __init__(self, max_tokens: int = 8000, model: str = "gpt-4o"):
        self.messages: list[dict] = []
        self.max_tokens = max_tokens
        self.encoding = tiktoken.encoding_for_model(model)

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def total_tokens(self) -> int:
        return sum(self.count_tokens(m["content"]) for m in self.messages)

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        if self.total_tokens() > self.max_tokens:
            self._compress()

    def _compress(self):
        """Summarize oldest messages to free space."""
        if len(self.messages) <= 4:
            return
        # Keep first 2 (system + first user) and last 2
        to_summarize = self.messages[2:-2]
        summary = summarize_messages(to_summarize)
        self.messages = [
            self.messages[0], self.messages[1],
            {"role": "system", "content": f"[Previous context summary]\n{summary}"},
            *self.messages[-2:]
        ]

    def get_messages(self) -> list[dict]:
        return self.messages
```

---

**Example 2: Conversation Summarizer**

```python
def summarize_messages(messages: list[dict]) -> str:
    """Compress a list of messages into a concise summary."""
    # Heuristic approach (no LLM needed)
    facts = []
    decisions = []
    for msg in messages:
        content = msg["content"].lower()
        if any(kw in content for kw in ["decided", "chose", "selected", "confirmed"]):
            decisions.append(msg["content"][:200])
        elif any(kw in content for kw in ["is", "are", "has", "works at", "located"]):
            facts.append(msg["content"][:150])

    parts = []
    if facts:
        parts.append("Key facts: " + "; ".join(facts[:5]))
    if decisions:
        parts.append("Decisions made: " + "; ".join(decisions[:3]))
    return " | ".join(parts) if parts else "Continued conversation."
```

---

**Example 3: Long-Term Memory Storage and Retrieval**

```python
from dataclasses import dataclass, field
from datetime import datetime
import uuid

@dataclass
class MemoryEntry:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    content: str = ""
    category: str = "general"  # preference, fact, decision, instruction
    confidence: float = 1.0
    source_conversation_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)

class LongTermMemory:
    def __init__(self, db_connection):
        self.db = db_connection

    def store(self, entry: MemoryEntry):
        self.db.execute(
            "INSERT INTO memories (id, user_id, content, category, confidence, source_id, created_at) "
            "VALUES (:id, :user_id, :content, :category, :confidence, :source_id, :created_at)",
            entry.__dict__
        )

    def retrieve_relevant(self, user_id: str, query: str, limit: int = 5) -> list[MemoryEntry]:
        rows = self.db.execute(
            "SELECT * FROM memories WHERE user_id = :uid ORDER BY last_accessed DESC LIMIT :lim",
            {"uid": user_id, "lim": limit}
        ).fetchall()
        return [MemoryEntry(**dict(r)) for r in rows]

    def update_access(self, memory_id: str):
        self.db.execute(
            "UPDATE memories SET last_accessed = NOW() WHERE id = :mid", {"mid": memory_id}
        )
```

---

**Example 4: User Profile Update from Conversation**

```python
def extract_facts_from_conversation(messages: list[dict], user_id: str) -> list[MemoryEntry]:
    """Extract persistent facts from a conversation to store in long-term memory."""
    facts = []
    for msg in messages:
        content = msg.get("content", "")
        # Simple heuristic extraction — in production use an LLM for extraction
        if "my name is" in content.lower():
            name = content.split("my name is")[-1].strip().split()[0].title()
            facts.append(MemoryEntry(
                user_id=user_id, content=f"User's name is {name}",
                category="fact", confidence=0.95
            ))
        elif "i work at" in content.lower():
            company = content.split("i work at")[-1].strip().split(".")[0]
            facts.append(MemoryEntry(
                user_id=user_id, content=f"User works at {company}",
                category="fact", confidence=0.9
            ))
    return facts
```

---

**Example 5: Loop Detector Implementation**

```python
from collections import Counter
from dataclasses import dataclass

@dataclass
class ToolCall:
    tool_name: str
    args_hash: str  # deterministic hash of arguments

class LoopDetector:
    def __init__(self, max_repeats: int = 3, max_total_calls: int = 15):
        self.history: list[ToolCall] = []
        self.max_repeats = max_repeats
        self.max_total_calls = max_total_calls

    def record(self, tool_name: str, args_hash: str):
        self.history.append(ToolCall(tool_name, args_hash))

    def is_stuck(self) -> bool:
        # Check for repeated identical calls
        if len(self.history) >= self.max_repeats:
            recent = [(t.tool_name, t.args_hash) for t in self.history[-self.max_repeats:]]
            if len(set(recent)) == 1:
                return True

        # Check for cycling between states
        if len(self.history) >= 6:
            recent_3 = [(t.tool_name, t.args_hash) for t in self.history[-3:]]
            prev_3 = [(t.tool_name, t.args_hash) for t in self.history[-6:-3]]
            if recent_3 == prev_3:
                return True

        # Check total calls
        if len(self.history) >= self.max_total_calls:
            return True

        return False

    def get_intervention_message(self) -> str:
        if len(self.history) >= self.max_total_calls:
            return "Maximum tool calls reached. Summarize progress and ask user for guidance."
        return "Detected repeated actions. Re-evaluate the approach and try a different strategy."
```

---

**Example 6: Tool Retry with Exponential Backoff**

```python
import asyncio
import random
from functools import wraps

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                        await asyncio.sleep(delay)
            raise last_error
        return wrapper
    return decorator

# Usage
@retry_with_backoff(max_retries=3, base_delay=1.0)
async def call_external_api(endpoint: str, payload: dict) -> dict:
    # Your API call here
    pass
```

---

**Example 7: Graceful Degradation Fallback Chain**

```python
from typing import Any, Callable

class FallbackChain:
    def __init__(self, strategies: list[tuple[str, Callable]]):
        self.strategies = strategies  # [(name, callable), ...]

    async def execute(self, *args, **kwargs) -> tuple[Any, str]:
        errors = []
        for name, strategy in self.strategies:
            try:
                result = await strategy(*args, **kwargs)
                return result, name
            except Exception as e:
                errors.append(f"{name}: {e}")
                continue

        # All strategies failed
        return {
            "error": "All strategies failed",
            "attempts": errors,
            "partial_result": None,
        }, "failed"

# Usage
chain = FallbackChain([
    ("primary_model", call_gpt4),
    ("fallback_model", call_haiku),
    ("cached_response", get_cached_answer),
    ("static_fallback", lambda *a, **k: "I'm unable to answer right now. Please try again."),
])
result, strategy_used = await chain.execute("What is our policy on...?")
```

---

**Example 8: Token Budget Tracker**

```python
class TokenBudget:
    def __init__(self, total: int):
        self.total = total
        self.allocated: dict[str, int] = {}

    def allocate(self, category: str, percentage: float):
        self.allocated[category] = int(self.total * percentage)

    def remaining_for(self, category: str) -> int:
        return self.allocated.get(category, 0)

    def is_within_budget(self, category: str, used: int) -> bool:
        return used < self.allocated.get(category, 0) * 0.9  # 90% warning threshold

# Setup
budget = TokenBudget(total=128000)
budget.allocate("system_prompt", 0.10)   # 12,800 tokens
budget.allocate("retrieved_context", 0.25)  # 32,000 tokens
budget.allocate("conversation", 0.40)   # 51,200 tokens
budget.allocate("output_buffer", 0.25)  # 32,000 tokens
```
