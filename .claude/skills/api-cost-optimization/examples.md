---
description: API Cost Optimization Examples
---

**Example 1: Token Counter Utility**

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        return len(text) // 4  # Fallback estimate

def count_message_tokens(messages: list[dict], model: str = "gpt-4o") -> int:
    total = 0
    for msg in messages:
        total += 4  # message overhead
        total += count_tokens(msg.get("content", ""), model)
    return total + 2  # priming tokens
```

---

**Example 2: Model Router Based on Task Complexity**

```python
from enum import Enum
from dataclasses import dataclass

class Complexity(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"

@dataclass
class ModelRoute:
    model: str
    max_tokens: int
    estimated_cost: float

COMPLEXITY_KEYWORDS = {
    Complexity.SIMPLE: ["summarize", "classify", "extract", "translate", "format"],
    Complexity.MEDIUM: ["analyze", "compare", "explain", "draft", "review"],
    Complexity.COMPLEX: ["architect", "design", "reason", "debate", "synthesize", "legal"],
}

MODEL_MAP = {
    Complexity.SIMPLE: ModelRoute("claude-haiku-4-5-20251001", 1024, 0.002),
    Complexity.MEDIUM: ModelRoute("claude-sonnet-4-6", 4096, 0.015),
    Complexity.COMPLEX: ModelRoute("claude-opus-4-8", 8192, 0.10),
}

def route_model(query: str) -> ModelRoute:
    query_lower = query.lower()
    for complexity, keywords in COMPLEXITY_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            return MODEL_MAP[complexity]
    return MODEL_MAP[Complexity.MEDIUM]  # Default to medium
```

---

**Example 3: Semantic Cache with Embeddings**

```python
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class CacheEntry:
    query_hash: str
    query_embedding: list[float]
    response: str
    timestamp: datetime
    ttl_seconds: int = 3600

class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.95):
        self.cache: list[CacheEntry] = []
        self.similarity_threshold = similarity_threshold

    def get(self, query_embedding: list[float]) -> str | None:
        now = datetime.now()
        for entry in self.cache:
            if (now - entry.timestamp).total_seconds() > entry.ttl_seconds:
                continue
            similarity = cosine_similarity(query_embedding, entry.query_embedding)
            if similarity >= self.similarity_threshold:
                return entry.response
        return None

    def set(self, query_embedding: list[float], response: str, ttl: int = 3600):
        self.cache.append(CacheEntry(
            query_hash=hashlib.sha256(str(query_embedding).encode()).hexdigest(),
            query_embedding=query_embedding,
            response=response,
            timestamp=datetime.now(),
            ttl_seconds=ttl,
        ))

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
```

---

**Example 4: Prompt Compression Before API Call**

```python
def compress_prompt(system_prompt: str, user_message: str, max_context_chars: int = 8000) -> tuple[str, str]:
    """Compress prompts by removing verbosity while preserving meaning."""
    # Compress system prompt: remove filler words
    compressed_system = system_prompt
    for filler in ["please ", "Please ", "make sure to ", "it is important to ", "always ", "Please ensure that "]:
        compressed_system = compressed_system.replace(filler, "")

    # Truncate user message if needed
    if len(user_message) > max_context_chars:
        user_message = user_message[:max_context_chars] + "\n[...truncated for length]"

    return compressed_system.strip(), user_message
```

---

**Example 5: Budget Tracker with Alerting**

```python
from datetime import datetime, timedelta
from dataclasses import dataclass, field

@dataclass
class BudgetEntry:
    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    feature: str

class BudgetTracker:
    def __init__(self, daily_limit: float = 50.0, alert_threshold: float = 0.8):
        self.daily_limit = daily_limit
        self.alert_threshold = alert_threshold
        self.entries: list[BudgetEntry] = []

    def record(self, model: str, input_tokens: int, output_tokens: int, cost: float, feature: str = "default"):
        self.entries.append(BudgetEntry(
            timestamp=datetime.now(), model=model,
            input_tokens=input_tokens, output_tokens=output_tokens,
            cost_usd=cost, feature=feature,
        ))
        if self.should_alert():
            print(f"WARNING: Daily spend at ${self.today_spend():.2f} / ${self.daily_limit:.2f}")

    def today_spend(self) -> float:
        today = datetime.now().replace(hour=0, minute=0, second=0)
        return sum(e.cost_usd for e in self.entries if e.timestamp >= today)

    def should_alert(self) -> bool:
        return self.today_spend() >= self.daily_limit * self.alert_threshold

    def is_over_budget(self) -> bool:
        return self.today_spend() >= self.daily_limit

    def summary(self) -> dict:
        today = datetime.now().replace(hour=0, minute=0, second=0)
        today_entries = [e for e in self.entries if e.timestamp >= today]
        return {
            "total_spend": self.today_spend(),
            "daily_limit": self.daily_limit,
            "usage_pct": self.today_spend() / self.daily_limit * 100,
            "requests_today": len(today_entries),
            "by_feature": self._by_dimension(today_entries, "feature"),
            "by_model": self._by_dimension(today_entries, "model"),
        }

    def _by_dimension(self, entries: list, key: str) -> dict:
        result = {}
        for e in entries:
            k = getattr(e, key, "unknown")
            result[k] = result.get(k, 0) + e.cost_usd
        return result
```

---

**Example 6: Cost-per-Feature Logging Middleware**

```python
import json
from datetime import datetime

class CostLoggingMiddleware:
    def __init__(self, log_file: str = "llm_costs.jsonl"):
        self.log_file = log_file

    def log_call(self, model: str, input_tokens: int, output_tokens: int,
                 cost_usd: float, feature: str, latency_ms: float):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost_usd, 6),
            "feature": feature,
            "latency_ms": round(latency_ms, 1),
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
```

---

**Example 7: Escalation Pattern (Cheap → Expensive Model)**

```python
async def query_with_escalation(prompt: str, confidence_threshold: float = 0.8):
    # Try cheap model first
    haiku_response = await call_model("claude-haiku-4-5-20251001", prompt)
    confidence = haiku_response.get("confidence", 0)

    if confidence >= confidence_threshold:
        return {"answer": haiku_response["answer"], "model": "haiku", "cost": 0.002}

    # Escalate to better model
    sonnet_response = await call_model("claude-sonnet-4-6", prompt)
    return {"answer": sonnet_response["answer"], "model": "sonnet", "cost": 0.015}
```

---

**Example 8: Daily Cost Report Generator**

```python
import json
from datetime import datetime

def generate_daily_report(log_file: str = "llm_costs.jsonl") -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    entries = []
    with open(log_file) as f:
        for line in f:
            entry = json.loads(line)
            if entry["timestamp"].startswith(today):
                entries.append(entry)

    if not entries:
        return f"No LLM calls recorded for {today}"

    total_cost = sum(e["cost_usd"] for e in entries)
    total_input = sum(e["input_tokens"] for e in entries)
    total_output = sum(e["output_tokens"] for e in entries)

    by_model = {}
    for e in entries:
        by_model.setdefault(e["model"], {"cost": 0, "calls": 0})
        by_model[e["model"]]["cost"] += e["cost_usd"]
        by_model[e["model"]]["calls"] += 1

    lines = [
        f"# LLM Cost Report — {today}",
        f"",
        f"- **Total calls**: {len(entries)}",
        f"- **Total cost**: ${total_cost:.4f}",
        f"- **Total tokens**: {total_input:,} in / {total_output:,} out",
        f"",
        f"## By Model",
        f"| Model | Calls | Cost |",
        f"|---|---|---|",
    ]
    for model, data in sorted(by_model.items(), key=lambda x: -x[1]["cost"]):
        lines.append(f"| {model} | {data['calls']} | ${data['cost']:.4f} |")

    return "\n".join(lines)
```
