---
description: Multi-Agent Orchestration Examples
---

**Example 1: Supervisor Agent Prompt with Task Decomposition**

```python
SUPERVISOR_PROMPT = """You are a task orchestrator. Break down complex requests into sub-tasks
and assign them to the appropriate specialist agent.

<available_agents>
- researcher: Finds and gathers information from internal and external sources
- analyst: Analyzes data, identifies patterns, draws conclusions
- writer: Creates well-structured written content
- reviewer: Reviews output for quality, accuracy, and completeness
</available_agents>

<handoff_format>
When assigning a task, output:
{
  "action": "delegate",
  "agent": "<agent_name>",
  "task": "<specific instruction>",
  "context": "<relevant background>",
  "constraints": ["<limitation 1>", "<limitation 2>"]
}
</handoff_format>

When all sub-tasks are complete:
{
  "action": "synthesize",
  "summary": "<final combined result>"
}
</handoff_format>
"""
```

---

**Example 2: Worker Agent with Handoff Protocol**

```python
from dataclasses import dataclass

@dataclass
class HandoffContext:
    from_agent: str
    to_agent: str
    task: str
    findings: list[str]
    constraints: list[str]
    parent_task_id: str

class ResearcherAgent:
    async def execute(self, context: HandoffContext) -> HandoffContext:
        findings = await self.research(context.task)

        return HandoffContext(
            from_agent="researcher",
            to_agent="analyst",
            task=f"Analyze these findings: {context.task}",
            findings=findings,
            constraints=context.constraints,
            parent_task_id=context.parent_task_id,
        )
```

---

**Example 3: Agent Context Transfer Format (JSON)**

```json
{
  "handoff": {
    "from": "researcher",
    "to": "analyst",
    "timestamp": "2025-01-15T10:30:00Z",
    "task_id": "task-001",
    "subtask_id": "subtask-003"
  },
  "findings": [
    {"source": "internal_docs", "content": "Revenue grew 15% YoY", "confidence": 0.95},
    {"source": "market_report", "content": "Industry average growth 8%", "confidence": 0.85}
  ],
  "metadata": {
    "tools_used": ["search_documents", "get_financials"],
    "tokens_consumed": 2450,
    "elapsed_seconds": 12
  },
  "next_action": {
    "agent": "analyst",
    "instruction": "Compare our growth vs industry. Identify drivers.",
    "constraints": ["Use only provided findings", "Quantify where possible"]
  }
}
```

---

**Example 4: Conflict Resolution with Confidence Voting**

```python
from dataclasses import dataclass

@dataclass
class AgentVote:
    agent_name: str
    answer: str
    confidence: float  # 0.0 - 1.0
    reasoning: str

def resolve_conflict(votes: list[AgentVote], strategy: str = "confidence_weighted") -> str:
    if strategy == "majority":
        from collections import Counter
        counts = Counter(v.answer for v in votes)
        return counts.most_common(1)[0][0]

    elif strategy == "confidence_weighted":
        scores: dict[str, float] = {}
        for vote in votes:
            scores[vote.answer] = scores.get(vote.answer, 0) + vote.confidence
        return max(scores, key=scores.get)

    elif strategy == "unanimous":
        answers = set(v.answer for v in votes)
        if len(answers) == 1:
            return votes[0].answer
        return "DISPUTED: Requires human review"

    raise ValueError(f"Unknown strategy: {strategy}")
```

---

**Example 5: LangGraph Supervisor-Worker Graph**

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    current_agent: str
    task_complete: bool

def supervisor(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    # Determine next agent based on content
    if "research" in last_message.lower():
        return {"current_agent": "researcher"}
    elif "analyze" in last_message.lower():
        return {"current_agent": "analyst"}
    elif "write" in last_message.lower():
        return {"current_agent": "writer"}
    return {"task_complete": True}

def researcher(state: AgentState) -> dict:
    # Research logic
    return {"messages": ["Research complete: findings XYZ"], "current_agent": "supervisor"}

def analyst(state: AgentState) -> dict:
    # Analysis logic
    return {"messages": ["Analysis complete: insights ABC"], "current_agent": "supervisor"}

def writer(state: AgentState) -> dict:
    # Writing logic
    return {"messages": ["Report written: ..."], "current_agent": "supervisor"}

# Build graph
graph = StateGraph(AgentState)
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher)
graph.add_node("analyst", analyst)
graph.add_node("writer", writer)

graph.add_conditional_edges("supervisor", lambda s: s["current_agent"], {
    "researcher": "researcher",
    "analyst": "analyst",
    "writer": "writer",
})
for agent in ["researcher", "analyst", "writer"]:
    graph.add_edge(agent, "supervisor")

app = graph.compile()
```

---

**Example 6: Shared State Management**

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import uuid

@dataclass
class SharedState:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version: int = 0
    global_context: dict[str, Any] = field(default_factory=dict)
    agent_outputs: dict[str, Any] = field(default_factory=dict)
    errors: list[dict] = field(default_factory=list)

    def update(self, agent_name: str, key: str, value: Any):
        self.agent_outputs.setdefault(agent_name, {})[key] = value
        self.version += 1

    def get(self, agent_name: str, key: str, default=None) -> Any:
        return self.agent_outputs.get(agent_name, {}).get(key, default)

    def add_error(self, agent_name: str, error: str):
        self.errors.append({
            "agent": agent_name,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        })
```

---

**Example 7: Agent Error Recovery and Re-assignment**

```python
class AgentOrchestrator:
    def __init__(self, agents: dict[str, Agent], max_retries: int = 2):
        self.agents = agents
        self.max_retries = max_retries

    async def execute_with_recovery(self, task: str, agent_name: str) -> dict:
        for attempt in range(self.max_retries):
            try:
                result = await self.agents[agent_name].execute(task)
                return {"status": "success", "result": result, "attempts": attempt + 1}
            except Exception as e:
                print(f"Agent {agent_name} failed (attempt {attempt + 1}): {e}")

                # Try fallback agent
                fallback = self._get_fallback(agent_name)
                if fallback:
                    print(f"Trying fallback: {fallback}")
                    try:
                        result = await self.agents[fallback].execute(task)
                        return {"status": "success", "result": result, "fallback": True}
                    except Exception:
                        pass

        return {"status": "failed", "error": f"All retries exhausted for {agent_name}"}

    def _get_fallback(self, agent_name: str) -> str | None:
        fallbacks = {"researcher": "analyst", "analyst": "researcher", "writer": "researcher"}
        return fallbacks.get(agent_name)
```

---

**Example 8: Blackboard Pattern for Collaborative Problem-Solving**

```python
class Blackboard:
    """Shared problem space that all agents read/write to."""

    def __init__(self):
        self.facts: list[dict] = []
        self.hypotheses: list[dict] = []
        self.decisions: list[dict] = []

    def add_fact(self, agent: str, fact: str, confidence: float):
        self.facts.append({"agent": agent, "fact": fact, "confidence": confidence})

    def add_hypothesis(self, agent: str, hypothesis: str, supporting_facts: list[int]):
        self.hypotheses.append({
            "agent": agent, "hypothesis": hypothesis,
            "supporting": supporting_facts, "validated": False
        })

    def get_unvalidated_hypotheses(self) -> list[dict]:
        return [h for h in self.hypotheses if not h["validated"]]

    def get_context_for_agent(self, agent_name: str) -> str:
        return (
            f"Facts: {[f['fact'] for f in self.facts]}\n"
            f"Open hypotheses: {[h['hypothesis'] for h in self.get_unvalidated_hypotheses()]}"
        )
```
