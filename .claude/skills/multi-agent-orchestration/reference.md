---
description: Multi-Agent Orchestration Reference
---

## Architecture Patterns

| Pattern | Description | Best For |
|---|---|---|
| Supervisor-Worker | One supervisor decomposes tasks, assigns to workers | Complex, decomposable tasks |
| Pipeline | Agents pass output as input in sequence | Multi-stage processing |
| Round-Robin | Agents take turns contributing | Collaborative brainstorming |
| Blackboard | All agents read/write shared problem space | Ill-defined, evolving problems |

## Handoff Protocol

When Agent A hands off to Agent B:
```json
{
  "from_agent": "researcher",
  "to_agent": "writer",
  "context": {
    "task": "Summarize findings into a report",
    "findings": [...],
    "constraints": ["Max 500 words", "Executive audience"],
    "parent_task_id": "task-001"
  },
  "reason": "Research complete, need writing expertise"
}
```

## Agent Communication

| Method | Use Case |
|---|---|
| Direct messaging | Point-to-point handoff |
| Shared memory | Read-only context all agents access |
| Event bus | Broadcast events (task completed, error occurred) |

## Conflict Resolution

| Strategy | How It Works | When to Use |
|---|---|---|
| Majority voting | 3+ agents vote, majority wins | Factual disagreements |
| Confidence-weighted | Higher confidence = more weight | Asymmetric expertise |
| Judge escalation | Neutral third agent decides | Unresolvable conflicts |
| Human escalation | Ask the user | High-stakes or ambiguous |

## State Management

- **Shared context**: Read-only for all agents, written by orchestrator only.
- **Agent state**: Private to each agent (memory, scratchpad).
- **Task state**: Tracks progress (pending → in_progress → completed/failed).
- **Versioning**: Context versioned to prevent lost updates.

## Error Handling

- **Agent failure**: Supervisor reassigns to backup agent with partial context.
- **Timeout**: Agent doesn't respond → mark as failed, reassign.
- **Hallucinated handoff**: Agent tries to call non-existent agent → validate target exists before routing.
- **Cascading failure**: Multiple agents fail → fall back to single-agent mode.

## Scaling

- Agent pooling: Pre-instantiate agents, assign from pool.
- Load balancing: Route to least-busy agent.
- Rate limiting: Per-agent rate limits to prevent runaway costs.
