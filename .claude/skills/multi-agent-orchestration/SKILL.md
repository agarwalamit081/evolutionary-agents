---
name: multi-agent-orchestration
description: Multi-agent system patterns — supervisor-worker architectures, handoff protocols, agent communication, conflict resolution, and collaborative decision-making.
---

**When to Use**
- Building multi-agent systems where agents have different roles.
- Designing agent handoffs and context transfer.
- Implementing supervisor/worker or pipeline patterns.
- Resolving disagreements between agents.

**Core Principles**
1. **Clear Role Boundaries**: Each agent has a single, well-defined responsibility and tool set.
2. **Structured Handoffs**: Explicit handoff protocol with context transfer (state, reasoning, next action).
3. **Supervisor Pattern**: Complex tasks → supervisor decomposes, assigns to workers, synthesizes results.
4. **Conflict Resolution**: When agents disagree — voting, confidence scoring, or judge escalation.
5. **Shared State**: All agents read shared context; only the owning agent writes to its domain.

**References**
- Load `reference.md` for architecture patterns, handoff protocols, conflict resolution, and state management.
- Load `examples.md` for supervisor prompts, handoff formats, and LangGraph patterns.

**Scripts**
- `scripts/validate_agent_config.py`: Validate agent configuration files for consistency.
