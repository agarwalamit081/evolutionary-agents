---
name: agent-tooling
description: Designing LLM-callable tools and functions — JSON schema definitions, error handling, idempotent design, tool orchestration, and best practices for tool-calling agents.
---

**When to Use**
- Defining tools/functions for LLM agents to call.
- Designing tool JSON schemas with proper validation.
- Implementing tool error handling and retry logic.
- Orchestrating multi-tool agent workflows.

**Core Principles**
1. **Strict JSON Schema**: Every tool must have complete schema (type, description, required, enums, examples).
2. **Descriptive Names**: Use `verb_noun` naming (e.g., `search_documents`). Descriptions explain when to use.
3. **Idempotent by Design**: Tools safe to call multiple times. Use idempotency keys for mutations.
4. **Graceful Errors**: Return structured error objects with error code, message, and retry guidance. Never throw unhandled.
5. **Input Validation**: Validate all inputs before processing. Reject early with clear messages.
6. **Output Contract**: Consistent output structure with `status` field (success/error/partial).

**References**
- Load `reference.md` for JSON Schema best practices, error handling, idempotency patterns, and orchestration.
- Load `examples.md` for tool definitions, orchestration patterns, and error responses.

**Scripts**
- `scripts/validate_tool_schema.py`: Validate a JSON tool schema for completeness.
