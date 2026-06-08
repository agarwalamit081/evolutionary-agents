---
description: Agent Tooling Reference
---

## JSON Schema Best Practices for LLM Tools

- Every parameter needs `description` and `type`.
- Use `enum` for constrained values (e.g., `["active", "suspended", "deleted"]`).
- Include `examples` in the schema for clarity.
- Mark required fields explicitly.
- Use `default` values where sensible.

## Tool Registration Patterns

| Framework | Pattern |
|---|---|
| Anthropic Claude | `tools` parameter with `name`, `description`, `input_schema` |
| OpenAI | `functions` or `tools` parameter with `type: "function"` |
| LangChain | `@tool` decorator or `StructuredTool` class |

## Error Handling Taxonomy

| Error Type | HTTP Analog | Response |
|---|---|---|
| `validation_error` | 400 | Clear field-level error messages |
| `not_found` | 404 | "Resource X not found. Check ID." |
| `permission_denied` | 403 | "Insufficient permissions for action Y." |
| `timeout_error` | 504 | "Operation timed out. Retry with smaller scope." |
| `rate_limit_error` | 429 | "Rate limit hit. Wait N seconds before retry." |
| `execution_error` | 500 | Generic error with correlation ID for debugging |

## Idempotency Patterns

- **Idempotency keys**: Client sends unique key, server checks for previous result.
- **Check-before-write**: Verify current state matches expected before mutating.
- **Upsert pattern**: `INSERT ... ON CONFLICT DO UPDATE` for safe re-execution.
- **Result caching**: Store tool results keyed by (tool_name, args_hash) for duplicate detection.

## Tool Orchestration Patterns

- **Sequential**: Tool B depends on Tool A's output. Chain calls.
- **Parallel**: Independent tools called concurrently with `asyncio.gather`.
- **Conditional**: Output of Tool A determines whether to call Tool B or Tool C.
- **Looping**: Agent iterates until a condition is met (with max iteration guard).

## Security

- **Input sanitization**: Never pass raw LLM output to shell commands or SQL.
- **Output filtering**: Remove sensitive data before returning to LLM.
- **Permission scoping**: Each tool should have minimum required permissions.
- **Audit logging**: Log all tool invocations with args, result, and duration.
