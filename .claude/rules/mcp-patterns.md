# MCP Server Patterns (fastmcp)

## Server Setup
- Use `fastmcp` for building MCP tool servers. Define the server with a clear name and version.
- Register tools using the `@mcp.tool()` decorator with proper type annotations on all parameters.
- Keep server initialization lightweight — defer heavy resource setup to tool execution.

## Tool Schema Design
- Write rich, detailed descriptions for every tool and every parameter — the LLM reads these to decide when and how to use the tool.
- Mark required vs optional parameters explicitly. Provide sensible defaults for optional parameters.
- Use string enums for parameters with fixed values (e.g., `"search_query"`, `"document_id"`).
- NEVER expose internal implementation details in tool descriptions.

## Error Responses
- Return structured error responses with a consistent format: `{"error": "category", "message": "user-facing description", "details": {}}`.
- NEVER expose internal stack traces, file paths, or database connection strings in error responses.
- Log full error details server-side using `loguru` for debugging, but return sanitized messages to the caller.

## Security
- Validate all input parameters on every tool — NEVER trust caller input.
- Implement rate limiting per tool to prevent abuse.
- Use sandboxed file paths: resolve all paths relative to a configured base directory. NEVER allow path traversal.
- Restrict network access: tools should only connect to explicitly allowlisted domains.

## Async Handlers
- ALWAYS use async tool handlers (`async def`). NEVER use synchronous handlers that block the event loop.
- Use `asyncio.Semaphore` for concurrency control on resource-intensive tools (embedding generation, web scraping).
- Properly clean up resources (connections, files) in finally blocks.

## Testing
- Write unit tests that call tools directly with mock inputs. Test both success and error cases.
- Validate tool schemas programmatically: ensure all registered tools have valid JSON schemas.
- Test with `fastmcp` client to verify end-to-end tool invocation.
