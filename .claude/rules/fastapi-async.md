# FastAPI & Async Patterns

## Route Handlers
- ALWAYS use `async def` for route handlers. NEVER use synchronous handlers in an async application — they block the event loop.
- Use `Depends()` for dependency injection: authentication, database sessions, configuration objects.
- ALWAYS specify `response_model` on routes for automatic serialization, validation, and OpenAPI documentation.

## Request Validation
- ALL request bodies MUST use Pydantic models. NEVER accept raw `dict` or untyped JSON.
- Use `Body()`, `Query()`, `Path()` for parameter validation with explicit types and constraints.
- Validate file uploads with `UploadFile`: check file type by magic bytes (not just extension), enforce maximum file size.

## Middleware
- Configure CORS with explicitly listed frontend domains. NEVER use `allow_origins=["*"]` in production.
- Implement error handling middleware that returns generic error messages — never expose stack traces, SQL errors, or file paths.
- Add request ID tracking middleware for distributed tracing correlation.

## Streaming Responses
- Use `StreamingResponse` for SSE and LLM token streams. Always set proper `Content-Type: text/event-stream`.
- Implement graceful cancellation: handle `asyncio.CancelledError` in streaming handlers and clean up resources.

## Background Tasks
- Use `BackgroundTasks` for non-blocking operations (email sending, analytics, notifications).
- NEVER perform blocking I/O in background tasks — use async equivalents (`aiofiles`, `httpx.AsyncClient`).

## Error Handling
- Raise `HTTPException` with proper status codes (400 for validation, 401 for auth, 403 for forbidden, 404 for not found, 422 for unprocessable, 500 for internal).
- Create custom exception classes for domain-specific errors and register global exception handlers.

## Health Endpoints
- ALWAYS implement `/health` (liveness) and `/ready` (readiness) endpoints for container orchestration.
- `/ready` MUST verify database connectivity and external service availability.
