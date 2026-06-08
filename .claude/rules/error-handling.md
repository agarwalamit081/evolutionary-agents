# Error Handling Patterns

## Retry with Tenacity
- Use `tenacity` for retry with exponential backoff + jitter on all external API calls.
- ONLY retry on transient errors: rate limits (429), server errors (5xx), timeouts, connection errors.
- NEVER retry on: authentication errors (401/403), validation errors (400/422), not found (404).
- Cap maximum retries (typically 3-5) and set a maximum wait time.
- Use `tenacity.before_sleep` hook to log retry attempts with context.

## Circuit Breaker
- Use `circuitbreaker` for provider outage protection with configurable failure thresholds.
- Set recovery timeout (typically 30-60 seconds) before attempting to close the circuit.
- Provide fallback responses when the circuit is open — cached data, degraded service, or graceful error messages.
- Monitor circuit state transitions as a metric for provider reliability.

## Exception Hierarchy
- Create custom exception classes that inherit from a base project exception. NEVER raise bare `Exception` or `RuntimeError`.
- Use specific exception types for different failure categories: `ExternalServiceError`, `ValidationError`, `ConfigurationError`, `ResourceExhaustedError`.
- NEVER use bare `except:` or `except Exception:` without logging. At minimum, log the exception via `loguru`.

## Structured Error Logging
- Use `loguru` for all error logging. Include structured context: correlation ID, operation name, input parameters (redacted), and error category.
- Log full stack traces at DEBUG level only. At WARNING/ERROR level, log a concise summary with the correlation ID for lookup.
- NEVER log sensitive data: passwords, API keys, tokens, PII. Use `presidio` for automated PII redaction in logs.

## FastAPI Exception Handlers
- Register global exception handlers that map domain exceptions to appropriate HTTP status codes.
- Return generic error messages to clients: `{"detail": "Something went wrong"}`. NEVER expose stack traces, SQL errors, or internal file paths.
- Log the full error details server-side with a correlation ID that matches the client response.

## Async Error Handling
- ALL `async def` functions MUST include `try/except` blocks when performing I/O operations.
- Handle `asyncio.CancelledError` explicitly in long-running operations — clean up resources before re-raising.
- Use `asyncio.gather(*tasks, return_exceptions=True)` to collect results without aborting on first failure.

## Database Error Handling
- Handle connection pool exhaustion gracefully: return 503 Service Unavailable with a retry-after hint.
- Implement deadlock retry: catch deadlock errors and retry with exponential backoff (up to 3 times).
- On migration failure: rollback immediately, log the failed migration ID, and alert the team. NEVER leave the database in a partially migrated state.
