# Security Guardrails

## Credential Management
- NEVER hardcode API keys, passwords, database URLs, tokens, webhooks, or any secrets in source code.
- ALWAYS fetch credentials from environment variables (`process.env`, `os.environ`) or `.env` configuration files managed by `pydantic-settings`.
- NEVER echo, print, or log sensitive information to console or log files. Use `loguru` with PII redaction via `presidio`.
- NEVER embed secrets in client-side code (React, Vue, etc.). All secrets belong in server-side `.env`.
- NEVER use `NEXT_PUBLIC_` or `VITE_` prefixes for sensitive keys — these are exposed to the browser bundle.
- NEVER commit `.env` files to git. Ensure `.env` is in `.gitignore`. If leaked: rotate every exposed key immediately and use `git filter-branch` or BFG Repo-Cleaner to purge history.

## Database Security
- ALWAYS enable Row Level Security (RLS) on every database table. NEVER assume the AI configured your database securely.
- ALWAYS use parameterized queries or ORM models (SQLAlchemy, Prisma, etc.). Raw SQL concatenation with user input is strictly forbidden.
- NEVER use string interpolation or f-strings for SQL queries: `f"SELECT * FROM users WHERE id = {user_id}"` is a critical security violation.
- ALWAYS validate and sanitize all user inputs before they reach the database layer.
- When altering ORM schemas, ALWAYS generate and commit the corresponding physical migration file (Alembic, Prisma migrate).

## CORS and Origin Policy
- NEVER use wildcard (`*`) CORS origins in production. Always specify explicit allowed origins (your frontend domains only).
- CORS with `Access-Control-Allow-Origin: *` is especially dangerous when the API uses cookies or auth headers — it bypasses the browser's same-origin security.
- CORS configuration should be environment-aware: `*` is only acceptable for local development with explicit documentation.

## XSS and Injection Prevention
- NEVER use `dangerouslySetInnerHTML` (React), `innerHTML` (vanilla JS), or template literals for rendering user-supplied content without explicit sanitization (DOMPurify, etc.).
- ALWAYS sanitize user inputs before rendering in HTML templates.
- NEVER construct HTML strings from user data. Use framework-provided escaping mechanisms.
- NEVER pass raw user inputs to shell commands. Use proper argument escaping or avoid shell execution entirely.
- Test XSS by entering `<script>alert('xss')</script>` into every text field and verifying it does not execute.

## Authentication and Authorization
- ALWAYS implement proper authentication checks on API endpoints.
- NEVER expose admin or privileged endpoints without proper authorization middleware.
- ALWAYS validate JWT tokens, session cookies, or API keys on protected routes.
- Test admin API routes directly as a regular user (e.g., `POST /api/admin/users/delete`). If it works, add role-based middleware.
- Review OAuth scope: does your app request more permissions than it needs? A read-only integration shouldn't have write access.

## Password and Session Security
- ALWAYS use bcrypt, scrypt, or argon2 for password hashing. NEVER use MD5 or SHA-1.
- JWTs MUST have reasonable expiration times (not 30 days). Validate on every request.
- Cookies MUST have `HttpOnly`, `Secure`, and `SameSite=Strict` (or `Lax`) flags.

## HTTPS and Transport Security
- ALL API calls MUST use HTTPS. `http://` MUST redirect to `https://` automatically.
- Block mixed content (HTTPS page loading HTTP resources).
- Set security headers: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Strict-Transport-Security` (HSTS), `Content-Security-Policy`.

## Input Validation and File Uploads
- Validate data types, string lengths, number ranges, and required fields. Use Pydantic for all API request validation.
- Validate file types server-side by checking actual file header/magic bytes — NOT just the extension.
- Set maximum file sizes for all upload endpoints.

## Rate Limiting
- ALWAYS add rate limiting, even if every request currently succeeds without throttling.

## Error Handling
- Trigger errors intentionally and ensure they return generic messages like "Something went wrong".
- NEVER return stack traces, SQL errors, or file paths in error responses.

## Logging and Privacy
- Server logs MUST NOT contain passwords, API keys, credit card numbers, or personal data. Redact sensitive fields.
- Use `presidio` for automated PII redaction in logs and telemetry.

## Webhook Security
- ALWAYS verify webhook signatures using the provider's verification library (e.g., `stripe.webhooks.constructEvent()`).
- Reject any request with an invalid or missing signature.

## Account Deletion
- If a user requests account deletion, delete their data from ALL tables, file storage, third-party services, linked records, uploaded files, and analytics data to comply with regulations.

## LLM-Specific Security
- Sanitize all user input before interpolating into LLM prompts. Implement prompt injection defense at every input surface.
- Review tool descriptions for injection leakage — adversarial prompts can manipulate tool-selection behavior.
- NEVER expose internal system prompts, reasoning chains, or tool schemas in LLM responses to end users.

## MCP Server Security
- Validate all tool input parameters. Restrict file system access to sandboxed directories. Restrict network access to allowlisted domains.
- NEVER expose internal stack traces or connection strings in tool error responses.

## pgvector Security
- Be aware of vector similarity attacks and embedding inversion risks when exposing vector search to untrusted users.

## Dependency and Supply Chain Security
- ALWAYS verify package versions before using new features. Check for known CVEs.
- NEVER use deprecated libraries, functions, methods, or APIs. Verify official documentation first.
- ALWAYS pin dependency versions in `requirements.txt`, `package.json`, or `pyproject.toml`.
