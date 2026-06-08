# Security Audit — Reference

Comprehensive security rules organized by category. Use this as a checklist when auditing code.

---

## Credential Management

- **No hardcoded secrets**: API keys, tokens, passwords, database URLs, and webhook secrets must never appear in source code.
- **Environment variables**: All secrets loaded from env vars or `.env` files. Ensure `.env` is in `.gitignore`.
- **No secrets in client-side code**: Frontend bundles are public. Any key shipped to the browser is compromised.
- **Git history**: Use `git log -p | grep -iE "api_key|password|secret|token"` to check for leaked secrets in history.
- **Remediation if leaked**: Rotate the credential immediately. Use `git filter-repo` or BFG Repo-Cleaner to purge history.

## SQL Injection Prevention

- **No raw SQL string concatenation** with user input. This includes `+`, `%s` via string formatting, and f-strings.
- **Parameterized queries only**: Use ORM models (SQLAlchemy, Prisma, Drizzle) or parameterized statements (`cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))`).
- **Never use f-strings or `.format()`** in SQL queries — even if you think the input is safe.
- **Watch for ORM escape hatches**: `session.execute(text(...))` with string interpolation is just as dangerous.

## XSS (Cross-Site Scripting) Prevention

- **React**: No `dangerouslySetInnerHTML` unless the content is first sanitized with DOMPurify.
- **User input must be sanitized** before rendering as HTML. Use context-appropriate encoding.
- **Content-Security-Policy headers**: Configure CSP to restrict script sources. Report-only mode for gradual rollout.
- **Manual testing**: Enter `<script>alert('xss')</script>` into every text field, URL parameter, and header.

## Authentication & Authorization

- **Protected endpoints**: Every endpoint that modifies data or returns user-specific data must have auth middleware.
- **Role-based access**: Admin/privileged endpoints must check user role, not just authentication status.
- **Testing**: Call admin API routes as a regular user and verify `403 Forbidden`.
- **JWT tokens**: Validate signature, expiration (`exp`), issuer (`iss`), and audience (`aud`) on every request. Set reasonable expiration (15 min access + refresh token pattern).
- **Cookies**: `HttpOnly` (prevents JS access), `Secure` (HTTPS only), `SameSite=Strict` or `Lax` (CSRF protection).

## CORS Configuration

- **No wildcard origins in production**: `Access-Control-Allow-Origin: *` is forbidden in production.
- **Explicitly list frontend domains**: e.g., `["https://app.example.com", "https://admin.example.com"]`.
- **Especially dangerous with credentials**: `allow_origins=["*"]` + `allow_credentials=True` is a critical security hole.

## Rate Limiting

- **Always add rate limiting** to API endpoints, especially auth endpoints (login, password reset, signup).
- **Use a per-user or per-IP throttle**: Even if not currently enforced, have the middleware in place.
- **Return 429 Too Many Requests** with a `Retry-After` header.

## Error Messages

- **Return generic errors to clients**: "Something went wrong. Please try again."
- **Never expose**: Stack traces, SQL error messages, file paths, internal service names, or configuration details.
- **Log the full error server-side** for debugging, but redact PII from logs.

## Password Hashing

- **Use bcrypt, scrypt, or argon2**: These are purpose-built for password hashing with salting and adjustable work factors.
- **Never use MD5, SHA-1, or SHA-256** for passwords — they are fast hashes designed for checksums, not security.
- **Never store plaintext passwords** — even temporarily.

## HTTPS Enforcement

- **Redirect all HTTP to HTTPS**: Configure at the load balancer or middleware level.
- **All API calls must use HTTPS**: No `http://` API URLs in production code.
- **Block mixed content**: Browsers warn about mixed content; enforce HTTPS for all resources.

## Security Headers

Configure these headers on all responses:

| Header | Value |
|---|---|
| `X-Content-Type-Options` | `nosniff` |
| `X-Frame-Options` | `DENY` |
| `Strict-Transport-Security` | `max-age=31536000; includeSubDomains` |
| `Content-Security-Policy` | `default-src 'self'; ...` |
| `X-XSS-Protection` | `0` (let CSP handle it) |
| `Referrer-Policy` | `strict-origin-when-cross-origin` |

## Log Redaction

- **Never log**: Passwords, API keys, credit card numbers, SSNs, or other PII.
- **Redact sensitive fields**: Use a logging filter or middleware that masks fields like `password`, `token`, `authorization`, `ssn`.
- **Audit log output**: Grep logs for patterns that match secrets or PII.

## Environment Variable Security

- **Never use `NEXT_PUBLIC_` or `VITE_` prefixes** for sensitive keys — these are bundled into client-side JavaScript.
- **Never commit `.env` to git**: It should always be in `.gitignore`.
- **If leaked**: Rotate all exposed keys immediately. Use BFG Repo-Cleaner to remove from git history.

## File Upload Security

- **Validate by magic bytes**, not just file extension. Use `python-magic` or `filetype` libraries.
- **Set maximum file sizes**: Reject uploads exceeding the limit at the server level.
- **Store uploads outside the web root** or in object storage (S3, GCS).
- **Generate new filenames**: Never use the user-provided filename directly.

## Input Validation

- **Validate data types**: Use Pydantic models for all API inputs.
- **String lengths**: Set `max_length` on all string fields.
- **Number ranges**: Set `ge`/`le` constraints on numeric fields.
- **Required fields**: Explicitly mark optional fields; assume everything else is required.
- **Reject unexpected fields**: Configure Pydantic to `extra = "forbid"`.

## OAuth Scope

- **Review requested permissions**: Only request scopes your application actually needs.
- **Minimize scope**: `read` is better than `read_write` if you only read data.
- **Audit third-party integrations**: Review what data each OAuth-connected service can access.

## Webhook Signatures

- **Always verify signatures**: Use the provider's official library (e.g., `stripe.webhooks.constructEvent()`).
- **Reject invalid or missing signatures**: Return `400` immediately.
- **Use constant-time comparison**: To prevent timing attacks on signature checks.

## Account Deletion

- **Delete from all systems**: Database tables, file storage (S3), third-party services, linked records, and analytics.
- **Verify completeness**: Query all tables for orphaned records after deletion.
- **Comply with regulations**: GDPR requires "right to be forgotten" within 30 days.

## Database Security

- **Enable Row Level Security (RLS)** on every table in Supabase/Postgres.
- **Never assume AI configured your database securely**: Always verify RLS policies manually.
- **Principle of least privilege**: Database users should only have the permissions they need.

## LLM-Specific Security

- **Prompt injection via tool outputs**: Tool/API responses can contain instructions that manipulate the LLM. Sanitize or clearly delimit tool outputs.
- **PII in LLM prompts**: Never send raw PII to external LLM APIs. Use anonymization or local models for sensitive data.
- **MCP server security**: Restrict tool access control. Run MCP servers in sandboxed environments. Never expose filesystem or network access without explicit allowlists.

## pgvector Security

- **Vector similarity attacks**: Adversarial embeddings can manipulate similarity search results. Validate embedding sources.
- **Embedding inversion risks**: Sufficient embedding queries can reconstruct fragments of the original text. Limit query frequency and log unusual access patterns.
