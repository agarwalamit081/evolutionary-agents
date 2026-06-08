---
description: Backend Patterns Reference
---

## Layered Architecture

| Layer | Responsibility | Example |
|---|---|---|
| Controller | Routing, input validation, HTTP status codes | Express router, FastAPI endpoint |
| Service | Business logic, orchestration | UserService, PaymentService |
| Repository | Data access, query construction | UserRepo, Prisma client wrapper |

**Rule**: Controllers never contain business logic. Services never contain SQL queries. Repositories never contain business logic.

## Authentication & Authorization

- **JWT**: Stateless auth. Short-lived access token (15min) + long-lived refresh token (7d).
- **API Keys**: For service-to-service auth. Rotate regularly, hash before storage.
- **OAuth2**: For third-party identity delegation (Google, GitHub login).
- **RBAC**: Role-Based Access Control — assign permissions to roles, roles to users.
- **Principle of Least Privilege**: Default deny, grant minimum access needed.

## Error Handling (RFC 7807 Problem Details)

```json
{
  "type": "https://example.com/errors/validation-error",
  "title": "Validation Error",
  "status": 400,
  "detail": "Email format is invalid",
  "instance": "/api/v1/users",
  "trace_id": "abc-123"
}
```

**Error Code Taxonomy**: validation_error (400), not_found (404), permission_denied (403), conflict (409), rate_limited (429), internal_error (500).

## Background Jobs & Queues

| Pattern | When to Use |
|---|---|
| Immediate async | Send email after signup (fire-and-forget) |
| Scheduled | Daily report generation |
| Recurring | Cleanup expired sessions every hour |
| Heavy processing | Video transcoding, report generation |

**Best practices**: Use idempotency keys, implement dead letter queues, retry with exponential backoff (max 3), log job start/end/duration.

## Rate Limiting

| Strategy | How | Best For |
|---|---|---|
| Token bucket | Refill tokens over time | General purpose, bursty traffic |
| Sliding window | Count requests in rolling window | Precise per-second limits |
| Fixed window | Count in discrete time windows | Simple, approximate |

Apply per-user and per-IP. Return `429 Too Many Requests` with `Retry-After` header.

## Caching at Service Layer

- **Cache-aside**: Service checks cache → miss → fetch from DB → populate cache.
- **Invalidation**: On write, delete cache key (lazy re-population).
- **TTL Guidance**: User settings (1h), product catalog (5min), API responses (30s).
- **Never cache authenticated user-specific data in shared cache.**

## Logging & Observability

- **Structured logging**: JSON format with `timestamp`, `level`, `message`, `trace_id`, `service`.
- **Correlation IDs**: Attach to every request at the edge, propagate through all services.
- **Log levels**: ERROR (action required), WARN (investigate), INFO (business events), DEBUG (dev only).
- **Never log**: Passwords, tokens, PII, full request bodies.

## API Versioning

| Strategy | URL Example | Pros/Cons |
|---|---|---|
| URL path | `/api/v1/users` | Explicit, easy to route |
| Header | `Accept: application/vnd.api.v1+json` | Clean URLs, harder to debug |
| Query param | `/api/users?version=1` | Simple, not RESTful |

Prefer URL path versioning for clarity.
