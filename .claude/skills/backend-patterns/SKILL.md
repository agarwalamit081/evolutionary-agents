---
name: backend-patterns
description: Design and implement enterprise-grade backend systems emphasizing security, scalability, observability, and clean architecture. Covers application-layer patterns (controllers, services, auth, jobs, rate limiting). For database-specific patterns, see backend-and-db-patterns.
---

**When to Use**
- Creating new API endpoints, services, or database models.
- Refactoring legacy backend code.
- Implementing authentication, authorization, or background jobs.
- Designing error handling, rate limiting, or event-driven systems.

**Core Principles**
1. **Separation of Concerns**: Controllers (routing/validation) → Services (business logic) → Repositories (data access).
2. **Fail Fast & Explicit Errors**: Validate inputs early. Return standardized error responses (RFC 7807 Problem Details).
3. **Idempotency**: Ensure retryable operations (especially POST/PUT) do not cause duplicate side effects.
4. **Observability**: Log with context (correlation IDs). Never log PII or secrets.
5. **Security First**: Parameterized queries, strict CORS, rate limiting, principle of least privilege.

**Workflow**
1. Define the API contract (request/response schema).
2. Implement input validation at the edge (controller).
3. Delegate to a service layer for business logic.
4. Use a repository/ORM layer for data persistence.
5. Add structured logging and error handling wrappers.

**References**
- Load `reference.md` for layered architecture, auth patterns, error handling, background jobs, rate limiting, caching, and API versioning.
- Load `examples.md` for controller/service patterns, error middleware, background jobs, and rate limiters.

**Scripts**
- `scripts/generate_crud.py`: Scaffold service classes in Python or TypeScript with optional repository pattern.
