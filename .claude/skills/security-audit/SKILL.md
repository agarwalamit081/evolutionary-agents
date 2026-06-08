---
name: security-audit
description: Comprehensive security vulnerability audit for code. Use before committing code, after major changes, or when reviewing third-party integrations.
---

# Security Audit Skill

## When to Use

- Before committing code that handles authentication, payments, or user data
- After integrating a new third-party service or library
- When reviewing pull requests that touch API endpoints, database queries, or auth logic
- After major refactors that move or restructure middleware, routes, or data models
- When setting up a new project and establishing security baselines

## Core Categories

| Category | Key Concern |
|---|---|
| **Credentials** | Hardcoded secrets, API keys in client code, `.env` in `.gitignore` |
| **SQL Injection** | Raw SQL with string concatenation or f-strings |
| **XSS** | `dangerouslySetInnerHTML`, unsanitized user input, missing CSP |
| **Authentication** | Missing auth middleware, weak JWT validation, insecure cookies |
| **CORS** | Wildcard origins in production, credentials with wildcard |
| **Dependencies** | Known CVEs, outdated packages with vulnerabilities |
| **Input Validation** | Missing type/length/range checks, unvalidated file uploads |
| **Shell Safety** | `eval()`, `exec()`, `os.system()` with user input |
| **Data Privacy** | PII in logs, missing redaction, exposed internal errors |

## References

- **Deep reference**: Read `reference.md` for comprehensive security rules across 20+ categories including credential management, SQL injection, XSS, auth, CORS, rate limiting, error handling, password hashing, HTTPS, security headers, log redaction, env var safety, file uploads, input validation, OAuth, webhooks, account deletion, database security (RLS), LLM-specific attacks, and pgvector risks.
- **Code patterns**: Read `examples.md` for 7 concrete examples covering secret detection, parameterized queries, Pydantic validation, CORS config, JWT middleware, PII redaction, and dependency auditing.

## Scripts

- **`scripts/security_scan.py`** — Automated regex-based scanner for hardcoded secrets, raw SQL, wildcard CORS, missing auth, insecure hashing, `eval`/`exec` usage, and more. Run it against any file or directory. Does not require external dependencies.
