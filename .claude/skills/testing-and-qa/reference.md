---
description: Testing and QA Reference
---

## Mocking Strategies

- **Unit Tests**: Mock at the module/function boundary. Do not mock the system under test.
- **Integration Tests**: Use test containers or in-memory databases (SQLite, Redis memory) instead of heavy mocking.

## Anti-Patterns to Avoid

- Testing implementation details (checking if an internal function was called) instead of observable behavior.
- Over-mocking (mocking everything makes tests brittle and useless).
- Shared state between tests (always use `beforeEach`/`afterEach` to reset state).
- Testing only happy paths — always include error and edge cases.

## Naming Conventions

- **File**: `[FileName].test.ts` or `test_[filename].py`
- **Describe blocks**: Group by feature or component.
- **It blocks**: `should return 404 when resource is not found`

## Code Review Checklist

- [ ] **Bugs/Edge Cases**: Null/undefined inputs, empty arrays, concurrent access, boundary values.
- [ ] **Security**: SQL injection, XSS, auth bypass, PII exposure in logs.
- [ ] **Performance**: N+1 queries, missing indexes, unnecessary re-renders, memory leaks.
- [ ] **Readability**: Clear naming, consistent style, no magic numbers.
- [ ] **Error Handling**: All failure paths covered, meaningful error messages.

## Quality Gates for PRs

1. All tests pass (unit + integration).
2. Coverage ≥ 80% on changed files.
3. No critical or high-severity security findings.
4. Code review approved by at least one peer.
5. No regression on E2E smoke tests.
