---
name: testing-and-qa
description: Comprehensive testing strategy and code quality review — unit, integration, E2E testing alongside code reviews for bugs, security, and performance.
---

**When to Use**
- Writing unit, integration, or E2E tests.
- Refactoring code while preserving behavior.
- Debugging flaky tests.
- Reviewing code for bugs, security, or performance issues.

**Core Principles**
1. **Arrange-Act-Assert (AAA)**: Strictly structure tests into setup, execution, and verification phases.
2. **Isolation**: Mock external dependencies (DB, network, file system). Tests must not depend on execution order.
3. **Descriptive Naming**: `it('should [expected behavior] when [condition]', ...)`.
4. **Cover Happy + Sad Paths**: Always test happy paths, edge cases, and error/failure states.
5. **Deterministic**: Avoid `setTimeout` or arbitrary waits. Use proper async/await and mocking.
6. **Test Behavior, Not Implementation**: Check observable outputs, not internal function calls.
7. **Quality Review Dimensions**: Bugs/edge cases, security concerns, performance, readability.
8. **Run Tests via uv**: Always execute tests with `uv run python -m pytest`. Never use bare `pytest` or `python -m pytest`.

**Test Pyramid**
- **Unit** (70%): Fast, isolated, mock at boundaries.
- **Integration** (20%): Test containers, in-memory DBs, real API contracts.
- **E2E** (10%): Playwright/Cypress, critical user journeys only.

**References**
- Load `reference.md` for mocking strategies, anti-patterns, naming conventions, and code review checklists.
- Load `examples.md` for test patterns across TypeScript and Python.

**Scripts**
- `scripts/generate_test_boilerplate.py`: Scaffold test files for .ts/.js/.py.
- `scripts/coverage_report.py`: Conceptual scaffold for coverage summary.
