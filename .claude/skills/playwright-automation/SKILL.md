---
name: playwright-automation
description: Design and implement robust, deterministic, and maintainable E2E and integration tests using Playwright. Prioritize test stability, fast execution, and clear failure diagnostics.
---

**When to Use**
- Writing new E2E or integration test suites.
- Refactoring brittle tests or implementing the Page Object Model (POM).
- Debugging flaky tests or CI pipeline failures.
- Mocking network requests to isolate frontend behavior.

**Core Principles**
1. **Page Object Model (POM)**: Encapsulate page structure in dedicated classes. Tests read like user stories, not DOM scripts.
2. **Resilient Selectors**: Priority: `getByRole` → `getByTestId` → `getByLabel` → `getByText`. Avoid brittle XPath.
3. **Auto-Waiting**: Rely on Playwright's auto-waiting. **NEVER** use `page.waitForTimeout()`. Use `await expect(locator).toBeVisible()`.
4. **Test Isolation**: Each test runs independently. Use `beforeEach` to reset state.
5. **Network Mocking**: Mock third-party APIs with `page.route()` for fast, deterministic tests.

**Workflow**
1. Identify the user journey or edge case to test.
2. Create or update the Page Object class with typed locators and actions.
3. Write the test spec (Arrange-Act-Assert, visible outcomes).
4. Add `data-testid` attributes to application source if selectors are brittle.
5. Run with `--trace on` for actionable debugging context.

**References**
- Load `reference.md` for Playwright config, test fixtures, browser contexts, parallel execution, and reporters.
- Load `examples.md` for POM, network mocking, API testing, visual regression, and auth patterns.

**Scripts**
- `scripts/generate_pom.py`: Generate Page Object Model class templates. Use `--url <url>` to scan a live page and pre-populate locators.
