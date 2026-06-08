---
description: Playwright Automation Reference
---

## Playwright Config (playwright.config.ts)

```typescript
import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  timeout: 30000,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: process.env.CI
    ? [['html', { open: 'never' }], ['github']]
    : 'html',
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
    { name: 'webkit', use: { ...devices['Desktop Safari'] } },
  ],
  webServer: {
    command: 'npm run dev',
    port: 3000,
    reuseExistingServer: !process.env.CI,
  },
});
```

## Test Fixtures

Custom fixtures for authenticated pages, database seeding, etc.

```typescript
import { test as base } from '@playwright/test';

type Fixtures = {
  authenticatedPage: Page;
};

export const test = base.extend<Fixtures>({
  authenticatedPage: async ({ page }, use) => {
    await page.goto('/login');
    await page.fill('[data-testid="email"]', 'test@example.com');
    await page.fill('[data-testid="password"]', 'password123');
    await page.click('[data-testid="submit"]');
    await page.waitForURL('/dashboard');
    await use(page);
  },
});
```

## Browser Contexts

- `browser.newContext()` for isolated sessions with fresh cookies/storage.
- `browser.newContext({ storageState: 'auth.json' })` to reuse auth state across tests.
- Save auth state: `await page.context().storageState({ path: 'auth.json' })`.

## Parallel Execution

- `workers` in config: number of parallel workers (default: half of CPU cores).
- `test.describe.configure({ mode: 'parallel' })` for parallel within a file.
- `test.describe.configure({ mode: 'serial' })` for dependent tests (must run in order).

## Reporters

| Reporter | Use Case |
|---|---|
| `html` | Local development (view in browser) |
| `github` | GitHub Actions annotations in PR |
| `json` | Custom processing or CI dashboards |
| `junit` | Jenkins, CircleCI, GitLab integration |

## Locator Priority

1. `getByRole('button', { name: 'Submit' })` — accessibility-aligned
2. `getByTestId('submit-btn')` — most resilient, requires app instrumentation
3. `getByLabel('Email')` — form inputs
4. `getByText('Welcome')` — static text
5. `getByPlaceholder('Enter email')` — input placeholders
6. CSS/XPath — last resort, brittle

## Assertion Types

- **Page assertions**: `expect(page).toHaveTitle()`, `expect(page).toHaveURL()`
- **Locator assertions**: `expect(locator).toBeVisible()`, `expect(locator).toHaveText()`
- **Soft assertions**: `expect.soft(locator).toBeVisible()` — doesn't stop test on failure
- **API response assertions**: `expect(response).toBeOK()`

## Anti-Patterns

- Using `page.$()` instead of `page.locator()` — `$` doesn't auto-wait.
- Testing implementation details (CSS classes, internal state) instead of user-visible behavior.
- Shared mutable state between tests (causes flakiness).
- Hardcoded waits: `page.waitForTimeout(5000)` — always use `expect` or `waitFor`.
