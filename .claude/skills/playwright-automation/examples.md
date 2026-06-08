---
description: Playwright Automation Examples
---

**Example 1: Page Object Model (TypeScript)**

```typescript
import { Page, Locator } from '@playwright/test';

export class LoginPage {
  readonly page: Page;
  readonly usernameInput: Locator;
  readonly passwordInput: Locator;
  readonly submitButton: Locator;

  constructor(page: Page) {
    this.page = page;
    this.usernameInput = page.getByLabel('Username');
    this.passwordInput = page.getByLabel('Password');
    this.submitButton = page.getByRole('button', { name: 'Sign In' });
  }

  async navigate() {
    await this.page.goto('/login');
  }

  async login(username: string, password: string) {
    await this.usernameInput.fill(username);
    await this.passwordInput.fill(password);
    await this.submitButton.click();
  }
}
```

---

**Example 2: Test Spec with Arrange-Act-Assert**

```typescript
import { test, expect } from '@playwright/test';
import { LoginPage } from '../pages/LoginPage';

test('successful login redirects to dashboard', async ({ page }) => {
  const loginPage = new LoginPage(page);
  await loginPage.navigate();

  // Act
  await loginPage.login('testuser', 'securepassword');

  // Assert
  await expect(page).toHaveURL('/dashboard');
  await expect(page.getByRole('heading', { name: 'Welcome' })).toBeVisible();
});
```

---

**Example 3: Network Mocking with page.route()**

```typescript
test('displays error when API returns 500', async ({ page }) => {
  // Mock the API to return a server error
  await page.route('**/api/users', (route) =>
    route.fulfill({
      status: 500,
      contentType: 'application/json',
      body: JSON.stringify({ error: 'Internal Server Error' }),
    })
  );

  await page.goto('/users');

  await expect(page.getByText('Something went wrong')).toBeVisible();
  // Verify the request was made
  await page.unroute('**/api/users');
});
```

---

**Example 4: API Testing with request Fixture**

```typescript
import { test, expect } from '@playwright/test';

test('API: create user returns 201', async ({ request }) => {
  const response = await request.post('/api/users', {
    data: { name: 'Jane Doe', email: 'jane@example.com' },
  });

  expect(response.ok()).toBeTruthy();
  expect(response.status()).toBe(201);

  const body = await response.json();
  expect(body.name).toBe('Jane Doe');
  expect(body.id).toBeDefined();
});
```

---

**Example 5: Visual Regression Testing**

```typescript
import { test, expect } from '@playwright/test';

test('dashboard matches visual snapshot', async ({ page }) => {
  await page.goto('/dashboard');
  await expect(page).toHaveScreenshot('dashboard.png', {
    maxDiffPixelRatio: 0.01,  // Allow 1% difference
    fullPage: true,
  });
});

// Update snapshots: npx playwright test --update-snapshots
```

---

**Example 6: Authentication State Reuse (globalSetup)**

```typescript
// global-setup.ts
import { chromium, FullConfig } from '@playwright/test';

async function globalSetup(config: FullConfig) {
  const browser = await chromium.launch();
  const page = await browser.newPage();

  await page.goto('/login');
  await page.fill('[data-testid="email"]', 'admin@test.com');
  await page.fill('[data-testid="password"]', 'password');
  await page.click('[data-testid="submit"]');
  await page.waitForURL('/dashboard');

  // Save auth state
  await page.context().storageState({ path: '.auth/admin.json' });
  await browser.close();
}

export default globalSetup;
```

```typescript
// playwright.config.ts — use saved state
projects: [
  {
    name: 'authenticated',
    use: {
      storageState: '.auth/admin.json',
    },
  },
]
```

---

**Example 7: Cross-Browser Parameterized Test**

```typescript
// playwright.config.ts — multi-browser projects
projects: [
  { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
  { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
  { name: 'webkit', use: { ...devices['Desktop Safari'] } },
  { name: 'mobile', use: { ...devices['iPhone 13'] } },
]

// Tests run automatically across all projects
test('homepage loads on all browsers', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Welcome' })).toBeVisible();
});
```
