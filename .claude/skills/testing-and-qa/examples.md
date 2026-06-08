---
description: Testing and QA Examples
---

**Example 1: Jest Unit Test with Mocking**

```typescript
describe('UserService', () => {
  it('should throw ValidationError when email is invalid', async () => {
    // Arrange
    const mockDb = { save: jest.fn() };
    const service = new UserService(mockDb);
    const invalidUser = { name: 'John', email: 'invalid-email' };

    // Act & Assert
    await expect(service.createUser(invalidUser)).rejects.toThrow('ValidationError');
    expect(mockDb.save).not.toHaveBeenCalled();
  });
});
```

---

**Example 2: Pytest Integration Test with Test DB**

```python
def test_create_user_integration(test_db_session):
    # Arrange
    user_data = {"username": "testuser", "email": "test@example.com"}

    # Act
    result = create_user(test_db_session, user_data)

    # Assert
    assert result.id is not None
    assert result.email == "test@example.com"

    db_user = test_db_session.query(User).filter_by(email="test@example.com").first()
    assert db_user is not None
```

---

**Example 3: Playwright E2E Test with Page Object Model**

```typescript
import { test, expect } from '@playwright/test';
import { LoginPage } from './pages/LoginPage';

test('should show error for invalid credentials', async ({ page }) => {
  const loginPage = new LoginPage(page);
  await loginPage.goto();
  await loginPage.fillCredentials('user@test.com', 'wrong-password');
  await loginPage.submit();

  await expect(loginPage.errorMessage).toHaveText('Invalid email or password');
});
```

---

**Example 4: API Contract Test**

```python
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_create_order_api(api_client: AsyncClient):
    response = await api_client.post("/api/v1/orders", json={
        "customer_id": "cust-123",
        "items": [{"product_id": "prod-456", "quantity": 2}]
    })

    assert response.status_code == 201
    data = response.json()
    assert "order_id" in data
    assert data["status"] == "created"
    assert data["total"] > 0
```

---

**Example 5: Snapshot Test for UI Component**

```typescript
import { render } from '@testing-library/react';
import { UserCard } from './UserCard';

it('matches snapshot', () => {
  const { container } = render(
    <UserCard name="Jane Doe" email="jane@example.com" role="admin" />
  );
  expect(container).toMatchSnapshot();
});
```

---

**Example 6: Parameterized Edge Case Tests**

```python
import pytest

@pytest.mark.parametrize("input_val,expected", [
    ("", False),                    # empty string
    ("a" * 256, False),             # too long
    ("valid@example.com", True),    # valid
    ("missing@domain", False),      # no TLD
    ("@nodomain.com", False),       # no local part
    (None, False),                  # null input
])
def test_email_validation(input_val, expected):
    assert is_valid_email(input_val) == expected
```

---

**Example 7: Test Fixture with Factory Pattern**

```python
import pytest
from factory import Factory, Faker

class UserFactory(Factory):
    class Meta:
        model = User

    username = Faker("user_name")
    email = Faker("email")
    role = "user"

@pytest.fixture
def admin_user(db_session):
    user = UserFactory(role="admin")
    db_session.add(user)
    db_session.commit()
    return user

def test_admin_can_delete_post(db_session, admin_user):
    post = PostFactory(author_id=admin_user.id)
    result = delete_post(db_session, post.id, admin_user)
    assert result is True
```

---

**Example 8: Actionable Code Review Comments**

```markdown
# Bad: "This is wrong."
# Good:

**Performance**: This N+1 query will fire once per item in the loop.
Consider batching:

```python
# Before (N+1)
for item in items:
    result = db.query(Item).get(item.id)

# After (batch)
results = db.query(Item).filter(Item.id.in_([i.id for i in items])).all()
```

This reduces DB calls from N to 1, which matters at scale.
```
