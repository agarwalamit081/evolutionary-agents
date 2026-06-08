# Update Tests — Examples

## Example 1: Finding all tests that reference a changed function

```bash
# Using the helper script
python scripts/find_affected_tests.py --function process_payment --path .

# Manual ripgrep — search for the function in test directories
rg -n "process_payment" tests/ src/tests/ --type py

# Also check fixtures and conftest files
rg -n "process_payment" --glob "conftest.py" --glob "fixture*"
```

## Example 2: Updating test fixtures after a Pydantic model change

```python
# BEFORE — User model had 'name' and 'email'
# AFTER  — 'name' split into 'first_name' and 'last_name'

# Old fixture (tests/conftest.py)
# @pytest.fixture
# def user_data():
#     return {"name": "Jane Doe", "email": "jane@example.com"}

# Updated fixture
@pytest.fixture
def user_data():
    return {"first_name": "Jane", "last_name": "Doe", "email": "jane@example.com"}


# Updated test
def test_create_user_returns_full_name(user_data):
    user = User(**user_data)
    assert user.full_name == "Jane Doe"
```

## Example 3: Adding a regression test for a bug fix

```python
# Bug: calculate_discount() returned 0 for amounts between 0 and 1
# Fix: handle fractional amounts correctly

def test_calculate_discount_with_fractional_amount():
    """Regression test: discount should apply to amounts between 0 and 1."""
    result = calculate_discount(amount=0.50, rate=0.10)
    assert result == 0.05  # was returning 0 before the fix
```

## Example 4: Updating mock data to match new API response format

```python
# BEFORE — API returned {"status": "ok", "data": {...}}
# AFTER  — API now returns {"status": "ok", "data": {...}, "meta": {"page": 1}}

# Old mock
# @patch("myapp.client.requests.get")
# def test_fetch_orders(mock_get):
#     mock_get.return_value.json.return_value = {"status": "ok", "data": []}

# Updated mock
@patch("myapp.client.requests.get")
def test_fetch_orders(mock_get):
    mock_get.return_value.json.return_value = {
        "status": "ok",
        "data": [],
        "meta": {"page": 1, "total": 0},
    }
    result = fetch_orders()
    assert result.orders == []
    assert result.total_pages == 0
```

## Example 5: Running tests in verbose mode to identify failures

```bash
# Run all tests with verbose output and full tracebacks
pytest -v --tb=long

# Run a single test file for faster iteration
pytest -v tests/test_payment_service.py

# Run with coverage to verify new code is exercised
pytest -v --cov=myapp --cov-report=term-missing tests/

# Stop on first failure for quick debugging
pytest -x -v tests/test_payment_service.py
```

## Example 6: Using pytest -k to run targeted test subsets

```bash
# Run only tests matching "discount" in their name
pytest -v -k "discount"

# Run tests for a specific class
pytest -v -k "TestUserCreation"

# Combine with file path for precision
pytest -v -k "test_calculate" tests/test_pricing.py

# Exclude slow integration tests
pytest -v -k "not integration"
```
