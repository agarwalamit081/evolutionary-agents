# Update Tests — Reference

## 5-Step Test Update Workflow

### Step 1: Baseline

Run the existing test suite and record which tests pass. This establishes a known-good state and prevents you from accidentally attributing pre-existing failures to your change.

```
pytest --tb=short -q 2>&1 | tee /tmp/baseline.txt
```

### Step 2: Identify Affected Tests

Use `scripts/find_affected_tests.py` or manual ripgrep to find every file that touches the changed symbol:

```bash
# Using the script
python scripts/find_affected_tests.py --function UserService.create --path .

# Manual ripgrep
rg -t py "UserService" tests/
rg -t py "create" tests/  # narrower if needed
```

Categorize findings:
- **Direct references** — test functions that call the changed symbol directly.
- **Fixture references** — `conftest.py` or fixture functions that construct or mock the symbol.
- **Import references** — files that import the symbol but may not exercise it directly.

All three categories need review.

### Step 3: Update Call Sites

For each affected test file, update:

1. **Imports** — If the symbol moved or was renamed, fix the import path.
2. **Argument lists** — Add/remove/reorder arguments to match the new signature.
3. **Fixture definitions** — Update factory defaults, fixture return values, and `conftest.py` setup.
4. **Mock data** — Adjust `Mock`, `patch`, or response fixtures to match the new return shape or side effects.
5. **Assertions** — Update expected values if the return type or business logic changed.

### Step 4: Add New Tests

Add tests for every new behavior your change introduces:

| Category | What to cover |
|---|---|
| Unit | Each new parameter, each new branch in logic |
| Edge case | Boundary values, empty inputs, None/null, zero-length collections |
| Error handling | New exceptions raised, validation failures, invalid input |
| Integration | End-to-end path through the changed component |
| Regression | Re-produce the original bug scenario, assert it is fixed |

### Step 5: Run and Fix

```bash
pytest -v --tb=long
```

- If a test fails: read the failure, fix the **application code**, re-run.
- Repeat until all tests pass.
- Never `@skip`, `xit`, or delete a test to make the suite green.

## Regression Testing Pattern

When fixing a bug:

1. Write a test that reproduces the bug against the **unfixed** code (it should fail).
2. Apply the fix to the application code.
3. Re-run the test — it should now pass.
4. Include both the test and the fix in the same commit.

## Test Quality Standards

### Naming

Use descriptive names that state the expected behavior:

```python
# Good
def test_create_user_raises_value_error_when_email_is_empty():
    ...

# Bad
def test_user():
    ...
```

### Isolation

Each test must be independent. No shared mutable state between tests. Use fixtures or factories for setup, and clean up in teardown or via fixture scoping.

### Determinism

Tests must produce the same result on every run. Avoid:
- Depending on system time (mock `datetime`).
- Depending on random values (seed your RNG or mock it).
- Depending on external services (mock HTTP calls).
- Depending on test execution order.

### Mocking

- Mock at the **boundary** (external services, I/O), not at internal collaborators.
- Prefer `patch` on the module under test's namespace (`@patch("myapp.service.requests.get")`).
- Verify mock call counts and arguments when the side effect is the primary outcome.

## Cross-Reference

For broader testing methodology (test pyramid, coverage targets, CI integration), see the `testing-and-qa` skill.
