# Safe Refactor — Examples

## Example 1: Renaming a Function

```bash
# Find all callers of the old function name
python scripts/find_callers.py --function get_user_by_id --type python

# Update all references synchronously
# Definition:
#   def get_user_by_id -> def fetch_user_by_id
# Callers: update every call site
# Tests: rename in test fixtures and assertions
# Exports: update __init__.py

# Verify with ripgrep that no old references remain
rg "get_user_by_id" --type py
```

## Example 2: Changing a Function Signature

```python
# Before:
def create_order(user_id, items):
    ...

# After (adding optional parameter with default):
def create_order(user_id, items, priority="normal"):
    ...

# Assess impact:
#   1. All existing call sites still work (new param has default)
#   2. Tests: add new test cases for the priority parameter
#   3. Type stubs: update the signature in .pyi if present
# Validate:
ruff check src/
mypy src/
pytest tests/test_orders.py -v
```

## Example 3: Moving a Python Module

```bash
# Before: src/api/users.py
# After:  src/api/v2/users.py

# 1. Find all imports of the old path
rg "from src.api.users import" --type py
rg "from src.api import users" --type py

# 2. Update imports in every file
#   from src.api.users import get_user  ->  from src.api.v2.users import get_user

# 3. Update barrel exports in src/api/__init__.py
#   Remove: from .users import get_user
#   Add:    from .v2.users import get_user

# 4. Add deprecation in old location if public API
# src/api/users.py -> from src.api.v2.users import *  # deprecated

# 5. Validate
python -c "from src.api.v2.users import get_user"
mypy src/
pytest tests/ -k user -v
```

## Example 4: Extracting a Method

```python
# Before: long method with a clear sub-task
def process_payment(order):
    validate_order(order)          # sub-task
    charge_credit_card(order)      # sub-task
    send_confirmation_email(order) # sub-task
    update_inventory(order)        # sub-task

# After: extract validation into its own function
def validate_order_for_payment(order):
    """Validate order before payment processing."""
    validate_order(order)

def process_payment(order):
    validate_order_for_payment(order)
    charge_credit_card(order)
    send_confirmation_email(order)
    update_inventory(order)

# Assess: no existing callers of validate_order_for_payment yet
# Action: add tests for the new extracted function
# Validate: ensure process_payment still passes all existing tests
```

## Example 5: Verifying Change Scope with git diff

```bash
# After refactoring, check the scope of changes
git diff --stat
# Expected output:
#   src/api/users.py         | 4 ++--
#   src/api/__init__.py      | 2 +-
#   tests/test_users.py      | 6 +++---
#   3 files changed, 5 insertions(+), 5 deletions(-)

# If the file list is larger than expected, investigate stray changes
git diff --stat | wc -l  # should match the number of files in impact assessment
```

## Example 6: Running Linter + Type Checker + Tests After Refactor

```bash
# Full validation pipeline after any refactoring operation

# 1. Lint
ruff check src/ tests/
# or for TypeScript:
# npx eslint src/ tests/

# 2. Type check
mypy src/ --ignore-missing-imports
# or for TypeScript:
# npx tsc --noEmit

# 3. Run tests (affected files first, then full suite)
pytest tests/test_users.py tests/test_orders.py -v
pytest tests/ -v --tb=short  # full suite as safety net

# 4. Confirm no old references remain
rg "old_function_name" --type py
# Should return zero results
```
