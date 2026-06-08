# Circular Dependency Check — Examples

## Example 1: Detecting a Python Circular Import via importlib Test

```python
"""Test that reveals a circular import at module load time."""
import importlib
import sys
import pytest


def test_no_circular_imports():
    """Fresh-import every module; any cycle raises ImportError."""
    modules_to_test = [
        "myapp.users",
        "myapp.orders",
        "myapp.notifications",
    ]
    for mod_name in modules_to_test:
        # Remove cached module to force a fresh import
        sys.modules.pop(mod_name, None)
        importlib.import_module(mod_name)  # raises ImportError if cycle exists
```

## Example 2: Refactoring Circular Imports by Extracting a Shared Module

```python
# Before (cycle): users.py imports from orders.py, orders.py imports from users.py

# users.py
from myapp.orders import get_user_orders  # circular!

# orders.py
from myapp.users import get_user_by_id   # circular!

# After: extract shared logic into myapp/shared/queries.py
# myapp/shared/queries.py  (no imports from users or orders)

def get_user_by_id(user_id: int):
    ...

def get_user_orders(user_id: int):
    ...

# users.py
from myapp.shared.queries import get_user_by_id  # one-way, no cycle

# orders.py
from myapp.shared.queries import get_user_orders  # one-way, no cycle
```

## Example 3: Python Lazy Import Pattern

```python
# orders.py — avoids circular import by deferring the import

from myapp.db import get_session

# Do NOT do: from myapp.users import get_user_by_id  (causes cycle)


def create_order(user_id: int, items: list):
    # Lazy import: only loaded when this function is called
    from myapp.users import get_user_by_id

    user = get_user_by_id(user_id)
    session = get_session()
    order = session.add(order_from(user, items))
    session.commit()
    return order
```

## Example 4: Python Dependency Injection to Break a Cycle

```python
# Before (cycle): notifier.py imports email_client.py, email_client.py imports notifier.py

# notifier.py
class Notifier:
    def __init__(self, send_fn=None):
        # Inject the sending function instead of importing the module
        self._send = send_fn

    def notify(self, user, message):
        self._send(user.email, message)

# wiring.py (composition root, imports both)
from myapp.notifier import Notifier
from myapp.email_client import send_email

notifier = Notifier(send_fn=send_email)
```

## Example 5: TypeScript Using Madge for Circular Detection

```bash
# Install and run madge
npx madge --circular src/

# Sample output:
# 1) src/users/service.ts > src/orders/service.ts > src/users/service.ts
# 2) src/logger.ts > src/config.ts > src/logger.ts

# JSON output for programmatic use
npx madge --json src/ | jq '.cycles'
```

```json
// package.json — add as a CI script
{
  "scripts": {
    "check:cycles": "madge --circular src/ && echo 'No cycles found'"
  }
}
```

## Example 6: TypeScript Interface Extraction to Break a Cycle

```typescript
// Before (cycle): user.service.ts imports order.service.ts, and vice-versa

// After: extract interfaces into a separate types module

// --- types/user.ts (no business logic imports)
export interface User {
  id: string;
  name: string;
  email: string;
}

// --- types/order.ts (no business logic imports)
export interface Order {
  id: string;
  userId: string;
  total: number;
}

// --- user.service.ts
import type { User } from '../types/user';
import type { Order } from '../types/order';
// No import from order.service.ts

export async function getUserOrders(userId: string): Promise<Order[]> {
  const resp = await fetch(`/api/users/${userId}/orders`);
  return resp.json();
}

// --- order.service.ts
import type { Order } from '../types/order';
import type { User } from '../types/user';
// No import from user.service.ts

export async function getOrderUser(orderId: string): Promise<User> {
  const resp = await fetch(`/api/orders/${orderId}/user`);
  return resp.json();
}
```
