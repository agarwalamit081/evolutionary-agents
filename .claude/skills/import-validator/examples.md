# Import Validator — Examples

## Example 1: Find and Fix Unused Python Imports with Ruff

```bash
# Dry-run: list all unused imports without modifying files
ruff check --select F401 src/

# Auto-fix: remove all unused imports in place
ruff check --select F401 --fix src/

# Check specific files for unused + redefined imports
ruff check --select F401,F811 src/models/user.py src/api/routes.py
```

Output example:
```
src/models/user.py:3:8: F401 [*] `os` imported but unused
src/api/routes.py:5:1: F811 [*] `from typing import Optional` redefined
```

## Example 2: Find Unused TypeScript Imports with ESLint

```bash
# Run eslint with no-unused-vars rule on TypeScript files
npx eslint --rule '@typescript-eslint/no-unused-vars: error' src/**/*.ts

# Using tsc to catch missing imports
npx tsc --noEmit
```

ESLint config snippet for strict import checking:
```json
{
  "rules": {
    "@typescript-eslint/no-unused-vars": ["error", { "argsIgnorePattern": "^_" }],
    "no-duplicate-imports": "error"
  }
}
```

## Example 3: Update `__init__.py` When Adding a Python Module

After creating `src/payments/stripe_handler.py`:

```python
# src/payments/__init__.py
from .processor import PaymentProcessor
from .stripe_handler import StripeHandler, create_charge
from .webhook import verify_signature

__all__ = [
    "PaymentProcessor",
    "StripeHandler",
    "create_charge",
    "verify_signature",
]
```

Verify no stale or missing re-exports:
```bash
ruff check --select F401 src/payments/__init__.py
```

## Example 4: Update Barrel File (`index.ts`) for New TypeScript Modules

After creating `src/components/UserAvatar.tsx`:

```typescript
// src/components/index.ts
export { Button } from "./Button";
export type { ButtonProps } from "./Button";
export { Modal } from "./Modal";
export type { ModalProps } from "./Modal";
export { UserAvatar } from "./UserAvatar";
export type { UserAvatarProps } from "./UserAvatar";
```

Verify with TypeScript compiler:
```bash
npx tsc --noEmit --pretty
```

## Example 5: Import Grouping with Ruff Format / Isort Conventions

Correct grouping — stdlib, third-party, local, separated by blank lines:

```python
# stdlib
import json
import os
from pathlib import Path

# third-party
import requests
from fastapi import APIRouter, HTTPException

# local
from app.config import Settings
from app.models import User
from app.services.auth import get_current_user
```

Enforce with ruff:
```bash
# Check import sorting
ruff check --select I001 src/

# Auto-fix import sorting
ruff check --select I001 --fix src/

# Or use ruff format which handles import sorting by default
ruff format src/
```

## Example 6: Using `import type` for TypeScript Type-Only Imports

```typescript
// CORRECT — type-only imports are erased at compile time
import type { User, UserRole } from "./types";
import type { Request, Response } from "express";
import { formatUser, deleteUser } from "./users";

function handleRequest(req: Request, res: Response): void {
    const user: User = formatUser(req.body);
    deleteUser(user.id);
}
```

```typescript
// INCORRECT — types imported as values may cause runtime issues
import { User, UserRole, formatUser, deleteUser } from "./users";
// ^ User and UserRole are types; they should use `import type`
```

ESLint rule to enforce this automatically:
```json
{
  "rules": {
    "@typescript-eslint/consistent-type-imports": ["error", {
      "prefer": "type-imports"
    }]
  }
}
```
