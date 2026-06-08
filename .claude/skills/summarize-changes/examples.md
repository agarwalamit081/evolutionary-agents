---
description: Summarize Changes Examples
---

**Example 1: Clean Feature Addition (Small Diff)**

```
Summary:
- Added email validation to the signup form (src/auth/validate.ts)
- Updated SignupForm component with error state display (src/components/SignupForm.tsx)
- Added 3 unit tests for edge cases (src/auth/validate.test.ts)

Risks: None detected.

Suggested commit: feat(auth): add email validation to signup form
```

---

**Example 2: Risky Changes Detected**

```
Summary:
- Refactored payment processing to use direct SQL (src/payments/service.ts)
- Removed error middleware from payment routes (src/routes/payments.ts)

⚠ RISKS:
1. [SECURITY] SQL injection risk — string interpolation in query at service.ts:47
   Fix: Use parameterized queries ($1, $2) instead of string formatting.
2. [CORRECTNESS] No error handling on payment route — unhandled exceptions will crash the server.
   Fix: Re-add error middleware or wrap handler in try/catch.
3. [TESTS] No tests updated for the refactored payment flow.
   Fix: Update src/payments/service.test.ts to cover new SQL-based flow.

Suggested commit: refactor(payments): migrate to direct SQL queries
⚠ Fix risks before committing.
```

---

**Example 3: Large Refactor (Grouped Summary)**

```
Summary (12 files changed, 340 insertions, 89 deletions):
- Auth module: Refactored to use JWT instead of sessions (4 files)
- API routes: Standardized error response format (3 files)
- Database: Added user_sessions table migration (2 files)
- Tests: Updated auth tests for JWT flow (3 files)

Risks:
- [BREAKING] `/api/auth/login` response format changed — verify API consumers can handle new format.

Suggested commit: refactor(auth): migrate from session-based to JWT authentication
```

---

**Example 4: Database Migration**

```
Summary:
- Added migration 004: create_orders_table with status, total, user_id columns
- Added index on user_id for query performance
- Included rollback (DROP TABLE)

Risks:
- [DATA LOSS] Down migration drops the orders table — ensure no production data exists before rollback.
- [PERFORMANCE] Consider adding index on `status` if filtering by status is common.

Suggested commit: feat(db): add orders table with indexes
```

---

**Example 5: Empty Diff**

```
No uncommitted changes detected.

- Working tree is clean (all changes committed)
- Current branch: feature/user-profiles
- Last commit: feat(profiles): add avatar upload (abc1234)
```

---

**Example 6: PR Description from Branch Diff**

```markdown
## What
Add user profile management with avatar upload, bio editing, and public profile pages.

## Why
Users cannot currently customize their profiles or add identifying information beyond email and name.

## Changes
- **API**: New endpoints `GET/PUT /api/users/:id/profile` with avatar URL and bio fields
- **Database**: Added `avatar_url`, `bio`, `public_profile` columns to users table
- **Frontend**: New ProfileSettings component with image upload and bio editor
- **Tests**: 12 new tests (8 unit, 4 integration)

## Testing
1. `pytest tests/` — all 89 tests passing
2. Manual: Upload avatar at /settings/profile, verify at /users/:id

## Risks
- Avatar images stored locally — consider S3 migration before production launch
```
