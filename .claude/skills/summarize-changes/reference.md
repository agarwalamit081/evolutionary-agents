---
description: Summarize Changes Reference
---

## Git Commands for Different Scopes

| Command | Scope |
|---|---|
| `git diff HEAD` | All uncommitted changes (staged + unstaged) |
| `git diff --cached` | Staged changes only |
| `git diff` | Unstaged changes only |
| `git diff main...HEAD` | All changes on current branch vs main |
| `git diff --stat` | File-level summary (no line details) |
| `git log --oneline -10` | Recent commits for context |

## Risk Taxonomy

### Security
- Hardcoded API keys, tokens, passwords, connection strings
- SQL injection (string concatenation in queries)
- Auth bypass (removed auth checks, broadened permissions)
- Exposed PII in logs or API responses

### Data Loss
- `DROP TABLE`, destructive migrations without rollback
- Hard deletes instead of soft deletes
- Removed backup/recovery logic

### Breaking Changes
- Changed API signatures (renamed/removed parameters)
- Removed exports or public methods
- Changed response format or status codes
- Modified database column types without migration

### Performance
- N+1 queries introduced
- Missing database indexes for new WHERE clauses
- Large file additions (binary assets, generated code)
- Synchronous operations that should be async

### Correctness
- Missing error handling (try/catch, error boundaries)
- Race conditions (concurrent access without locks)
- Off-by-one errors in loops or pagination
- Missing null/undefined checks

## Commit Message Conventions

Format: `type(scope): description`

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`
Breaking: `feat(api)!: change authentication flow`

## Diff Size Handling

| Size | Lines | Approach |
|---|---|---|
| Small | <100 | Full analysis with line-level risk flags |
| Medium | 100-500 | Grouped summary by module, risk highlights |
| Large | >500 | High-level file groups, critical risks only |

## PR Description Template

```markdown
## What
[2-3 sentences summarizing the change]

## Why
[Business context / problem being solved]

## Changes
- [Grouped bullet points by area]

## Testing
- [How to verify]

## Risks
- [Any risks or follow-ups needed]
```
