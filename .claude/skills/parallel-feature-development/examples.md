---
description: Parallel Feature Development Examples
---

**Example 1: Dispatch Plan Output**

```
Parallel plan (4 features in 2 batches):
  Batch 1 (3 features): dark-mode-toggle, search-autocomplete, export-csv
  Batch 2 (1 feature):  refactor-auth-module (touches auth.ts shared with batch 1 features)

Estimated wall-clock saving vs serial: ~45%
```

---

**Example 2: Conflict Matrix**

```
Touch-set prediction:
  dark-mode-toggle:    [theme.ts, ThemeContext.tsx, Header.tsx]
  search-autocomplete: [SearchBar.tsx, api/search.ts, hooks/useSearch.ts]
  export-csv:          [utils/csv.ts, ExportButton.tsx, api/reports.ts]
  refactor-auth:       [auth.ts, api/auth.ts, middleware.ts, auth.test.ts]

Conflict matrix:
                dark   search  export  refactor
  dark-mode      -      -       -        -
  search         -      -       -        -
  export         -      -       -        -
  refactor       -      -       -        -

Schedule: All 4 features in Batch 1 (no conflicts detected).
```

---

**Example 3: Per-Feature Agent Outcome Table**

| # | Feature | Worktree | Branch | Status | Commit | Notes |
|---|---------|----------|--------|--------|--------|-------|
| 1 | dark-mode-toggle | ../worktrees/dark-mode | feat/dark-mode | ✅ Success | abc1234 | ThemeContext + toggle button |
| 2 | search-autocomplete | ../worktrees/search-auto | feat/search-auto | ✅ Success | def5678 | Debounced API hook + dropdown |
| 3 | export-csv | ../worktrees/export-csv | feat/export-csv | ✅ Success | ghi9012 | CSV util + export button |
| 4 | refactor-auth | ../worktrees/refactor-auth | feat/refactor-auth | ❌ Blocked | — | Missing env var, needs manual setup |

---

**Example 4: Post-Merge Failure and Rollback**

```
Merging feat/dark-mode → main ... ✅ Tests pass.
Merging feat/search-auto → main ... ❌ Tests FAILED.
  - auth.test.ts: 2 failures (search hook conflicts with auth middleware)
  - Action: git reset --hard ORIG_HEAD (undo merge)
  - feat/search-auto left unmerged for manual review.
Merging feat/export-csv → main ... ✅ Tests pass.

Result: 3/4 merged successfully. search-auto needs manual reconciliation.
```

---

**Example 5: Final Report**

```markdown
## Parallel Features Report — 2025-01-15

### Plan
- Features: 4
- Batches: 1 (max parallel: 4)
- Estimated saving vs serial: ~45%

### Per-Feature Outcomes
| # | Feature | Status | Commit |
|---|---------|--------|--------|
| 1 | dark-mode-toggle | ✅ Merged | abc1234 |
| 2 | export-csv | ✅ Merged | ghi9012 |
| 3 | refactor-auth | ❌ Blocked | — |
| 4 | search-autocomplete | ❌ Merge conflict | def5678 |

### Merged to main
- 2 features merged sequentially, base pushed.

### Manual Review Required
- refactor-auth — agent blocked on missing env var — branch at `feat/refactor-auth`
- search-autocomplete — merge conflict with dark-mode in auth.test.ts — branch at `feat/search-auto`

### Wall-Clock
- Actual: 8m
- Serial estimate: 16m
- Saving: 50%
```

---

**Example 6: Dry Run Output**

```
$ parallel-feature-development "dark mode; search autocomplete; export CSV; refactor auth" --dry-run

=== DRY RUN — No worktrees will be created ===

Features parsed:
  1. dark mode
  2. search autocomplete
  3. export CSV
  4. refactor auth

Touch-set prediction:
  1. dark mode → [theme.ts, ThemeContext.tsx, Header.tsx]
  2. search autocomplete → [SearchBar.tsx, api/search.ts, hooks/useSearch.ts]
  3. export CSV → [utils/csv.ts, ExportButton.tsx, api/reports.ts]
  4. refactor auth → [auth.ts, api/auth.ts, middleware.ts, auth.test.ts]

Schedule: Batch 1 (all 4) — no conflicts detected.
Estimated saving: ~45% wall-clock.

Run without --dry-run to proceed.
```
