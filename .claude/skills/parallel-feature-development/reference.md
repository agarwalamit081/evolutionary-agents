---
description: Parallel Feature Development Reference
---

## Git Worktree Commands

```bash
# Create worktree with new branch
git worktree add ../worktrees/<slug> -b feat/<slug>

# List all worktrees
git worktree list

# Remove worktree (must be clean)
git worktree remove ../worktrees/<slug>

# Force remove (if dirty — use with caution)
git worktree remove --force ../worktrees/<slug>

# Prune stale worktree references
git worktree prune
```

**Gotchas:**
- Cannot have the same branch checked out in multiple worktrees.
- Worktree directories are siblings to the main repo, not inside `.git`.
- Removing a worktree does NOT delete the branch — use `git branch -d` separately.

## Independence Analysis Methodology

1. **File Touch-Set Prediction**: For each feature, use a Plan agent to predict which files will be modified.
2. **Conflict Matrix**: Build an N×N matrix where entry (i,j) = files touched by both feature i and feature j.
3. **Hard Conflict**: Same file predicted for both features → cannot run in parallel.
4. **Soft Conflict**: Sibling files in the same module → allow with warning.
5. **Batch Scheduling**: All non-conflicting features → Batch 1. Features conflicting with Batch 1 → Batch 2. Repeat.

## Merge Strategy

1. Always use `git merge --no-ff feat/<slug>` to create a visible merge commit.
2. After each merge, re-run the full test suite.
3. If post-merge tests fail: `git reset --hard ORIG_HEAD` to undo the merge.
4. Leave the failed feature branch unmerged for manual reconciliation.
5. After all successful merges: `git push origin <base>` once.

## Conflict Resolution

When batch 2 features conflict with batch 1 results:
1. Attempt auto-merge (git usually handles this for non-overlapping changes).
2. If merge conflict: do NOT attempt automatic resolution. Leave for manual review.
3. Report the conflicting files and branches to the user.

## Agent Dispatch Template

Each agent receives a self-contained prompt:
```
You are implementing one feature in an isolated git worktree at <path>.
Feature: <description>
Steps:
1. cd <worktree-path>
2. Implement the feature following the repo's CLAUDE.md.
3. Run the project's test suite. Tests must pass before commit.
4. Commit using the project's commit message conventions.
5. Report: success/failure, summary of changes, commit SHA.
Max 30 turns. If blocked needing user input, STOP and report.
```

## Edge Cases

- **Documentation-only features**: Touch 0 code files. Safe to parallelize with anything.
- **No test suite**: Skip verification step, warn about risk.
- **External commits mid-dispatch**: Pull before each merge, re-verify.
