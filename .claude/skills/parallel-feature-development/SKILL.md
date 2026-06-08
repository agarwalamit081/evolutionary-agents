---
name: parallel-feature-development
description: Dispatch independent feature work in parallel git worktrees. Spawns one agent per feature, each in its own isolated worktree, then merges sequentially. Use when 3+ independent features need shipping.
---

**When to Use**
- 3+ independent features that would run serially and waste wall-clock time.
- User explicitly asks for parallel feature development.
- Batch sprint with multiple non-conflicting features.

**Core Principles**
1. **Independence First**: Never dispatch features that touch the same files. Predict file touch-sets BEFORE spawning.
2. **Clean Working Tree Gate**: Refuse to dispatch if uncommitted changes exist.
3. **Sequential Merging**: NEVER merge in parallel — merge one at a time, testing after each.
4. **Test Every Worktree**: Each worktree must pass tests independently before merge is allowed.
5. **Max-Parallel Cap**: Default 4 concurrent agents. Configurable with `--max-parallel`.
6. **Never Force-Push**: Feature branches are never pushed to origin. Only merged base is pushed.

**Workflow**
1. **Safety Checks**: Verify clean working tree, base branch up-to-date, 2+ features.
2. **Independence Analysis**: Predict file touch-sets per feature, build conflict matrix, schedule batches.
3. **Worktree Creation**: `git worktree add` per feature, one branch per feature.
4. **Parallel Dispatch**: Spawn one agent per feature in its worktree. 30-turn limit per agent.
5. **Verification**: Re-run tests + build in each worktree before allowing merge.
6. **Sequential Merge**: `git merge --no-ff` one feature at a time, test after each. Rollback if tests fail.
7. **Cleanup**: Remove merged worktrees and branches. Leave unmerged for manual review.

**References**
- Load `reference.md` for git worktree commands, independence analysis methodology, and conflict resolution.
- Load `examples.md` for dispatch plans, conflict matrices, and output templates.

**Scripts**
- `scripts/worktree_manager.sh`: Worktree CRUD helper (create/verify/merge/cleanup).
