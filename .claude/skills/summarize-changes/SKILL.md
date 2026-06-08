---
name: summarize-changes
description: Summarize uncommitted or branch changes and flag risky patterns — missing error handling, hardcoded secrets, breaking API changes, missing tests.
---

**When to Use**
- User asks "what changed?" or "review my diff".
- Generating a commit message from staged/unstaged changes.
- Writing a PR description from a branch diff.
- Pre-commit review to catch risky patterns.

**Core Principles**
1. **Risk-First Analysis**: Always flag security issues, data-loss risks, breaking changes, and performance regressions before summarizing.
2. **Structured Output**: Summary → Risks → Suggested commit message. Consistent format every time.
3. **Diff Scope Awareness**: Adapt analysis depth to diff size (small: full detail, large: grouped summary).
4. **Actionable**: Every risk flag should include a specific fix suggestion.
5. **Concise**: 2-5 bullet summary, not a line-by-line walkthrough.

**Workflow**
1. Determine diff scope: staged only, all uncommitted, or branch vs base.
2. Read the diff.
3. Categorize changes by module/area.
4. Identify risks (check `reference.md` for risk taxonomy).
5. Generate summary + risk flags + commit message suggestion.

**References**
- Load `reference.md` for git diff commands, risk taxonomy, and commit message conventions.

**Scripts**
- `scripts/summarize_diff.sh`: Diff wrapper with scope/format options.
