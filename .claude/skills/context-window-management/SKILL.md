---
name: context-window-management
description: Strategies for conserving Claude Code's own context window tokens. Use when context is filling up, working with large codebases, or to proactively prevent token exhaustion.
---

# Context Window Management

> This skill manages Claude Code's own context window budget. For application-level API cost optimization, see `api-cost-optimization`.

## When to Use

- Context is filling up and you need to conserve remaining tokens
- Working with large codebases where full file reads would exhaust context
- Proactively optimizing tool usage to prevent token exhaustion mid-task
- After `/compact` when deciding what to re-read and what to skip

## Core Strategies

1. **Targeted search over full reads**: Use `grep`/`ripgrep` to locate relevant code, then `Read` with `offset`/`limit` for only the lines you need. Never `cat` entire files.
2. **Surgical edits**: Use the `Edit` tool for precise string replacements instead of reading and rewriting entire files. This avoids doubling the file's token cost.
3. **Filtered shell output**: Pipe commands through `head`, `tail`, `grep`, `--stat`, or `--statistics` to get summaries instead of verbose output.
4. **Avoid re-reading**: Trust the Edit tool's success confirmation. Don't re-read files to verify edits — the harness tracks file state.
5. **Compact aggressively**: Run `/compact` when context exceeds 60%. NEVER wait until 80%+ — at that point, the remaining context is insufficient for complex operations. If context exceeds 70%, immediately stop reading new files and compact before proceeding.

## References

- `reference.md` — Deep patterns for file reading, shell discipline, codebase exploration, and exclude configurations
- `examples.md` — Concrete before/after examples of token-efficient tool usage

## Scripts

- `scripts/estimate_context.py` — Estimate token usage for files/directories against model context windows
