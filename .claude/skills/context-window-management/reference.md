# Context Window Management — Reference

> This skill manages Claude Code's context window budget. For application-level API cost optimization, see `api-cost-optimization`.

## File Reading Optimization

- **Prefer targeted grep over full file reads.** Use `ripgrep` (`rg`) or `grep -rn` to locate the exact lines you need, then read only that region with `offset` and `limit` parameters.
- **Use `Read` with `offset`/`limit` for large files.** A 2000-line file consumes roughly 50k tokens — read only the sections relevant to your task.
- **Never re-read a file after editing.** The Edit tool confirms success or errors. The harness tracks file state. Re-reading wastes tokens for zero new information.
- **Avoid reading generated/derivative files.** Skip `package-lock.json`, `yarn.lock`, `.min.js`, compiled output, and vendored dependencies. These are huge and rarely relevant.

## Edit Efficiency

- **Always use `Edit` for targeted changes.** Replacing a 10-line function via Edit costs ~10 lines of tokens. Rewriting the entire file via Write costs 2x the file size (old + new).
- **Use `replace_all: true` for repeated patterns** instead of issuing multiple Edit calls for the same substitution.
- **Batch related edits.** Plan all changes to a file, then issue them sequentially. Each Edit call is cheaper than a full Write.

## Shell Command Discipline

- **Filter output at the source.** Use `head`, `tail`, `grep`, `awk` to extract only what matters.
- **Use summary flags.** `git diff --stat`, `ruff check . --statistics`, `pytest --co -q` — these give overviews without flooding context.
- **Never repeat commands.** If you ran `git status` already, don't run it again unless you expect different results (e.g., after a commit).
- **Pipe large outputs to counts.** `grep -c pattern file` instead of `grep pattern file | wc -l` — and both are better than unfiltered `grep`.

## Code Organization for Token Efficiency

- **Keep files under 500 lines.** Large files are expensive to read and edit. Split modules when they grow beyond this threshold.
- **Separate concerns into modules.** Targeted imports and focused reads are cheaper than navigating monolithic files.
- **Use descriptive names.** Self-documenting code reduces the need to read surrounding context to understand a symbol's purpose.

## Large Log Analysis Patterns

- **Never `cat` log files.** Use `grep`, `head`, `tail`, and line-count combinations.
- **Pattern: `grep ERROR logfile | head -20`** — sample errors without loading the full log.
- **Pattern: `grep -c 'pattern' file`** — count occurrences without displaying matches.
- **Pattern: `awk '{print $NF}' file | sort | uniq -c | sort -rn | head`** — distribution summary.

## Codebase Exploration Patterns

1. **Glob** to discover file structure — cheaper than reading directory listings.
2. **Grep** to locate relevant symbols — cheaper than reading every file.
3. **Read with offset/limit** only the files/sections grep identifies as relevant.

This "Glob -> Grep -> Read" pipeline minimizes tokens spent on exploration.

## Exclude Patterns for `.claude/settings.json`

Configure `excludePatterns` to prevent Claude Code from indexing noise directories:

```json
{
  "excludePatterns": [
    "node_modules/**",
    ".git/**",
    "__pycache__/**",
    "*.pyc",
    "dist/**",
    "build/**",
    ".next/**",
    ".cache/**",
    "coverage/**",
    "*.lock",
    "*.min.js",
    "*.min.css"
  ]
}
```

This reduces background context consumption from irrelevant files.

## Cross-Reference

- For **application-level API cost optimization** (token pricing, batching, caching strategies for API calls), see `api-cost-optimization`.
- This skill focuses solely on conserving Claude Code's own context window during interactive sessions.
