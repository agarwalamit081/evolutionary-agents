# Context Window Management — Examples

## Example 1: Targeted ripgrep search instead of reading entire files

```bash
# BAD: Reading entire files to find a function
# Read file1.ts (500 lines), Read file2.ts (300 lines), Read file3.ts (400 lines)
# Token cost: ~3000 lines

# GOOD: Use ripgrep to locate, then read only the match
rg "function handleAuth" --line-number src/
# Output: src/auth/handler.ts:42:function handleAuth(req: Request) {
# Then: Read src/auth/handler.ts with offset=35, limit=30
# Token cost: ~30 lines
```

## Example 2: Reading file sections with line offsets

```bash
# BAD: Reading an entire 1500-line config file
Read config/defaults.yaml  # 1500 lines → ~37k tokens

# GOOD: Read only the section you need
Read config/defaults.yaml offset=200 limit=50  # 50 lines → ~1.2k tokens

# Use grep to find the line number first:
rg "database:" --line-number config/defaults.yaml
# Output: config/defaults.yaml:187:database:
# Then read offset=185, limit=40
```

## Example 3: Listing tests without running them

```bash
# BAD: Running full test suite to see what's available
pytest tests/ -v  # Runs all tests, floods output

# GOOD: Collect only — list test names without execution
pytest tests/ --co -q
# Output:
# tests/test_auth.py::test_login
# tests/test_auth.py::test_logout
# tests/test_users.py::test_create_user
# 3 tests collected
```

## Example 4: Summary-only lint output

```bash
# BAD: Full lint output for every violation
ruff check src/  # Could be hundreds of lines

# GOOD: Statistics summary only
ruff check src/ --statistics
# Output:
# 12    unused-import
#  8    line-too-long
#  3    missing-docstring
# 23 errors total
```

## Example 5: Change overview without full diff

```bash
# BAD: Full diff of all changes (could be thousands of lines)
git diff  # Every added/removed line in context

# GOOD: Stat overview first, then target specific files
git diff --stat
# Output:
# src/auth.py      |  12 ++++++------
# src/models.py    |   3 ++-
# tests/test_auth.py |  28 ++++++++++++++++++++++++++++
# 3 files changed, 36 insertions(+), 7 deletions(-)

# Then read only the files that matter: git diff src/auth.py
```

## Example 6: Exclude patterns in settings.json

```json
// .claude/settings.json
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
    "*.min.css",
    "logs/**"
  ]
}
```

This prevents Claude Code from spending context budget indexing directories that are never relevant to code changes.
