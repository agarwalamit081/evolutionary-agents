# /code-review - Comprehensive Code Review Workflow

When the user invokes `/code-review` or when code is about to be committed, follow this workflow:

## Step 1: Get the Diff
```bash
git diff --stat        # Summary of changed files
git diff              # Full diff of staged changes
```

## Step 2: Run Automated Checks
1. **Linting:** `ruff check .` (Python) or `eslint .` (JS/TS)
2. **Type Checking:** `mypy .` (Python) or `tsc --noEmit` (JS/TS)
3. **Tests:** `uv run python -m pytest` (Python) or `npm test` (JS/TS)
4. **Secret Scan:** `grep -rnE "(sk-[a-zA-Z0-9]{20,}|password\s*=)" src/`
5. **Circular Dependencies:** `npx madge --circular src/` (JS/TS)

## Step 3: Review Each Modified File
For each changed file, verify:
- [ ] No unused imports
- [ ] No undefined function calls
- [ ] All functions are called somewhere
- [ ] Proper error handling in async operations
- [ ] No hardcoded credentials
- [ ] Proper state management (no direct mutation)
- [ ] Proper dependency arrays in useEffect (React)
- [ ] No circular dependencies introduced
- [ ] Proper `__init__.py` updates for new modules
- [ ] No curly braces in file/folder names
- [ ] No raw SQL or unsanitized inputs
- [ ] No `any` or `@ts-ignore` in TypeScript

## Step 4: Full-Stack Sync Check
- [ ] Frontend changes have corresponding backend updates
- [ ] Backend changes have corresponding frontend type updates
- [ ] Database schema changes have migration files
- [ ] TypeScript interfaces match Pydantic/Zod schemas

## Step 5: Report
Organize findings by priority:
- **Critical** (must fix): Security vulnerabilities, broken functionality, data loss risk
- **Warning** (should fix): Code quality issues, missing tests, potential bugs
- **Suggestion** (consider): Performance improvements, readability, refactoring opportunities

Include specific file paths and line numbers for each finding.
Provide concrete fix suggestions with code examples.
