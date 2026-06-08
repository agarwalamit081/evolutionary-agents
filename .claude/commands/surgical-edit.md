# /surgical-edit - Surgical File Edit Workflow

When the user invokes `/surgical-edit <file> <description>`, follow this exact workflow:

## Step 1: Read the Target File
1. Read ONLY the relevant portion of the target file.
2. If the file is large (> 200 lines), identify the specific section using `grep`/`ripgrep` first.
3. NEVER read the entire file into context if only a small portion needs changing.

## Step 2: Identify Exact Changes
1. Identify the EXACT lines or blocks that need to change.
2. Determine the minimal set of modifications needed.
3. If the change affects function signatures, find ALL callers across the codebase before proceeding.

## Step 3: Apply Targeted Edit
1. Use the `Edit` or search-and-replace tool to modify ONLY the identified lines.
2. NEVER output the entire file unless it is under 50 lines total.
3. If multiple changes are needed in the same file, apply them as separate surgical edits.

## Step 4: Validate the Edit
1. Run the linter: `ruff check <file>` or `eslint <file>`
2. Run the type checker if applicable: `mypy <file>` or `tsc --noEmit`
3. Run related tests: `pytest` or `npm test`
4. Review `git diff` to confirm only the intended lines were changed.

## Step 5: Check for Side Effects
1. If function signatures changed: verify all callers were updated.
2. If types changed: verify all dependent files were updated.
3. If imports changed: verify `__init__.py` or barrel exports were updated.
4. Check for circular dependencies that may have been introduced.

## Error Recovery
If the edit causes unexpected errors:
1. Use `git checkout -- <file>` to restore the original.
2. Re-analyze the problem with fresh context.
3. Try a different, more targeted approach.

## Critical Rules
- NEVER replace a 500+ line file to change 5 lines.
- NEVER create stubs or placeholders. Write the full implementation.
- NEVER forget to review the git diff before declaring done.
- If business logic is ambiguous, ASK the user rather than guessing.
