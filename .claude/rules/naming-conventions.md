# Folder and File Naming Conventions

## Strict Naming Rules
- NEVER use curly braces `{}` in folder or file names. `{scripts}` is WRONG, `scripts` is correct.
- ALL folder names must be lowercase, using hyphens or underscores as word separators.
- File names should be descriptive and follow the language convention (snake_case for Python, camelCase or kebab-case for JS/TS).
- NEVER use spaces or special characters in file or folder names.

## Standard Project Directory Structure
```
project-root/
  scripts/          # Utility and automation scripts
  src/              # Application source code
  tests/            # Test suites
  logs/             # Application log output (NEVER /tmp/)
  config/           # Configuration files
  docs/             # Documentation and ADRs
  .claude/          # Claude Code configuration
```

## Python-Specific Naming
- Modules and packages: `snake_case` (e.g., `data_processor.py`, `auth_utils/`)
- Classes: `PascalCase` (e.g., `UserDataProcessor`, `AuthUtils`)
- Functions and variables: `snake_case` (e.g., `process_user_data`, `auth_token`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_RETRY_COUNT`, `DEFAULT_TIMEOUT`)
- Test files: `test_<module_name>.py` (e.g., `test_data_processor.py`)

## JavaScript/TypeScript-Specific Naming
- Components: `PascalCase` (e.g., `UserProfile.tsx`, `DataTable.tsx`)
- Utility files: `camelCase` or `kebab-case` (e.g., `formatDate.ts`, `date-utils.ts`)
- Test files: `*.test.ts`, `*.test.tsx`, `*.spec.ts`
- CSS modules: `*.module.css` or `*.module.scss`

## Configuration File Standards
- Environment variables: `UPPER_SNAKE_CASE` in `.env` (e.g., `DATABASE_URL`, `API_SECRET_KEY`)
- JSON/YAML config keys: `camelCase` for JS projects, `snake_case` for Python projects
- NEVER store secrets in configuration files committed to version control
