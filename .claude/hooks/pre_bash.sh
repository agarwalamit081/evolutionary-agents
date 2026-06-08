#!/bin/bash
# pre_bash.sh - Validates Bash commands before execution
# Prevents dangerous operations and enforces safe patterns

# Read JSON input from stdin
INPUT=$(cat)

# Extract the command field from tool_input using jq
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

if [ -z "$COMMAND" ]; then
  exit 0
fi

# ============================================================
# Phase 1: BLOCKED checks (exit 2 on match)
# These run FIRST to ensure dangerous ops are always caught,
# even if a warning pattern also matches the command.
# ============================================================

# 1. Block dangerous write operations on databases
if echo "$COMMAND" | grep -iE '\b(DROP\s+DATABASE|DROP\s+TABLE|TRUNCATE\s+TABLE|DROP\s+SCHEMA)\b' > /dev/null; then
  echo "BLOCKED: Destructive database operation detected. Use the /db-migrate command instead." >&2
  exit 2
fi

# 2. Block clearing git history force-pushes
if echo "$COMMAND" | grep -iE '(git\s+push\s+.*--force|git\s+reset\s+--hard\s+HEAD~)' > /dev/null; then
  echo "BLOCKED: Destructive git operation. This could cause irreversible data loss." >&2
  exit 2
fi

# 3. Block `rm -rf /` or recursive root deletion
if echo "$COMMAND" | grep -E 'rm\s+-rf\s+/' > /dev/null; then
  echo "BLOCKED: Dangerous recursive deletion from root." >&2
  exit 2
fi

# 4. Block `chmod 777` on any directory
if echo "$COMMAND" | grep -E 'chmod\s+777' > /dev/null; then
  echo "BLOCKED: Overly permissive file permissions (777). Use minimum required permissions." >&2
  exit 2
fi

# 5. Block .env file read/write
if echo "$COMMAND" | grep -iE '(cat|head|tail|less|more|vim|nano|grep|awk|sed|source|\.|\s)\s+.*\.env($|\s)' > /dev/null; then
  echo "BLOCKED: Reading .env files is not allowed. Use environment variables via the application's config system." >&2
  exit 2
fi
if echo "$COMMAND" | grep -iE '(echo|cat|printf|tee|cp|mv)\s+.*[>].*\.env($|\s)' > /dev/null; then
  echo "BLOCKED: Writing to .env files is not allowed. Manage environment variables manually." >&2
  exit 2
fi

# 6. Block pip install / uv pip install -- require user permission
if echo "$COMMAND" | grep -iE '\b(pip\s+install|uv\s+pip\s+install|pip-sync)\b' > /dev/null; then
  echo "BLOCKED: Package installation modifies the environment. Run the install command yourself, or confirm you want Claude to proceed. Use: uv pip install <package>" >&2
  exit 2
fi

# ============================================================
# Phase 2: WARNING checks (exit 0 on match)
# These run after all BLOCKED checks, so dangerous ops are
# never short-circuited by a warning.
# ============================================================

# 7. Warn about reading large files (potential token waste)
if echo "$COMMAND" | grep -iE '(cat|less|more|head\s+-[a-z]*\s+[0-9]+)\s+\S*\.(log|sql|dump|min\.js|min\.css)' > /dev/null; then
  echo "WARNING: Reading large log/dump/minified files wastes tokens. Use grep/ripgrep for targeted searches." >&2
  exit 0
fi

# 8. Warn about running same command pattern (potential loop)
if echo "$COMMAND" | grep -iE '(while\s+true|while\s+:|for\s+\(\(\s*;)\s*' > /dev/null; then
  echo "WARNING: Potential infinite loop detected. Ensure there is a proper exit condition." >&2
  exit 0
fi

# 9. Warn on sudo usage
if echo "$COMMAND" | grep -iE '\bsudo\b' > /dev/null; then
  echo "WARNING: sudo detected. This modifies system state outside the project directory." >&2
  exit 0
fi

# 10. Warn on alembic downgrade without specific revision
if echo "$COMMAND" | grep -iE 'alembic\s+downgrade\s+(base|head|-1)' > /dev/null; then
  echo "WARNING: Alembic downgrade without specific revision. Specify the target revision to avoid accidental data loss." >&2
  exit 0
fi

# 11. Warn on bare python commands -- should use uv run python
if echo "$COMMAND" | grep -P '(?<!\w)(python|python3)\s+' > /dev/null; then
  if ! echo "$COMMAND" | grep -E 'uv\s+run' > /dev/null; then
    echo "WARNING: Bare 'python' detected. Use 'uv run python' to ensure the correct virtual environment." >&2
    exit 0
  fi
fi

exit 0
