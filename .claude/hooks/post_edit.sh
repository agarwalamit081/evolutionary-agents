#!/bin/bash
# post_edit.sh - Runs automated validation after file edits
# Catches syntax errors, secret leaks, and code quality issues

# Read JSON input from stdin
INPUT=$(cat)

# Extract the file path from tool_input
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // .tool_input.filepath // empty' 2>/dev/null)

# Also try to get from tool_result if available
if [ -z "$FILE_PATH" ]; then
  FILE_PATH=$(echo "$INPUT" | jq -r '.file_path // .filepath // empty' 2>/dev/null)
fi

if [ -z "$FILE_PATH" ] || [ ! -f "$FILE_PATH" ]; then
  exit 0
fi

# 0. Block .env file edits (allow documentation variants)
if echo "$FILE_PATH" | grep -iE '(\.env$|\.env\.)' > /dev/null; then
  if ! echo "$FILE_PATH" | grep -iE '\.env\.(example|template|sample|gitignore)$' > /dev/null; then
    echo "BLOCKED: Editing .env files is not allowed. Manage environment variables manually." >&2
    exit 2
  fi
fi

# 1. Secret Detection - Block any write containing hardcoded secrets
if grep -qE "(sk-[a-zA-Z0-9]{20,}|sk-ant-[a-zA-Z0-9]{20,}|password\s*=\s*['\"][^'\"]+['\"]|AWS_ACCESS_KEY|AWS_SECRET_ACCESS_KEY|DATABASE_URL\s*=\s*['\"]postgres(?!ql://localhost))" "$FILE_PATH" 2>/dev/null; then
  echo "ERROR: Potential hardcoded secret detected in $FILE_PATH. Remove secrets and use environment variables." >&2
  exit 1
fi

# 2. Python Linting
if [[ "$FILE_PATH" == *.py ]]; then
  # Check for stdlib logging usage (should use loguru)
  if grep -qE "import logging|from logging import" "$FILE_PATH" 2>/dev/null; then
    echo "WARNING: Standard 'logging' module detected. Use 'loguru' instead." >&2
  fi

  # Check for raw SQL (basic detection)
  if grep -qE 'execute\s*\(\s*f["\x27]|execute\s*\(\s*".*\%|execute\s*\(\s*".*\+|execute\s*\(\s*".*\{.*\}' "$FILE_PATH" 2>/dev/null; then
    echo "WARNING: Potential raw SQL with string interpolation detected. Use parameterized queries." >&2
  fi

  # Run ruff if available
  if command -v ruff &> /dev/null; then
    ruff check "$FILE_PATH" 2>&1 || true
  fi

  # Check for FastAPI routes without auth
  if grep -qE '@(app|router)\.(get|post|put|delete|patch)\(' "$FILE_PATH" 2>/dev/null; then
    if ! grep -qE 'Depends\(' "$FILE_PATH" 2>/dev/null; then
      echo "WARNING: FastAPI route detected without Depends() for authentication. Ensure routes are properly protected." >&2
    fi
  fi

  # Check for asyncpg raw SQL with f-strings
  if grep -qE '(conn|pool)\.(execute|fetch|fetchrow|fetchval)\s*\(\s*f["\x27]' "$FILE_PATH" 2>/dev/null; then
    echo "WARNING: asyncpg raw query with f-string detected. Use parameterized queries to prevent SQL injection." >&2
  fi

  # Check for missing type annotations on function parameters
  if grep -Pn 'def \w+\([a-zA-Z_]+[,\)]' "$FILE_PATH" 2>/dev/null | grep -v '#' > /dev/null; then
    echo "WARNING: Function parameter missing type annotation. All parameters must have type hints (e.g., 'def foo(x: int)' not 'def foo(x)')." >&2
  fi

  # Check for missing return type annotations
  if grep -Pn 'def \w+\(.*\):\s*$' "$FILE_PATH" 2>/dev/null | grep -v -e '->' > /dev/null; then
    echo "WARNING: Function missing return type annotation. All functions must declare return types (e.g., '-> None')." >&2
  fi

  # Check __init__.py staleness for new modules
  DIR_PATH=$(dirname "$FILE_PATH")
  FILE_BASE=$(basename "$FILE_PATH" .py)
  if [ -f "$DIR_PATH/__init__.py" ] && [ "$FILE_BASE" != "__init__" ] && [ "$FILE_BASE" != "__future__" ]; then
    if grep -qE '(^def |^class |^[A-Z_]+\s*=\s*)' "$FILE_PATH" 2>/dev/null; then
      if ! grep -qE "from\s+\.$FILE_BASE\s+import|from\s+\.$FILE_BASE\b" "$DIR_PATH/__init__.py" 2>/dev/null; then
        echo "WARNING: Module '$FILE_BASE.py' has public symbols but is not exported from '$DIR_PATH/__init__.py'. Update __init__.py to include: from .$FILE_BASE import <symbols>" >&2
      fi
    fi
  fi
fi

# 3. TypeScript/JavaScript Linting
if [[ "$FILE_PATH" == *.ts || "$FILE_PATH" == *.tsx || "$FILE_PATH" == *.js || "$FILE_PATH" == *.jsx ]]; then
  # Check for 'any' type usage
  if grep -qE ":\s*any\b|<any>|@ts-ignore|@ts-expect-error" "$FILE_PATH" 2>/dev/null; then
    echo "WARNING: TypeScript 'any' type, @ts-ignore, or @ts-expect-error detected. Fix the root cause." >&2
  fi

  # Check for dangerouslySetInnerHTML without sanitization
  if grep -q "dangerouslySetInnerHTML" "$FILE_PATH" 2>/dev/null; then
    if ! grep -q "DOMPurify\|sanitize\|sanitizeHTML" "$FILE_PATH" 2>/dev/null; then
      echo "WARNING: dangerouslySetInnerHTML without sanitization detected." >&2
    fi
  fi

  # Run eslint if available
  if command -v eslint &> /dev/null; then
    eslint "$FILE_PATH" 2>&1 || true
  fi
fi

exit 0
