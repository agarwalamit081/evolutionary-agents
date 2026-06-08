#!/bin/bash
# validate-readonly-query.sh - Blocks SQL write operations, allows SELECT queries only
# Used by subagents with read-only database access

# Read JSON input from stdin
INPUT=$(cat)

# Extract the command field from tool_input using jq
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

if [ -z "$COMMAND" ]; then
  exit 0
fi

# Block write operations (case-insensitive)
if echo "$COMMAND" | grep -iE '\b(INSERT\s+INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM|DROP\s+TABLE|DROP\s+DATABASE|CREATE\s+TABLE|ALTER\s+TABLE|TRUNCATE\s+TABLE|CREATE\s+INDEX|REPLACE\s+INTO|MERGE\s+INTO|GRANT\s+|REVOKE\s+|CREATE\s+EXTENSION|ALTER\s+EXTENSION|COPY\s+|VACUUM|COMMENT\s+ON)\b' > /dev/null; then
  echo "BLOCKED: Write/DDL operations not allowed. Use SELECT queries only." >&2
  exit 2
fi

# Allow common read-only operations
if echo "$COMMAND" | grep -iE '\b(SELECT|EXPLAIN|SHOW|DESCRIBE|WITH\s+\w+\s+AS)\b' > /dev/null; then
  exit 0
fi

# If command doesn't match any SQL patterns, allow it (might be a utility command)
exit 0
