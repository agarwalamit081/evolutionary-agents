#!/bin/bash
# db-setup.sh - Safely initializes pgvector extension on the local PostgreSQL database
# Usage: Run via /db-setup command or directly: bash scripts/db-setup.sh

set -euo pipefail

echo "=== pgvector Extension Setup ==="
echo ""

# Configuration with environment variable fallbacks
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_USER="${DB_USER:-postgres}"
DB_PASSWORD="${DB_PASSWORD:-postgres}"
DB_NAME="${DB_NAME:-development}"

echo "Database Configuration:"
echo "  Host: $DB_HOST:$DB_PORT"
echo "  User: $DB_USER"
echo "  Database: $DB_NAME"
echo ""

# Check if psql is available
if ! command -v psql &> /dev/null; then
  echo "ERROR: 'psql' command not found. Please install PostgreSQL client tools." >&2
  exit 1
fi

# Step 1: Check if the database exists
echo "Step 1: Checking database connectivity..."
if ! PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" > /dev/null 2>&1; then
  echo "ERROR: Cannot connect to database '$DB_NAME'. Please verify the connection parameters." >&2
  exit 1
fi
echo "  Connection successful."

# Step 2: Create pgvector extension
echo ""
echo "Step 2: Creating pgvector extension..."
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>&1

# Step 3: Verify the extension
echo ""
echo "Step 3: Verifying extension state..."
EXTENSION_INFO=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -A -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';" 2>&1)

if echo "$EXTENSION_INFO" | grep -q "vector"; then
  echo "  Extension 'vector' is installed."
  echo "  Version: $(echo "$EXTENSION_INFO" | cut -f2)"
  echo ""
  echo "=== pgvector setup complete ==="
else
  echo "ERROR: pgvector extension verification failed." >&2
  echo "  Extension does not appear to be installed." >&2
  exit 1
fi
