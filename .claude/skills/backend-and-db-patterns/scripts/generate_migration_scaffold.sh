#!/bin/bash
# Generate a timestamped migration file scaffold with safety headers.
# Usage: ./generate_migration_scaffold.sh <description>
set -euo pipefail

DESCRIPTION="${1:?Usage: $0 <description>}"
TIMESTAMP=$(date +"%Y%m%d%H%M%S")
SAFE_NAME=$(echo "$DESCRIPTION" | tr ' ' '_' | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_]/_/g')
FILENAME="${TIMESTAMP}_${SAFE_NAME}.sql"

cat > "$FILENAME" <<EOF
-- Migration: ${SAFE_NAME}
-- Created: $(date +"%Y-%m-%d %H:%M:%S")
-- Reversible: YES
-- Blocking: NO (verify with EXPLAIN ANALYZE)

-- UP
BEGIN;

-- Add your migration here.
-- Remember:
--   - Use IF NOT EXISTS for idempotency
--   - Use CONCURRENTLY for index creation
--   - Avoid NOT NULL additions without a default on large tables

COMMIT;

-- DOWN
-- BEGIN;
-- Reverse the migration here.
-- COMMIT;
EOF

echo "Generated: ${FILENAME}"
