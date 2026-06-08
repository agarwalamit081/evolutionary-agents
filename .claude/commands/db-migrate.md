# /db-migrate - Database Migration Workflow

When the user invokes `/db-migrate`, follow this exact workflow:

## Step 1: Analyze ORM Schema Changes
1. Read the current ORM models (SQLAlchemy, Prisma, Sequelize, Django models).
2. Identify what changed: new tables, new columns, modified column types, new constraints.
3. List all changes clearly before proceeding.

## Step 2: Generate Migration File
Generate the appropriate physical migration file:
- **Python (SQLAlchemy + Alembic):** Run `alembic revision --autogenerate -m "description_of_change"`
- **Node (Prisma):** Run `npx prisma migrate dev --name description_of_change`
- **Node (Sequelize):** Use `npx sequelize-cli migration:generate --name description`

## Step 3: Handle pgvector (if applicable)
If the migration involves pgvector or vector columns:
1. Execute this exact command to create the extension:
   ```bash
   PGPASSWORD=$DB_PASSWORD psql -h localhost -U postgres -d $DB_NAME -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>&1
   ```
2. Verify the extension is active:
   ```bash
   PGPASSWORD=$DB_PASSWORD psql -h localhost -U postgres -d $DB_NAME -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"
   ```
3. If the extension creation fails, diagnose and report the error. NEVER proceed without verifying.

## Step 4: Apply Migration
1. Review the generated migration file content.
2. Show the user the Git diff of the migration.
3. Ask for explicit approval before applying.
4. Apply: `alembic upgrade head` (Python) or `npx prisma migrate deploy` (Node).

## Step 5: Verify
1. Run the migration and confirm it applied successfully.
2. Verify the database schema matches the ORM models.
3. Run any related tests to ensure nothing broke.

## Full-Stack Sync
After the database migration:
- Update ALL API response DTOs if schema changed
- Update ALL frontend TypeScript types to match
- Update any mock data or fixtures
