---
description: Backend and Database Patterns Examples
---

**Example 1: Standard PostgreSQL Table with Constraints and Indexes**

```sql
CREATE TABLE user_accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    status VARCHAR(20) DEFAULT 'active'
        CHECK (status IN ('active', 'suspended', 'deleted')),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMPTZ
);

CREATE INDEX idx_user_accounts_email
    ON user_accounts(email);
CREATE INDEX idx_user_accounts_active
    ON user_accounts(status) WHERE deleted_at IS NULL;
```

---

**Example 2: SQLAlchemy 2.0 Model with Mapped Columns**

```python
from datetime import datetime
from uuid import uuid4

from sqlalchemy import String, DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
import uuid


class Base(DeclarativeBase):
    pass


class UserAccount(Base):
    __tablename__ = "user_accounts"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="active")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
```

---

**Example 3: Safe Column Addition Migration (Alembic)**

```python
# revision: 001_add_status_column

def upgrade():
    # Step 1: Add column without default (avoids table rewrite lock)
    op.add_column("users", sa.Column("status", sa.String(20), nullable=True))
    # Step 2: Backfill existing rows (usually a separate data migration)
    # Step 3: Make NOT NULL and set default for future inserts
    op.alter_column("users", "status", nullable=False, server_default="active")


def downgrade():
    op.drop_column("users", "status")
```

---

**Example 4: Concurrent Index Creation**

```sql
-- Up
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_user_id
    ON orders(user_id);

-- Down
DROP INDEX IF EXISTS idx_orders_user_id;
```

---

**Example 5: Safe Insert with ON CONFLICT and RETURNING**

```sql
INSERT INTO user_accounts (email, status)
VALUES ($1, $2)
ON CONFLICT (email)
DO UPDATE SET updated_at = CURRENT_TIMESTAMP
RETURNING id, email, status;
```

---

**Example 6: CTE-Based Reporting Query**

```sql
WITH active_users AS (
    SELECT id, email
    FROM users
    WHERE status = 'active' AND deleted_at IS NULL
),
recent_orders AS (
    SELECT user_id, COUNT(*) AS order_count
    FROM orders
    WHERE created_at > NOW() - INTERVAL '30 days'
    GROUP BY user_id
)
SELECT u.email, COALESCE(o.order_count, 0) AS orders
FROM active_users u
LEFT JOIN recent_orders o ON u.id = o.user_id
ORDER BY orders DESC;
```

---

**Example 7: Keyset Pagination**

```python
def get_users_paginated(cursor: UUID | None = None, limit: int = 50):
    query = select(User).order_by(User.id)
    if cursor:
        query = query.where(User.id > cursor)
    return session.scalars(query.limit(limit)).all()
```

---

**Example 8: Soft Delete with Filtered Index**

```sql
-- Soft delete
UPDATE user_accounts
SET deleted_at = CURRENT_TIMESTAMP, status = 'deleted'
WHERE id = $1;

-- Filtered index ensures queries only see active rows
CREATE INDEX idx_user_accounts_active
    ON user_accounts(email, status)
    WHERE deleted_at IS NULL;

-- Query automatically benefits from the partial index
SELECT id, email FROM user_accounts
WHERE email = $1 AND deleted_at IS NULL;
```
