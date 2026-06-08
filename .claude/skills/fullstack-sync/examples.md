# Fullstack Sync — Examples

## Example 1: Adding a New API Field (End-to-End)

A product endpoint needs a new `discount_price` field.

```python
# Backend: SQLAlchemy model
class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    price = Column(Float, nullable=False)
    discount_price = Column(Float, nullable=True)  # new column

# Backend: Pydantic DTO
class ProductResponse(BaseModel):
    id: int
    name: str
    price: float
    discount_price: float | None = None  # new field

# Backend: FastAPI route (no change needed if response_model is ProductResponse)
@router.get("/products/{product_id}", response_model=ProductResponse)
async def get_product(product_id: int, db: Session = Depends(get_db)):
    ...
```

```typescript
// Frontend: TypeScript interface
interface Product {
  id: number;
  name: string;
  price: number;
  discount_price: number | null;  // new field
}

// Frontend: component update
function ProductCard({ product }: { product: Product }) {
  return (
    <div>
      <h3>{product.name}</h3>
      {product.discount_price !== null && (
        <span className="text-green-600">${product.discount_price}</span>
      )}
    </div>
  );
}
```

## Example 2: Renaming a Database Column

Rename `fname` to `first_name` in the `users` table.

```python
# Alembic migration
def upgrade() -> None:
    op.alter_column("users", "fname", new_column_name="first_name")

def downgrade() -> None:
    op.alter_column("users", "first_name", new_column_name="fname")

# SQLAlchemy model update
class User(Base):
    __tablename__ = "users"
    first_name = Column(String, nullable=False)  # renamed from fname

# Pydantic DTO update
class UserResponse(BaseModel):
    first_name: str  # renamed from fname

# FastAPI route — no handler change, response_model handles it
```

```typescript
// TypeScript interface update
interface User {
  first_name: string;  // renamed from fname
}

// Update all frontend references: user.fname -> user.first_name
```

## Example 3: Adding a New Enum Value

Add `ARCHIVED` to a `Status` enum used across the stack.

```python
# Python: enums.py
class Status(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"  # new value
```

```typescript
// TypeScript: enums.ts
type Status = "active" | "inactive" | "archived";  // added "archived"

// Or with a const enum:
const enum StatusEnum {
  ACTIVE = "active",
  INACTIVE = "inactive",
  ARCHIVED = "archived",  // new value
}
```

Update any UI filter dropdowns or status badges that render the enum values.

## Example 4: Validating Pydantic Model Matches TypeScript Interface

Using a test to assert field-level consistency.

```python
# Test: test_sync.py
from pydantic import BaseModel
from backend.models import UserResponse

def test_user_response_fields_match_typescript():
    expected_fields = {"id", "email", "first_name", "last_name", "is_active"}
    actual_fields = set(UserResponse.model_fields.keys())
    missing = expected_fields - actual_fields
    extra = actual_fields - expected_fields
    assert not missing, f"Fields missing from Pydantic model: {missing}"
    assert not extra, f"Extra fields in Pydantic model: {extra}"
```

```typescript
// Reference TypeScript interface
interface User {
  id: number;
  email: string;
  first_name: string;
  last_name: string;
  is_active: boolean;
}
```

## Example 5: Cross-Referencing API Endpoints

Map frontend fetch calls to backend routes.

```python
# Backend: routes/users.py
@router.get("/api/users", response_model=list[UserResponse])
async def list_users(db: Session = Depends(get_db)): ...

@router.post("/api/users", response_model=UserResponse, status_code=201)
async def create_user(data: UserCreate, db: Session = Depends(get_db)): ...

@router.delete("/api/users/{user_id}", status_code=204)
async def delete_user(user_id: int, db: Session = Depends(get_db)): ...
```

```typescript
// Frontend: api/users.ts
export async function fetchUsers(): Promise<User[]> {
  const res = await fetch("/api/users");       // matches GET /api/users
  return res.json();
}

export async function createUser(data: UserCreate): Promise<User> {
  const res = await fetch("/api/users", {      // matches POST /api/users
    method: "POST",
    body: JSON.stringify(data),
  });
  return res.json();
}

// NOTE: No frontend call for DELETE /api/users/{id} — orphaned route or missing client?
```

## Example 6: Generating TypeScript Types from OpenAPI Spec

Use FastAPI's auto-generated OpenAPI schema to produce TypeScript interfaces.

```bash
# Export the OpenAPI spec
python -c "
from backend.main import app
import json
with open('openapi.json', 'w') as f:
    json.dump(app.openapi(), f, indent=2)
"

# Generate TypeScript types (using openapi-typescript)
npx openapi-typescript openapi.json --output src/types/api.d.ts
```

```typescript
// Generated file: src/types/api.d.ts
export interface operations {
  list_users: {
    responses: {
      200: {
        content: {
          "application/json": components["schemas"]["UserResponse"][];
        };
      };
    };
  };
}
```

Add a CI step that re-generates types and fails if the output changes (detecting drift).
