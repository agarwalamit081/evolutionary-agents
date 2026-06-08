# Security Audit — Examples

Concrete code patterns for common security scenarios. Use these as reference when writing or reviewing code.

---

## Example 1: Detecting Hardcoded Secrets with Regex

```python
import re

SECRET_PATTERNS = [
    (r'sk-[a-zA-Z0-9]{20,}', "OpenAI API key"),
    (r'AKIA[0-9A-Z]{16}', "AWS Access Key ID"),
    (r'ghp_[a-zA-Z0-9]{36}', "GitHub Personal Access Token"),
    (r'xoxb-[0-9]{10,}-[0-9]{10,}-[a-zA-Z0-9]{24}', "Slack Bot Token"),
    (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key assignment"),
]

def scan_for_secrets(content: str, filepath: str) -> list[dict]:
    findings = []
    for pattern, description in SECRET_PATTERNS:
        for match in re.finditer(pattern, content):
            findings.append({
                "file": filepath,
                "line": content[:match.start()].count("\n") + 1,
                "type": description,
                "matched": match.group()[:20] + "...",  # truncate to avoid logging the full secret
            })
    return findings
```

---

## Example 2: Parameterized Query vs Raw SQL (SQLAlchemy)

**VULNERABLE — Raw SQL with string concatenation:**

```python
# BAD: SQL injection possible
user_input = request.args.get("name")
query = f"SELECT * FROM users WHERE name = '{user_input}'"
results = db.session.execute(text(query))
```

**SECURE — Parameterized query:**

```python
# GOOD: Parameterized query
user_input = request.args.get("name")
results = db.session.execute(
    text("SELECT * FROM users WHERE name = :name"),
    {"name": user_input},
)
```

**SECURE — ORM model:**

```python
# GOOD: ORM with filters
results = User.query.filter(User.name == user_input).all()
```

---

## Example 3: FastAPI Input Validation with Pydantic

```python
from pydantic import BaseModel, Field, EmailStr, field_validator
import re

class UserCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    age: int = Field(..., ge=13, le=120)

    model_config = {"extra": "forbid"}  # Reject unexpected fields

    @field_validator("name")
    @classmethod
    def name_must_be_safe(cls, v: str) -> str:
        if re.search(r"[<>\"'&]", v):
            raise ValueError("Name contains disallowed characters")
        return v.strip()
```

---

## Example 4: CORS Configuration for FastAPI

**VULNERABLE — Wildcard CORS:**

```python
# BAD: Allows any origin with credentials
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,  # Critical: credentials + wildcard = disaster
)
```

**SECURE — Explicit origins:**

```python
# GOOD: Explicit allowed origins
ALLOWED_ORIGINS = [
    "https://app.example.com",
    "https://admin.example.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)
```

---

## Example 5: JWT Token Validation Middleware in FastAPI

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timezone

security = HTTPBearer()
JWT_SECRET = os.environ["JWT_SECRET"]
JWT_ALGORITHM = "HS256"

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    # Verify required claims
    if "sub" not in payload or "exp" not in payload:
        raise HTTPException(status_code=401, detail="Malformed token")

    return payload
```

---

## Example 6: PII Redaction in Loguru Logs

```python
from loguru import logger
import re

def redact_pii(message: str) -> str:
    patterns = {
        r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b': "[SSN-REDACTED]",
        r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b': "[CARD-REDACTED]",
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b': "[EMAIL-REDACTED]",
        r'"password"\s*:\s*"[^"]*"': '"password": "[REDACTED]"',
        r'"token"\s*:\s*"[^"]*"': '"token": "[REDACTED]"',
        r'"api_key"\s*:\s*"[^"]*"': '"api_key": "[REDACTED]"',
    }
    for pattern, replacement in patterns.items():
        message = re.sub(pattern, replacement, message, flags=re.IGNORECASE)
    return message

# Install as a loguru sink filter
logger.add("app.log", format=redact_pii("{message}"))
```

---

## Example 7: Dependency Audit with pip-audit

```bash
# Install pip-audit
pip install pip-audit

# Scan current environment for known CVEs
pip-audit

# Scan from a requirements file
pip-audit -r requirements.txt

# Output in JSON for CI integration
pip-audit --format json > audit-report.json

# Check specific package
pip-audit --desc --ignore-vuln PYSEC-2023-123  # suppress known false positive
```
