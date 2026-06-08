---
description: Code Quality and Patterns Examples
---

**Example 1: Fixing SRP Violation**

*Before:*
```python
class ReportGenerator:
    def generate(self, data):
        conn = psycopg2.connect("...")          # DB concern
        formatted = f"Report: {data}"            # Formatting concern
        smtplib.SMTP(...).sendmail(...)           # Notification concern
```

*After:*
```python
class ReportFormatter:
    def format(self, data: dict) -> str:
        return f"Report: {data}"

class ReportService:
    def __init__(self, fetcher: DataFetcher, notifier: Notifier, formatter: ReportFormatter):
        self.fetcher = fetcher
        self.notifier = notifier
        self.formatter = formatter

    def generate_and_send(self):
        data = self.fetcher.fetch()
        report = self.formatter.format(data)
        self.notifier.send(report)
```

---

**Example 2: DIP Violation → Dependency Injection**

*Before:*
```python
class OrderService:
    def __init__(self):
        self.db = MySQLConnection()  # Tight coupling
```

*After:*
```python
class OrderService:
    def __init__(self, db: DatabaseInterface):  # Depends on abstraction
        self.db = db
```

---

**Example 3: Strategy Pattern Replacing if/else**

*Before:*
```python
class PaymentProcessor:
    def process(self, method, amount):
        if method == "credit_card":
            pass  # 50 lines
        elif method == "paypal":
            pass  # 50 lines
```

*After:*
```python
class PaymentStrategy(Protocol):
    def pay(self, amount: float) -> None: ...

class CreditCardPayment:
    def pay(self, amount: float): ...

class PayPalPayment:
    def pay(self, amount: float): ...

class PaymentProcessor:
    def __init__(self, strategy: PaymentStrategy):
        self.strategy = strategy

    def process(self, amount: float):
        self.strategy.pay(amount)
```

---

**Example 4: Decorator Pattern (Adding Caching)**

```python
import functools
import time

def cache_result(ttl_seconds=300):
    def decorator(func):
        cache = {}
        @functools.wraps(func)
        def wrapper(*args):
            if args in cache:
                result, timestamp = cache[args]
                if time.time() - timestamp < ttl_seconds:
                    return result
            result = func(*args)
            cache[args] = (result, time.time())
            return result
        return wrapper
    return decorator

@cache_result(ttl_seconds=60)
def fetch_user(user_id: str):
    return db.query(User).get(user_id)
```

---

**Example 5: Factory Method Pattern**

```python
from abc import ABC, abstractmethod

class Notification(ABC):
    @abstractmethod
    def send(self, message: str): ...

class EmailNotification(Notification):
    def send(self, message: str):
        print(f"Sending email: {message}")

class SMSNotification(Notification):
    def send(self, message: str):
        print(f"Sending SMS: {message}")

class NotificationFactory:
    @staticmethod
    def create(channel: str) -> Notification:
        match channel:
            case "email": return EmailNotification()
            case "sms": return SMSNotification()
            case _: raise ValueError(f"Unsupported channel: {channel}")
```

---

**Example 6: Observer Pattern**

```python
class EventBus:
    def __init__(self):
        self._subscribers = {}

    def subscribe(self, event_type: str, handler):
        self._subscribers.setdefault(event_type, []).append(handler)

    def publish(self, event_type: str, data):
        for handler in self._subscribers.get(event_type, []):
            handler(data)

# Usage
bus = EventBus()
bus.subscribe("order.created", lambda d: print(f"Email: New order {d['order_id']}"))
bus.subscribe("order.created", lambda d: print(f"Analytics: Order value ${d['total']}"))
bus.publish("order.created", {"order_id": "123", "total": 49.99})
```

---

**Example 7: Builder Pattern**

```python
from dataclasses import dataclass, field

@dataclass
class QueryBuilder:
    table: str = ""
    columns: list[str] = field(default_factory=lambda: ["*"])
    where_clauses: list[str] = field(default_factory=list)
    limit_value: int | None = None

    def select(self, *cols) -> "QueryBuilder":
        self.columns = list(cols)
        return self

    def from_table(self, name: str) -> "QueryBuilder":
        self.table = name
        return self

    def where(self, condition: str) -> "QueryBuilder":
        self.where_clauses.append(condition)
        return self

    def limit(self, n: int) -> "QueryBuilder":
        self.limit_value = n
        return self

    def build(self) -> str:
        sql = f"SELECT {', '.join(self.columns)} FROM {self.table}"
        if self.where_clauses:
            sql += f" WHERE {' AND '.join(self.where_clauses)}"
        if self.limit_value:
            sql += f" LIMIT {self.limit_value}"
        return sql

# Usage
query = (QueryBuilder()
    .select("id", "name", "email")
    .from_table("users")
    .where("status = 'active'")
    .where("created_at > '2024-01-01'")
    .limit(100)
    .build())
```

---

**Example 8: Adapter Pattern for Third-Party API**

```python
class PaymentGateway(Protocol):
    def charge(self, amount: float, currency: str) -> str: ...

class StripeAdapter:
    """Adapts Stripe SDK to our PaymentGateway interface."""
    def __init__(self, api_key: str):
        self.client = stripe.Client(api_key=api_key)

    def charge(self, amount: float, currency: str) -> str:
        response = self.client.charges.create(amount=int(amount * 100), currency=currency)
        return response["id"]

class PaymentService:
    def __init__(self, gateway: PaymentGateway):
        self.gateway = gateway

    def process_payment(self, amount: float, currency: str = "usd") -> str:
        return self.gateway.charge(amount, currency)
```
