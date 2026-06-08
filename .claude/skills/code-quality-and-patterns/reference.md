---
description: Code Quality and Patterns Reference
---

## SOLID Principles

### Single Responsibility Principle (SRP)
- **Definition**: A module should be responsible to one, and only one, actor.
- **Smell**: A class named `UserManager` that handles DB queries, email sending, and password hashing.
- **Fix**: Split into `UserRepository`, `EmailService`, and `PasswordHasher`.

### Open/Closed Principle (OCP)
- **Definition**: Extend behavior without modifying existing source code.
- **Smell**: Adding a new `elif` to a core processing function for every new feature.
- **Fix**: Use polymorphism (Strategy pattern) or plugin architectures.

### Liskov Substitution Principle (LSP)
- **Definition**: Derived classes must be substitutable for their base classes.
- **Smell**: A `Square` inheriting from `Rectangle` but throwing on independent width/height.
- **Fix**: Favor composition over inheritance; preserve pre/post conditions.

### Interface Segregation Principle (ISP)
- **Definition**: Clients should not depend on methods they do not use.
- **Smell**: A massive `IWorker` with `work()`, `eat()`, `sleep()` — forcing `Robot` to implement `eat()`.
- **Fix**: Split into `IWorkable` and `IFeedable`.

### Dependency Inversion Principle (DIP)
- **Definition**: High-level modules should not depend on low-level modules. Both depend on abstractions.
- **Smell**: `class OrderService: def __init__(self): self.db = MySQLConnection()`
- **Fix**: `class OrderService: def __init__(self, db: DatabaseInterface): ...`

## Design Patterns

### Creational (Object Creation)
- **Factory Method**: Delegate creation to subclasses. Use when type determined at runtime.
- **Builder**: Construct complex objects step-by-step. Ideal for many optional parameters.
- **Singleton**: Ensure one instance. *Caution*: Often an anti-pattern in modern DI frameworks.

### Structural (Object Composition)
- **Adapter**: Make incompatible interfaces work together. Perfect for third-party API integration.
- **Decorator**: Add behavior dynamically without affecting other objects (e.g., caching, logging).
- **Facade**: Simplified interface to a complex subsystem.

### Behavioral (Object Communication)
- **Strategy**: Encapsulate interchangeable algorithms. Replaces complex `if/else` or `switch`.
- **Observer**: One-to-many dependency — when one changes, all dependents are notified.
- **Command**: Encapsulate a request as an object — enables queuing and undo.

## Anti-Patterns

- **God Class**: One class doing everything. Fix with SRP.
- **Copy-Paste Programming**: Duplicate code instead of abstracting. Fix with DRY.
- **Premature Abstraction**: Creating interfaces for every class "just in case." Fix with YAGNI.
- **Deep Inheritance Trees**: Inheriting 4+ levels deep. Fix with composition.
