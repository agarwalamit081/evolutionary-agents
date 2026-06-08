---
name: code-quality-and-patterns
description: SOLID principles, DRY, and standard design patterns (GoF + modern) for maintainable, extensible, and testable code.
---

**When to Use**
- During code reviews or refactoring sessions.
- When a class is becoming a "God Class" (too many responsibilities).
- When code exhibits smells (massive if/else chains, tight coupling, duplicated creation logic).
- When writing new interfaces or abstract base classes.
- When changes in one part of the system unexpectedly break unrelated parts.

**Core Rules**
1. **Single Responsibility**: A class should have one, and only one, reason to change.
2. **Open/Closed**: Extend behavior without modifying existing source code.
3. **Liskov Substitution**: Subtypes must be substitutable for base types without altering correctness.
4. **Interface Segregation**: Many client-specific interfaces are better than one general-purpose interface.
5. **Dependency Inversion**: Depend on abstractions (interfaces/protocols), not concretions.
6. **DRY**: Don't Repeat Yourself — extract shared logic into reusable functions or classes.
7. **YAGNI**: Don't apply a pattern just to use a pattern. Favor composition over inheritance.
8. **Name Intent, Not Implementation**: `PaymentStrategy` beats `PayPalOrStripeHandler`.

**References**
- Load `reference.md` for SOLID principle details, anti-patterns, and design pattern catalog.
- Load `examples.md` for step-by-step refactoring of violations.

**Scripts**
- `scripts/solid_linter.py`: AST analysis for potential SRP/DIP violations.
- `scripts/pattern_suggester.py`: Keyword-based design pattern recommendations.
