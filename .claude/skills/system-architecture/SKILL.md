---
name: system-architecture
description: High-level software architecture and distributed systems design — modularity, scalability, maintainability, capacity planning, fault tolerance, and explicit trade-offs.
---

**When to Use**
- Designing a new system, major feature, or microservice.
- Refactoring a monolithic codebase into modular components.
- Evaluating technology stacks or architectural patterns (Clean Architecture, Hexagonal, Microservices).
- Addressing scalability, performance bottlenecks, or high availability.
- Choosing between databases, caching strategies, or message brokers.

**Core Rules**
1. **Separation of Concerns**: Strict boundaries between domains, application logic, and infrastructure.
2. **Dependency Rule**: Source code dependencies must point inward. Inner layers (Domain) must not depend on outer layers (Infrastructure/UI).
3. **Explicit over Implicit**: Favor explicit dependency injection over global state or hidden magic.
4. **Evolutionary Design**: Design for today's requirements, but structure code to adapt without massive rewrites.
5. **State Trade-offs Explicitly**: Always discuss CAP theorem implications, consistency vs. availability, latency vs. throughput.
6. **Design for Failure**: Assume networks drop, disks fail, dependencies timeout. Use retries, circuit breakers, dead-letter queues.
7. **Scale Out, Not Up**: Prefer horizontal scaling with stateless services over vertical scaling.
8. **Measure First**: Do not optimize or add complexity without estimated QPS, payload size, or latency requirements.

**References**
- Load `reference.md` for architectural patterns, scaling strategies, caching, resilience patterns, and the C4 model.
- Load `examples.md` for refactoring examples, capacity calculations, and Mermaid diagram templates.

**Scripts**
- `scripts/generate_c4_mermaid.py`: Generate C4 Context Mermaid diagrams from component descriptions.
- `scripts/capacity_estimator.py`: Estimate bandwidth, storage, and instance counts from QPS and payload size.
