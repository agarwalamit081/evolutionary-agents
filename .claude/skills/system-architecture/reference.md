---
description: System Architecture Reference
---

## Clean / Hexagonal Architecture

- **Domain Layer**: Enterprise business rules, entities, and interfaces. Zero external dependencies.
- **Application Layer**: Use cases, application business rules. Depends only on the Domain layer.
- **Infrastructure Layer**: Frameworks, databases, external APIs, UI. Implements interfaces defined in Domain/Application layers.

## C4 Model (Context, Containers, Components, Code)

- **Level 1: System Context** — High-level view of the system and external dependencies.
- **Level 2: Containers** — Applications, data stores, microservices, and communication.
- **Level 3: Components** — Internal structure of a container (controllers, services, repositories).
- **Level 4: Code** — Class-level details (usually generated or omitted).

## Scaling Strategies

- **Stateless Services**: Session data in Redis, not in-memory. Enables horizontal scaling behind a Load Balancer.
- **Database Scaling**: Read replicas for read-heavy workloads; Sharding for write-heavy/large datasets.

## Caching Patterns

- **Cache-Aside (Lazy Loading)**: Application checks cache first; if miss, reads DB and populates cache.
- **Write-Through**: Data written to cache and DB simultaneously. Ensures consistency but higher write latency.
- **TTL & Eviction**: Always define Time-To-Live and eviction policies (LRU, LFU) to prevent cache bloat.

## Asynchronous Processing

- Use Message Queues (RabbitMQ, Kafka, SQS) to decouple services, handle burst traffic, ensure eventual consistency.
- Implement **Idempotency** for all message consumers to safely handle duplicate deliveries.

## Resilience Patterns

- **Circuit Breaker**: Fail fast when a downstream service is unhealthy.
- **Bulkhead**: Isolate resources (thread pools) so one failing feature doesn't take down the whole system.
- **Retry with Exponential Backoff and Jitter**: Prevents thundering herd on recovery.

## Enterprise Architecture Checklist

- [ ] Are domain models free of framework annotations (no `@Entity` or `@Table` in core domain)?
- [ ] Is there a clear anti-corruption layer (ACL) for integrating with legacy or third-party systems?
- [ ] Are cross-cutting concerns (logging, auth, metrics) handled via middleware or interceptors, not duplicated?
- [ ] Is session/state externalized for horizontal scaling?
- [ ] Are all external calls wrapped with timeout, retry, and circuit breaker?
