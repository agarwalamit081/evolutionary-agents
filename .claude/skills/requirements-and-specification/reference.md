---
description: Requirements and Specification Reference Structures
---

## BRD — Business Requirement Document

**Audience:** Executives, Sponsors, Project Managers

1. **Document Control** — Project Name, Version, Author, Date, Status, Approvers
2. **Executive Summary** — 2-3 sentences: initiative, problem, expected value
3. **Business Objectives & Drivers**
   - Current State / Pain Points
   - Future State / Vision
   - Strategic Alignment (OKRs)
4. **Stakeholder Analysis** — Sponsor, Primary Stakeholders, Secondary Stakeholders
5. **Project Scope** — In Scope, Out of Scope, Assumptions & Constraints
6. **Cost-Benefit Analysis & ROI** — Estimated Costs, Expected Benefits, Payback Period
7. **High-Level Timeline & Milestones** — Phased delivery dates
8. **Risk Assessment & Mitigation** — Risk, Impact (H/M/L), Mitigation Strategy
9. **Approval & Sign-off**

## PRD — Product Requirement Document

**Audience:** Engineering, Design, Product

1. **Executive Summary** — Product/Feature Name, Owner, Status, TL;DR
2. **Problem Statement & Opportunity** — Current State, Proposed State, Business Value
3. **Target Audience** — Primary Persona, Secondary Persona
4. **Goals & Success Metrics** — North Star Metric, Key Results (OKRs)
5. **Requirements (User Stories)** — Format: `As a [user], I want to [action], so that [benefit].` Include Acceptance Criteria (Given/When/Then).
6. **Out of Scope** — Explicitly list exclusions
7. **Dependencies & Risks** — Technical, legal, resource dependencies

## FRD — Functional Requirement Document

**Audience:** QA, Engineering

1. **Document Control** — Version, Date, Author, Approver
2. **System Overview** — System boundaries, context diagram reference
3. **Functional Requirements** — Format: `FR-[Module]-[Number]`: The system shall [action].
   - Priority (High/Medium/Low)
   - Source (PRD/BRD link)
   - Acceptance Criteria (Given/When/Then)
4. **Business Rules** — Explicit logic (e.g., "Discounts cannot exceed 50%")
5. **Data Requirements** — Required fields, data types, validation rules
6. **Error Handling** — Error messages and system behaviors for failure states

## SRS — System Requirements Specification

**Audience:** System Architects, Compliance, Engineering Leads

1. **Introduction** — Purpose, Scope, Definitions, Acronyms, References
2. **Overall Description** — Product Perspective, User Classes, Operating Environment, Design Constraints
3. **Specific Requirements**
   - Functional Requirements (linked to FRD)
   - External Interface Requirements (UI, Hardware, Software/APIs, Communications)
4. **Non-Functional Requirements** — MUST have numbers
   - Performance: "Support 10,000 concurrent users with <2s page load"
   - Security: "All data at rest encrypted using AES-256"
   - Reliability: "99.99% availability"
   - Maintainability: "Code coverage above 80%"
5. **Compliance & Regulatory** — GDPR, HIPAA, SOC2, industry standards

## TRD — Technical Requirements Document

**Audience:** Senior Engineers, Architects, DevOps

1. **Context & Goals** — Link to PRD/FRD, technical goals
2. **Architecture Overview** — High-level system diagram (Mermaid.js), component interaction
3. **Technology Stack** — Languages, Frameworks, Databases, Third-party services
   - Alternatives Considered: Why X over Y
4. **Data Model** — ERD (Mermaid), schema definitions, data retention policies
5. **API Specifications** — Endpoints, request/response payloads (JSON), authentication
6. **Infrastructure & Deployment** — Cloud provider, containerization, CI/CD stages, environments
7. **Cross-Cutting Concerns** — AuthN/AuthZ, observability (logging, metrics, tracing), testing strategy
8. **Rollout & Rollback Plan** — Feature flags, canary release, rollback triggers and procedures
