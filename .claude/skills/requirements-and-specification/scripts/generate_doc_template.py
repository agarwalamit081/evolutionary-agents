"""Unified document template generator for requirements and specifications.

Usage:
    python generate_doc_template.py --type brd [--output BRD_Draft.md]
    python generate_doc_template.py --type prd [--output PRD_Draft.md]
    python generate_doc_template.py --type frd [--output FRD_Draft.md]
    python generate_doc_template.py --type srs [--output SRS_Draft.md]
    python generate_doc_template.py --type trd [--output TRD_Draft.md]
"""
import argparse
from datetime import datetime

TEMPLATES = {
    "brd": """# Business Requirement Document: [Project Name]

## 1. Document Control
- **Project Name:** [Project Name]
- **Document Version:** 1.0
- **Author/Owner:** [Name/Title]
- **Date:** {date}
- **Status:** Draft
- **Approvers:** [Sponsor Name], [Key Stakeholder Name]

## 2. Executive Summary
- **TL;DR:** [2-3 sentences: initiative, problem, expected value]

## 3. Business Objectives & Drivers
- **Current State / Pain Points:** [What is broken or inefficient]
- **Future State / Vision:** [Desired outcome after implementation]
- **Strategic Alignment:** [Link to company OKRs or annual goals]

## 4. Stakeholder Analysis
| Stakeholder | Role | Interest | Communication Cadence |
|---|---|---|---|
| [Name] | Sponsor | ROI, budget | Bi-weekly |
| [Name] | Primary User | Workflow efficiency | Weekly demos |

## 5. Project Scope
- **In Scope:**
  - [Capability 1]
  - [Capability 2]
- **Out of Scope:**
  - [Explicitly excluded items]
- **Assumptions & Constraints:**
  - [Budget cap, regulatory requirements, deadlines]

## 6. Cost-Benefit Analysis & ROI
| Category | Item | Annual Amount |
|---|---|---|
| Cost | Licensing | $X |
| Cost | Implementation | $Y |
| Savings | Labor reallocation | $A |
| Savings | Error reduction | $B |
| **Net** | **Year 1 ROI** | **X% — Payback: N months** |

## 7. High-Level Timeline
| Phase | Description | Target Date |
|---|---|---|
| Phase 1 | Discovery & Design | [Month Year] |
| Phase 2 | Development & Testing | [Month Year] |
| Phase 3 | Deployment & Training | [Month Year] |

## 8. Risk Assessment
| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| [Risk 1] | High/Med/Low | High/Med/Low | [Strategy] |

## 9. Approval & Sign-off
- [ ] [Sponsor Name], [Title] — Date: _________
- [ ] [Stakeholder Name], [Title] — Date: _________
""",

    "prd": """# Product Requirement Document: [Product/Feature Name]

## 1. Executive Summary
- **Owner:** [Name]
- **Status:** Draft
- **TL;DR:** [2-3 sentences summarizing what we're building and why]

## 2. Problem Statement & Opportunity
- **Current State:** [What is the pain point or limitation?]
- **Proposed State:** [How does this solve the problem?]
- **Business Value:** [Why now?]

## 3. Target Audience
- **Primary Persona:** [Who is this for? Role, needs, frustrations]
- **Secondary Persona:** [Who else is affected?]

## 4. Goals & Success Metrics
- **North Star Metric:** [One key metric]
- **Key Results:**
  - KR1: e.g., "Increase conversion by X%"
  - KR2: e.g., "Reduce support tickets by Y%"

## 5. User Stories & Acceptance Criteria
- [ ] As a [user], I want to [action], so that [benefit].
  - *AC:* Given [context], When [action], Then [result]
  - *Edge Cases:* [Boundary conditions and error states]

## 6. Out of Scope
- [Explicitly listed exclusions to prevent scope creep]

## 7. Dependencies & Risks
- **Dependencies:** [Technical, legal, resource]
- **Risks:** [What could go wrong and mitigation]
""",

    "frd": """# Functional Requirement Document: [System Name]

## 1. Document Control
- **Version:** 1.0
- **Date:** {date}
- **Author:** [Name]
- **Approver:** [Name]

## 2. System Overview
[Brief description of the system and its boundaries]

## 3. Functional Requirements

### Module: [Module Name]

**FR-[MOD]-001:** The system shall [action].
- **Priority:** High / Medium / Low
- **Source:** PRD Section [X.X]
- **Acceptance Criteria:**
  - Given [context]
  - When [action]
  - Then [result]

**FR-[MOD]-002:** The system shall [action].
- **Priority:** [Level]
- **Source:** PRD Section [X.X]
- **Acceptance Criteria:**
  - Given [context]
  - When [action]
  - Then [result]

## 4. Business Rules
- [Explicit logic, e.g., "Discounts cannot exceed 50% of total cart value"]
- [Validation rules, e.g., "Email must match RFC 5322 format"]

## 5. Data Requirements
| Field | Type | Required | Validation |
|---|---|---|---|
| [field_name] | [type] | Yes/No | [rule] |

## 6. Error Handling
| Error Condition | System Behavior | User Message |
|---|---|---|
| [Condition] | [Behavior] | [Message] |
""",

    "srs": """# System Requirements Specification: [System Name]

## 1. Introduction
- **Purpose:** [What this document covers]
- **Scope:** [System boundaries]
- **Definitions:** [Key terms]
- **References:** [BRD, PRD, FRD links]

## 2. Overall Description
- **Product Perspective:** [How this system fits in the larger ecosystem]
- **User Classes:** [Admin, End User, API Consumer, etc.]
- **Operating Environment:** [Cloud, on-prem, browser, mobile]
- **Design Constraints:** [Tech stack mandates, regulatory]

## 3. Specific Requirements

### 3.1 Functional Requirements
[Link to FRD requirements by ID, e.g., FR-AUTH-01, FR-ORD-03]

### 3.2 External Interface Requirements
- **User Interfaces:** [Web dashboard, mobile app]
- **Software Interfaces:** [APIs consumed and exposed]
- **Communications Interfaces:** [Protocols, formats]

## 4. Non-Functional Requirements

| ID | Category | Requirement | Metric | Target |
|---|---|---|---|---|
| NFR-PERF-01 | Performance | API response latency | p95 latency | <200ms |
| NFR-PERF-02 | Performance | Throughput | concurrent users | 10,000 |
| NFR-SEC-01 | Security | Data at rest encryption | Algorithm | AES-256 |
| NFR-SEC-02 | Security | Auth token expiry | TTL | 15 minutes |
| NFR-REL-01 | Reliability | Uptime | Availability | 99.99% |
| NFR-MAINT-01 | Maintainability | Test coverage | % | >80% |

## 5. Compliance & Regulatory
- [GDPR / CCPA / HIPAA / SOC2 / industry-specific requirements]
""",

    "trd": """# Technical Requirements Document: [System Name]

## 1. Context & Goals
- **Parent Documents:** [Link to PRD/FRD]
- **Technical Goals:** [e.g., "Migrate from monolith to microservice architecture"]

## 2. Architecture Overview

```mermaid
graph TD
    Client[Client] -->|HTTPS| APIGW[API Gateway]
    APIGW --> Auth[Auth Service]
    APIGW --> Core[Core Service]
    Core --> DB[(Database)]
    Core --> Cache[(Cache)]
```

## 3. Technology Stack
| Layer | Technology | Justification |
|---|---|---|
| Frontend | [Framework] | [Why] |
| Backend | [Language/Framework] | [Why] |
| Database | [DB] | [Why] |
| Cache | [Redis/Memcached] | [Why] |
| Queue | [Kafka/RabbitMQ/SQS] | [Why] |

**Alternatives Considered:**
- [Option A] vs [Option B]: Chose [A] because [reason].

## 4. Data Model

```mermaid
erDiagram
    USER ||--o{ ORDER : places
    ORDER ||--|{ ORDER_ITEM : contains
    ORDER_ITEM }o--|| PRODUCT : references
```

## 5. API Specifications
[Key endpoints with request/response contracts — see examples.md for format]

## 6. Infrastructure & Deployment
- **Cloud Provider:** [AWS/GCP/Azure]
- **Containerization:** [Docker/K8s]
- **CI/CD:** [Pipeline stages]
- **Environments:** Dev → Staging → Prod

## 7. Cross-Cutting Concerns
- **AuthN/AuthZ:** [JWT, OAuth2, session management]
- **Observability:** Logging ([format]), Metrics ([Prometheus/Datadog]), Tracing ([Jaeger])
- **Testing Strategy:** Unit → Integration → E2E → Load

## 8. Rollout & Rollback Plan
- **Strategy:** [Feature flags / Canary / Blue-Green]
- **Rollback Trigger:** [Error rate >X%, latency >Yms]
- **Rollback Procedure:** [Steps]
"""
}


def main():
    parser = argparse.ArgumentParser(description="Generate requirements document templates")
    parser.add_argument("--type", required=True, choices=["brd", "prd", "frd", "srs", "trd"],
                        help="Document type to generate")
    parser.add_argument("--output", help="Output filename (default: auto-generated)")
    args = parser.parse_args()

    template = TEMPLATES[args.type].format(date=datetime.now().strftime("%Y-%m-%d"))
    filename = args.output or f"{args.type.upper()}_Draft.md"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(template)

    print(f"Generated: {filename}")


if __name__ == "__main__":
    main()
