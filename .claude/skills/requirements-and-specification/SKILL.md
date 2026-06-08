---
name: requirements-and-specification
description: Unified requirements and specification generator covering user stories, business logic, scope, functional requirements, architecture, endpoints, data models, and technical constraints. Supports BRD, PRD, FRD, SRS, and TRD document types.
---

**When to Use**
- Writing any form of requirements or specification document.
- Defining business cases, product requirements, functional specs, system requirements, or technical designs.
- Translating business needs into engineering specifications.

**Document Type Selection**

| Audience | Document | Focus |
|---|---|---|
| Executives, Sponsors | **BRD** | ROI, cost-benefit, stakeholder alignment, risk |
| Product, Design, Eng | **PRD** | User stories, success metrics, personas, scope |
| QA, Engineering | **FRD** | Testable functional specs, business rules, edge cases |
| System Architects | **SRS** | Non-functional requirements, compliance, performance |
| Engineering, DevOps | **TRD** | Architecture, tech stack, APIs, data models, deployment |

Progression: BRD → PRD → FRD → SRS → TRD (increasing technical detail).

**Core Principles**
1. **Audience Awareness**: Tailor language and detail level to the document's primary audience.
2. **Quantifiable**: Every metric, NFR, or success criterion must have numbers (e.g., "99.9% uptime", "<200ms latency").
3. **Traceability**: Link requirements back to parent documents (FR → PRD section, NFR → BRD goal).
4. **Edge Cases**: Proactively identify error states, boundary conditions, and fallback behaviors.
5. **No Fluff**: Concise, action-oriented language. Avoid marketing jargon.

**Workflow**
1. Identify the document type based on audience and depth needed.
2. Read `reference.md` for the standard structure of the selected document type.
3. Generate the document with clear placeholders `[LIKE THIS]` for missing information.
4. For TRDs, suggest Mermaid.js diagrams for architecture and data models.
5. Read `examples.md` for tone and formatting patterns.

**Scripts**
- `scripts/generate_doc_template.py`: Generate a baseline markdown template for any document type. Usage: `python generate_doc_template.py --type brd|prd|frd|srs|trd [--output FILENAME]`
