# Software Architecture Plugin

| Property | Value |
|---|---|
| ID | `software-architecture` |
| Name | Software Architecture |
| Version | `0.1.0` |
| Max Score | 20 |
| Streaming | Yes |

## Task

The model acts as a senior backend/coding architect and produces a comprehensive software architecture document for the FlowState productivity platform based on a PRD.

The document must include:

1. Executive Summary
2. Requirements Summary
3. Architecture Style
4. Component Diagram / Description
5. Data Model
6. API Design
7. Technology Stack
8. Deployment Architecture
9. Security Considerations
10. Scalability & Performance
11. Trade-offs & Decisions

## Scoring Rubric

| Criterion | Max | Description |
|---|---|---|
| Executive Summary | 1 | Brief architecture overview |
| Requirements Summary | 2 | Functional and non-functional requirements |
| Architecture Style | 2 | Microservices, monolith, event-driven, etc. |
| Component Description | 3 | Major components and responsibilities |
| Data Model | 3 | Entities, relationships, storage choices |
| API Design | 2 | REST/GraphQL endpoints |
| Technology Stack | 2 | Languages, frameworks, databases |
| Deployment Architecture | 2 | Cloud, containers, CI/CD |
| Security Considerations | 2 | Auth, encryption, TLS |
| Scalability & Performance | 2 | Caching, load balancing, sharding |
| Trade-offs & Decisions | 1 | Rationale for major choices |

## Temperature

Default temperature can be set with:

```json
"software-architecture_temperature": 0.5
```

## Tips for Models

- Be specific with technology names.
- Include concrete endpoints or schema examples.
- Mention scalability numbers (e.g., 1M DAU).
- Justify major architectural decisions.
