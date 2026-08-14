# Software Architecture Plugin

| Property | Value |
|---|---|
| ID | `software-architecture` |
| Name | Software Architecture |
| Version | `1.0.0` |
| Max Score | 20 |
| Streaming | Yes |

## Task

The model produces a comprehensive architecture for the FlowState platform.
The response is sectioned and should include:

- Executive Summary and Requirements Summary
- Architecture Style and Component Diagram / Description
- Real-Time Sync & Communication
- Data Model, API Design, and Technology Stack
- Deployment Architecture
- Resiliency & Failure Modes
- Security Considerations
- Scalability & Performance
- Trade-offs & Decisions
- Observability & SLOs

## Scoring

The evaluator first awards up to 3 points for the required section set. The
remaining points are section-local: architecture and components (2.5), data
and API design (2.5), real-time communication (2.5), scalability/capacity
(2.5), resiliency (2.5), security (2.5), and observability/SLOs (1.5). Capacity
claims without a workload estimate and availability claims without supporting
failure handling receive bounded deductions. Global mentions do not satisfy a
section's criterion.

## Temperature

```json
"software_architecture_temperature": 0.5
```
