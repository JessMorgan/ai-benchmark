# PRD Creation Plugin

| Property | Value |
|---|---|
| ID | `prd-creation` |
| Name | PRD Creation |
| Version | `0.1.0` |
| Max Score | 20 |
| Streaming | Yes |

## Task

The model acts as a senior product manager and creates a comprehensive Product Requirements Document (PRD) for the FlowState productivity app idea.

The PRD must include:

1. Executive Summary
2. Problem Statement
3. Goals & Objectives
4. Target Users & Personas
5. User Stories
6. Functional Requirements
7. Non-Functional Requirements
8. Success Metrics / KPIs
9. Competitive Analysis
10. Timeline / Milestones
11. Open Questions / Risks

## Scoring Rubric

| Criterion | Max | Description |
|---|---|---|
| Executive Summary | 2 | Overview of the product |
| Problem Statement | 2 | Pain points and challenges |
| Goals & Objectives | 2 | Specific, measurable goals |
| Target Users & Personas | 2 | Two distinct personas |
| User Stories | 2 | At least 3 properly formatted stories |
| Functional Requirements | 3 | Distinct features and capabilities |
| Non-Functional Requirements | 2 | Performance, security, reliability, scalability |
| Success Metrics / KPIs | 2 | Quantitative metrics |
| Competitive Analysis | 2 | At least 2 competitors |
| Timeline / Milestones | 2 | Phases or release milestones |
| Open Questions / Risks | 1 | Risks and unresolved questions |

## Temperature

Default temperature can be set with:

```json
"prd-creation_temperature": 0.5
```

## Tips for Models

- Use clear headings for each section.
- Make goals specific and measurable (percentages, timeframes).
- Format user stories as `As a <persona>, I want <goal>, so that <benefit>`.
- Compare against real competitors with specific differentiators.
