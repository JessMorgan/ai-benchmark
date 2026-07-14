# Orchestration & Workflow Plugin

| Property | Value |
|---|---|
| ID | `orchestration` |
| Name | Orchestration & Workflow |
| Version | `0.1.0` |
| Max Score | 16 |
| Streaming | Yes |

## Task

The model acts as an orchestration AI handling a complex data pipeline:

> Process 1TB of raw server logs, perform GeoIP lookup on IPs, run anomaly detection, and generate a final PDF report.

The model must produce:

1. Task decomposition into distinct steps
2. Dependency graph with `[PARALLEL]` and `[SEQUENTIAL]` markers and `[DEPENDS_ON: task_id]` tags
3. Execution trace showing initialization, running, and completion states

## Scoring Rubric

| Criterion | Max | Description |
|---|---|---|
| Task breakdown | 4 | Distinct steps covering the pipeline |
| Dependency tagging | 4 | `[DEPENDS_ON: ...]` tags |
| Parallel/sequential logic | 4 | `[PARALLEL]` and `[SEQUENTIAL]` markers |
| Execution trace | 4 | Init/start/running/complete states |

## Temperature

Default temperature can be set with:

```json
"orchestration_temperature": 0.5
```

## Tips for Models

- Use explicit tags so the scorer can detect them.
- Show a clear execution trace with state transitions.
