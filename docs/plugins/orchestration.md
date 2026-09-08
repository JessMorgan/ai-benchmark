# Orchestration & Workflow Plugin

| Property | Value |
|---|---|
| ID | `orchestration` |
| Name | Orchestration & Workflow |
| Version | `1.3.0` |
| Max Score | 16 |
| Streaming | Yes |

## Task

The model acts as an orchestration AI handling a complex data pipeline:

> Process 1TB of raw server logs, perform GeoIP lookup on IPs, run anomaly detection, and generate a final PDF report.

The model must produce:

1. Task decomposition into exactly four tasks with IDs 1-4 (a `task N`/`step N` prefix or a numbered-list declaration such as `1.` both work)
2. A dependency graph with `[DEPENDS_ON: task_id]` tags (plain-language dependencies such as "Task 2 depends on Task 1" are also accepted)
3. A parallel or sequential label per task
4. An execution trace showing initialization, running, and completion states for every task

## Scoring Rubric

| Criterion | Max | Description |
|---|---|---|
| Task breakdown | 4 | Credit scales with the lesser of ID coverage (IDs 1-4 declared) and operation coverage (how many of the four tasks reference a pipeline operation: logs, GeoIP, anomaly detection, PDF report). Declaring more than four tasks is penalized by 2.0; numbered-list items count as declared tasks, so a numbered summary list of five or more items also triggers the penalty. |
| Dependency tagging | 4 | 4.0 for a valid graph with 3+ edges, 2.0 for a valid graph with 1-2 edges, 0.0 otherwise (an invalid graph — cyclic, referencing unknown tasks, or a task labeled both parallel and sequential — earns no credit, in both `task N`/`step N` and numbered-list formats). |
| Parallel/sequential logic | 4 | Each task needs one non-contradictory parallel/sequential label; a contradiction anywhere in a task's block (consistent with the graph validator's per-task label check, and also covering numbered lines the validator ignores) fails the criterion. |
| Execution trace | 4 | Every task needs init/start/running and complete/done/finish states. |

## Temperature

Default temperature can be set with:

```json
"orchestration_temperature": 0.5
```

## Tips for Models

- Use exactly four tasks with IDs 1-4, and name the pipeline operation in each task.
- Mark each task parallel or sequential — not both, even across lines.
- Show a clear execution trace with state transitions for every task.
