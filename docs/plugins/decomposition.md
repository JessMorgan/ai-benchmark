# Design-Doc Decomposition Plugin

| Property | Value |
|---|---|
| ID | `decomposition` |
| Name | Design-Doc Decomposition |
| Version | `0.2.0` |
| Max Score | 20 |
| Streaming | Yes |

## Task

The model is given a realistic design document for a distributed log-ingestion
pipeline and must decompose it into an ordered execution plan. The plan must
cover every deliverable in the document (durable buffered ingestion, GeoIP
enrichment + normalization, anomaly detection on the normalized stream,
real-time alert feed, nightly aggregate report, and observability/metrics
export), declare the dependency edges between tasks with `[DEPENDS_ON: N]`
tags, and justify the ordering and which stages run in parallel versus
sequentially. Task IDs may use any stable naming (`Task 1`, `T1`, ...); the
scorer does not require a specific scheme.

The rubric is deliberately **format-then-semantics**: emitting a well-formed
plan without reasoning about the document's actual deliverables or the
direction of its dependencies earns a fraction of the available marks. This
tests design-doc comprehension and decomposition quality rather than the
mechanical reproduction of tags, which is the failure mode of a purely
structural scorer.

## Scoring Rubric

| Criterion | Max | Description |
|---|---|---|
| Dependency graph validity | 4 | The plan parses as a well-formed task graph with at least one declared dependency edge; task IDs may be named `Task N` or `T<N>` |
| Coverage of design-doc deliverables | 6 | Every deliverable domain in the design document appears in the plan |
| Semantic dependency direction | 6 | Each required dependency earns one third of the criterion in the semantically correct direction; every declared reversed (forbidden) edge is penalized individually at the same weight, capped by the criterion maximum. Each task's domain is bound to the first line that mentions its ID, so an appended mapping section cannot override the plan |
| Parallelization reasoning | 2 | The plan identifies which stages run in parallel versus sequentially, and justifies the parallel stages with independence language (e.g. "independent", "no dependency") |
| Ordering rationale | 2 | The plan explains its ordering in terms of data flow or prerequisites and references the plan's specific tasks (by task ID or named deliverable) |

## Temperature

Default temperature can be set with:

```json
"decomposition_temperature": 0.5
```

## Notes

- The prompt embeds a fixed design document, so the plugin instance is
  stateless and thread-safe (a single shared instance is used across all
  model workers).
- `get_judge_instructions()` guides the semantic judge on coverage,
  dependency correctness, decomposition boundaries, and ordering rationale,
  and tells it not to penalize valid alternative decompositions.
