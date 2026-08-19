# Data Transformation Plugin

| Property | Value |
|---|---|
| ID | `data-transformation` |
| Version | `1.0.2` |
| Max Score | 22 |
| Streaming | No |

This challenge evaluates deterministic multi-record processing rather than
mere JSON generation. The model receives an order feed containing historical
versions, duplicates, filtered records, inconsistent formatting, and sorting
ties. It must produce the current eligible records in a strict JSON object.

The task requires the model to:

1. Keep the newest version of each order.
2. Filter by payment status, refund state, amount, region, and channel.
3. Normalize customer names and numeric totals.
4. Sort by total descending and customer name ascending.
5. Assign deterministic ranks.
6. Compute count, total, and top-order summary fields.

The request schema remains strict and provider-compatible. It deliberately
leaves the `records` array unbounded at the provider boundary because some
llama.cpp grammar builds cannot compile bounded repetition of nested objects.
The evaluator still requires 1–5 records locally. Schema compliance is worth
only **one of 22 points**. The remaining points measure
selection/filtering, deduplication, normalization, sorting/ranking, and the
derived summary. A schema-valid response containing the wrong records can
therefore still score poorly.

| Criterion | Points |
|---|---:|
| Structured schema contract | 1 |
| Record selection and filtering | 7 |
| Deduplication and latest versions | 4 |
| Normalization | 3 |
| Sorting and ranking | 3 |
| Derived summary | 3 |
| Strict format and no decoys | 1 |
| **Total** | **22** |

The evaluator independently validates JSON and the request schema. Per-cell
metadata records `schema_requested`, `schema_request_status`,
`response_schema_valid`, `schema_enforcement_verified`,
`schema_fallback_used`, and `schema_fallback_error`. A normal valid response
cannot prove provider-side enforcement; use the separate `--schema-sentinel`
tool for that operational check.

If a provider returns a grammar-sampler initialization error while compiling
the strict schema, the benchmark retries that cell once with JSON-object mode.
The response is still validated locally against the full schema and semantic
rubric, and the metadata status becomes
`schema_fallback_json_object_valid` (or an invalid/failed variant). This lets
Nanbeige-like llama.cpp builds participate without treating server grammar
support as model-quality failure.

Use the plugin ID in all configurations:

```yaml
plugins_whitelist:
  - data-transformation
```

The former `structured-output` ID is no longer recognized. Update existing
configuration selectors and per-plugin temperature keys before running.
When an older state file contains the former task, the changed plugin set is
treated as a normal task change and requires the usual resume decision; its
scores are not reused for this task.
