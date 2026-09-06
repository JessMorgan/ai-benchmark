# Concurrent Event Processor Plugin

| Property | Value |
|---|---|
| ID | `event-processor` |
| Version | `0.4.0` |
| Max Score | 20 |
| Streaming | Yes |

The model implements an exact `EventProcessor(handler, max_workers, max_retries)` API. The evaluator executes it in the sandbox in two independent phases:

- **Validation semantics (4 points)** — malformed events (non-dict element, non-string id, empty id, missing id) and invalid constructor arguments (wrong types, out-of-range values, non-callable handler) must raise `TypeError`/`ValueError`, never be silently skipped or dropped.
- **Behavioral event tests (8 points)** — duplicate events within a call and across separate `process()` calls (idempotency is instance-scoped), transient-failure retry counts, permanent-failure reporting, output ordering, and concurrent execution.

The two phases are scored separately, so a response with correct behavior but a validation gap (or vice versa) earns partial credit instead of losing the whole execution block. Execution evidence in `meta.json` retains the sandbox output and error tails for diagnosis. Lexical mentions do not substitute for passing the harness.
