# Concurrent Event Processor Plugin

| Property | Value |
|---|---|
| ID | `event-processor` |
| Version | `0.5.0` |
| Max Score | 20 |
| Streaming | Yes |

The model implements an exact `EventProcessor(handler, max_workers, max_retries)` API. The evaluator executes it in the sandbox in two independent phases:

- **Validation semantics (4 points)** — malformed events (non-dict element, non-string id, empty id, missing id) must raise `TypeError` or `ValueError`, and invalid constructor arguments must raise `TypeError` for wrong types (a non-callable `handler`, a non-int `max_workers`) or `ValueError` for out-of-range values (`max_workers < 1`, `max_retries < 0`); input is never silently skipped or dropped. Each check expects its own exception type, so a response that raises `ValueError` for a wrong-type check is not credited.
- **Behavioral event tests (8 points)** — duplicate events within a call and across separate `process()` calls (idempotency is instance-scoped), transient-failure retry counts, permanent-failure reporting, and output ordering.

Both execution phases are gated on a completion sentinel (`HARNESS_EXEC_OK`): a response that exits `0` before the harness runs (`sys.exit(0)`/`os._exit(0)`/`SystemExit`) is not credited, so a zero-implementation response cannot masquerade as a clean pass.

The two phases are scored separately, so a response with correct behavior but a validation gap (or vice versa) earns partial credit instead of losing the whole execution block. Execution evidence in `meta.json` retains the sandbox output and error tails for diagnosis. Lexical mentions do not substitute for passing the harness.

**Concurrency is assessed lexically, not by a runtime probe.** The behavioral harness runs its assertions sequentially and contains no true-concurrency check; a real overlap probe is deferred because it is flaky under the resource-limited local fallback. Concurrency is instead credited through the lexical "Concurrent idempotent design" criterion (mentions of `ThreadPoolExecutor`/`asyncio`, retry, dedup/idempotency, failure handling, and locking). The keyword criteria are pure lexical hits and are deliberately bounded (a prose-only response caps well below the maximum) — a documented lexical-design tradeoff, not a gap.
