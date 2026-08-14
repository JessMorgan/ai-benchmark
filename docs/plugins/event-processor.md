# Concurrent Event Processor Plugin

| Property | Value |
|---|---|
| ID | `event-processor` |
| Version | `0.1.0` |
| Max Score | 20 |
| Streaming | Yes |

The model implements an exact `EventProcessor(handler, max_workers, max_retries)` API. The evaluator executes it with duplicate events, transient failures, permanent failures, and concurrent work. It verifies idempotency, retry counts, output ordering, and failed-event reporting rather than awarding implementation credit from keywords alone.
