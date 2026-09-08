# Error Recovery Plugin

| Property | Value |
|---|---|
| ID | `error-recovery` |
| Version | `1.5.0` |
| Max Score | 20 |
| Streaming | Yes |

The model implements an exact injectable asynchronous `WeatherClient.fetch` API and `AllProvidersFailedError`, plus `get_weather_resilient(city, client)` that tries every provider concurrently, treats a malformed error payload as a failure, returns the first successful response unchanged, and raises `AllProvidersFailedError` with per-provider details when all fail.

The rubric splits into lexical and executable parts:

- **Lexical design (10 points)** — required API contract (3), typed injectable signatures (2; accepts both `Name` and string-literal forward-reference annotations), recovery-design concepts (2; the concurrency concept accepts bare `gather`/`ensure_future` calls, not only `asyncio.`-prefixed forms), non-stub implementation (1), demo scenarios (1; accepts hyphenated `all-success`/`partial-failure`/`all-failure` labels), and type hints/docstrings (1).
- **Behavioral execution (10 points)** — four independent 2.5-point mode criteria (all-success, partial-failure, error-payload, all-failure) instead of a single all-or-nothing 10-point block, so a response correct in three modes keeps 7.5 points.

The behavioral criteria are gated on a completion sentinel (`HARNESS_EXEC_OK`): a response that exits `0` before the harness runs (`sys.exit(0)`/`os._exit(0)`/`SystemExit`) — including one that prints fake `MODE_RESULT` markers — is not credited, so a zero-implementation response cannot masquerade as a clean pass. The execution preamble also blocks the network-facing socket entry points (`create_connection`/`getaddrinfo`) so a module-level demo cannot hang the local-restricted check on a live call.

**Concurrency is assessed lexically, not by a runtime probe.** The behavioral harness runs its modes sequentially and contains no true-concurrency check; a real overlap probe is deferred because it is flaky under the resource-limited local fallback. Concurrency is instead credited through the lexical "Recovery design" criterion (mentions of `asyncio.gather`/`create_task`/`as_completed`/`ensure_future`, bare `gather(`/`ensure_future(` calls, or `TaskGroup`). The keyword criteria are pure lexical hits and are deliberately bounded (a prose-only response caps well below the maximum) — a documented lexical-design tradeoff, not a gap.
