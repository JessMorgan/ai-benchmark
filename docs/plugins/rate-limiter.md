# Rate Limiter Plugin

| Property | Value |
|---|---|
| ID | `rate-limiter` |
| Version | `1.4.0` |
| Max Score | 20 |
| Streaming | Yes |

## Contract

Implement `TokenBucket`, `SlidingWindowLog`, and `FixedWindow`. Each constructor is `(..., limit: int, window_seconds: float)` and each class must expose `allow_request(client_id: str, now: float) -> bool`, `get_usage_stats(client_id: str) -> dict`, and `cleanup(now: float) -> int`.

Validation semantics (since 1.4.0): a `limit` of zero or less raises `ValueError` — zero is invalid configuration, not a "deny all" mode — as do non-positive or non-finite `window_seconds`; wrong argument types raise `TypeError`.

The evaluator executes all three strategies with deterministic time, independent clients, invalid limits (`0`), invalid `window_seconds` (`0.0`, negative, `nan`, `inf` — the non-finite cases only pass with an explicit `math.isfinite`-style guard), stale cleanup, and concurrent calls. The behavioral contract is worth 10 points; lexical mentions do not substitute for passing the API tests. Execution evidence in `meta.json` retains the sandbox output and error tails for diagnosis (before 1.4.0, a failure was only recorded as `container exited 1` on the Podman path — `process exited N` on the `local-restricted` fallback — which made post-run diagnosis, e.g. a `get_usage_stats` referencing the undefined `now`, need a local re-run).
