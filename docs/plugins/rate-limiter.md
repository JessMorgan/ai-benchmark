# Rate Limiter Plugin

| Property | Value |
|---|---|
| ID | `rate-limiter` |
| Version | `1.0.0` |
| Max Score | 20 |
| Streaming | Yes |

## Contract

Implement `TokenBucket`, `SlidingWindowLog`, and `FixedWindow`. Each constructor is `(..., limit: int, window_seconds: float)` and each class must expose `allow_request(client_id: str, now: float) -> bool`, `get_usage_stats(client_id: str) -> dict`, and `cleanup(now: float) -> int`.

The evaluator executes all three strategies with deterministic time, independent clients, invalid limits, stale cleanup, and concurrent calls. The behavioral contract is worth 3 points; lexical mentions do not substitute for passing the API tests.
