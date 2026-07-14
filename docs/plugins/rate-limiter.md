# Rate Limiter Plugin

| Property | Value |
|---|---|
| ID | `rate-limiter` |
| Name | Rate Limiter |
| Version | `0.1.0` |
| Max Score | 20 |
| Streaming | Yes |

## Task

The model is asked to design and implement a concurrent rate limiter in Python supporting multiple strategies:

- Token Bucket
- Sliding Window Log
- Fixed Window

The implementation must include:

- A clean abstract base class or protocol
- Thread-safe code with minimal lock contention
- Per-client limits
- `allow_request(client_id: str) -> bool`
- `get_usage_stats(client_id: str) -> dict`
- Automatic cleanup of stale entries
- Edge-case handling (burst traffic, zero limits, concurrent requests)

## Scoring Rubric

| Criterion | Max | Description |
|---|---|---|
| Interface design | 3 | ABC/Protocol, `allow_request`, `get_usage_stats` |
| Token Bucket | 4 | Class, refill logic, consume logic |
| Sliding Window | 3 | Class, timestamp tracking, pruning |
| Thread safety | 3 | Locking, minimal contention |
| Cleanup | 2 | Stale entry eviction |
| Type hints | 2 | Parameter & return annotations |
| Docstrings | 2 | Comprehensive documentation |
| Error handling | 1 | Input validation, exceptions |

## Temperature

Default temperature can be set with:

```json
"rate-limiter_temperature": 0.2
```

## Tips for Models

- Use `threading.Lock` or `threading.RLock` for thread safety.
- Implement `__repr__` or `__str__` for easier debugging.
- Include usage examples in the response.
