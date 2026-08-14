# Logical Reasoning Plugin

| Property | Value |
|---|---|
| ID | `reasoning` |
| Version | `1.0.0` |
| Max Score | 20 |
| Streaming | Yes |

The task is a constrained scheduling puzzle. The evaluator parses the four
exact final fields and independently checks numbered time-chain deductions,
derived time assignments, ownership deductions, and the priority chain.
Contradictory service/time assignments are penalized; the final answer alone
cannot receive full credit.

The correct result is:

```text
FAILED_SERVICE: Search
OWNER: Ben
PRIORITY: P5
TIME: 09:30
```

P4 belongs to Profile; the ordering constraints leave Upload at P6, Search at
P5, and Billing at P3.
