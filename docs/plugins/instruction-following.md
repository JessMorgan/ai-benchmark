# Instruction Following Plugin

| Property | Value |
|---|---|
| ID | `instruction-following` |
| Version | `1.0.0` |
| Max Score | 20 |
| Streaming | Yes |

The evaluator parses every `ORDER` line into an exact record. It requires exactly the four eligible IDs, correct amount/name transformation, amount/name tie-break ordering, exact summary arithmetic, no duplicates or unknown records, and no extra fields or lines.
