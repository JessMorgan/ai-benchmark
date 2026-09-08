# Instruction Following Plugin

| Property | Value |
|---|---|
| ID | `instruction-following` |
| Version | `1.1.0` |
| Max Score | 20 |
| Streaming | Yes |

The evaluator parses `ORDER` lines that match the prompt's exact format
(`ORDER <id> | CUSTOMER <NAME> | TOTAL <amount>`) into records and requires
exactly the four eligible IDs, correct amount/name transformation,
amount/name tie-break ordering, and exact summary arithmetic. The match is a
strict `fullmatch` that is prompt-aligned: the prompt specifies two-decimal
amounts and no leading bytes, so a BOM or a three-decimal amount such as
`TOTAL 120.000` is rejected rather than coerced. Duplicate or unknown records
and extra lines are penalized by the exact-discipline criterion.
