# Structured Output Plugin

| Property | Value |
|---|---|
| ID | `structured-output` |
| Version | `1.0.0` |
| Max Score | 22 |
| Streaming | No |

The model returns exactly one JSON or YAML employee object. Scoring uses typed nested validation: UUID v4, email, integer ranges, enums, arrays, booleans, five-digit ZIP, ISO-8601 timestamps with timezone, and placeholder detection. YAML scalar timestamps are normalized deliberately. Additional top-level keys and explanatory text outside a fenced object lose strict-format credit.
