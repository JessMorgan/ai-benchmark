# Structured Output Plugin

| Property | Value |
|---|---|
| ID | `structured-output` |
| Version | `1.2.0` |
| Max Score | 22 |
| Streaming | No |

The model returns exactly one JSON employee object. The prompt requires non-empty name, address-city/street, and tag-name strings; those are prompt-level requirements rather than JSON-schema string-length bounds so llama.cpp can compile the schema grammar reliably. Scoring uses typed nested validation: UUID v4, email, integer ranges, enums, arrays, booleans, five-digit ZIP, ISO-8601 timestamps with timezone, and placeholder detection. Additional top-level keys and explanatory text outside a fenced object lose strict-format credit.
