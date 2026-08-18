# Structured Output Plugin

| Property | Value |
|---|---|
| ID | `structured-output` |
| Version | `1.3.0` |
| Max Score | 22 |
| Streaming | No |

The model returns exactly one JSON employee object. The request schema uses a conservative Ollama/llama.cpp-compatible subset: anchored patterns with explicit character classes, finite enums, typed nested objects/arrays, and explicit `additionalProperties: false`. Non-empty name, address-city/street, and tag-name requirements are prompt-level rather than JSON-schema string-length bounds, and the fractional metadata score range is validated after generation because llama.cpp does not grammar-enforce bounds on JSON `number` values. Scoring uses typed nested validation: UUID v4, email, integer ranges, enums, arrays, booleans, five-digit ZIP, ISO-8601 timestamps with timezone, and placeholder detection. Additional top-level keys and explanatory text outside a fenced object lose strict-format credit.
