# Structured Output Plugin

| Property | Value |
|---|---|
| ID | `structured-output` |
| Version | `1.4.0` |
| Max Score | 22 |
| Streaming | No |

The model must extract the current employee profile from a source packet containing an archived decoy, then normalize and derive the requested fields. It must reverse `Family, Given` names, calculate age from a fixed reference date, normalize email and locale, map organization/access labels, split and normalize the address, sort roles/tags, convert timestamps, and convert a percentage score to a decimal. The strict request schema still removes syntax and shape noise, but most semantic points are awarded by exact comparison with the current profile rather than by merely producing schema-valid JSON. The evaluator separately checks typed constraints, placeholders, extra keys, and explanatory text. The schema uses a conservative Ollama/llama.cpp-compatible subset: anchored patterns with explicit character classes, finite enums, typed nested objects/arrays, and explicit `additionalProperties: false`.
