# Structured Output Plugin

| Property | Value |
|---|---|
| ID | `structured-output` |
| Version | `1.5.0` |
| Max Score | 22 |
| Streaming | No |

The model must extract the current employee profile from a source packet containing an archived decoy, then normalize and derive the requested fields. It must reverse `Family, Given` names, calculate age from a fixed reference date, normalize email and locale, map organization/access labels, split and normalize the address, sort roles/tags, convert timestamps, and convert a percentage score to a decimal.

The strict request schema remains useful for removing syntax and shape noise, but schema compliance is worth only **one point**. The rest of the score comes from semantic extraction, normalization, current-profile selection, strict output shape, and placeholder avoidance. A schema-valid response with archived or invented values therefore still scores poorly.

The evaluator independently validates the returned object against the request schema and stores these diagnostic fields in the per-plugin result metadata:

- `schema_requested`
- `schema_request_status`
- `response_schema_valid`
- `schema_enforcement_verified`

A normal benchmark response cannot prove that the provider enforced the schema, so `schema_enforcement_verified` remains false. Run the separate compatibility probe with `--schema-sentinel` when provider enforcement needs to be checked. The probe does not affect benchmark scores or state.

The schema uses a conservative Ollama/llama.cpp-compatible subset: anchored patterns with explicit character classes, finite enums, typed nested objects/arrays, and explicit `additionalProperties: false`.
