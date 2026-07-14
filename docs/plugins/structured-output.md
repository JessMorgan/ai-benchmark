# Structured Output Plugin

| Property | Value |
|---|---|
| ID | `structured-output` |
| Name | Structured Output |
| Version | `0.2.0` |
| Max Score | 20 |
| Streaming | No |

## Task

The model must produce a valid JSON or YAML object representing an employee record. No explanatory text outside the structured data.

Required top-level fields:

- `id` — UUID v4 string
- `name` — string
- `age` — integer, 18–120
- `email` — valid email
- `department` — one of `Engineering`, `Sales`, `Marketing`, `HR`
- `roles` — array of strings from `admin`, `editor`, `viewer`, `auditor`
- `address` — object with `street`, `city`, `state` (2-letter US code), `zip` (5-digit string)
- `settings` — object with `theme` (`dark`, `light`, `auto`), `notifications` (object with `email`, `sms`, `push` booleans), `language` (2-letter ISO code)
- `tags` — array of objects with `name` and `priority` (1–5)
- `metadata` — object with `created_at` (ISO 8601 datetime), `active` (boolean), `score` (float 0.0–1.0)

## Scoring Rubric

| Criterion | Max | Description |
|---|---|---|
| Valid JSON/YAML syntax | 4 | Parses successfully |
| Required fields present | 4 | All top-level fields |
| Basic types and constraints | 6 | UUID, email, ranges, enums |
| Completeness | 4 | Non-empty values |
| Strict format | 2 | No extra top-level keys |
| No placeholders | 2 | No `unknown`, `n/a`, `null`, etc. |

## Temperature

Default temperature can be set with:

```json
"structured-output_temperature": 0.2
```

## Tips for Models

- Return only the structured data, ideally inside a markdown code fence.
- Use exact formats for UUID, email, state codes, zip, and ISO datetime.
