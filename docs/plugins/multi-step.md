# Multi-Step Instructions Plugin

| Property | Value |
|---|---|
| ID | `multi-step` |
| Version | `1.3.0` |
| Max Score | 20 |
| Streaming | Yes |

The response must contain exactly three fenced Python blocks and the exact summary `[SUMMARY: 3 functions, 3 code blocks, completed all steps].` The required typed functions are `greet_user`, `validate_name`, and `format_greeting`.

The evaluator parses the definitions and executes boundary tests for greetings, spaces/invalid names, length limits, repetition, and `times < 1`. Missing fences, prose, stubs, or a main block lose contract points; behavioral tests dominate implementation credit.
