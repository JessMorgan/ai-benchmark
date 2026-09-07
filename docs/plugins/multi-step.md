# Multi-Step Instructions Plugin

| Property | Value |
|---|---|
| ID | `multi-step` |
| Version | `1.4.0` |
| Max Score | 20 |
| Streaming | Yes |

The response must contain exactly three fenced Python blocks and the exact summary `[SUMMARY: 3 functions, 3 code blocks, completed all steps].` The required typed functions are `greet_user`, `validate_name`, and `format_greeting`.

The evaluator parses the definitions and executes boundary tests for greetings, spaces/invalid names, length limits, repetition, and `times < 1`. The "exactly three blocks" and "no prose" scans run on the whole text minus the Python blocks, so prose hidden in a non-Python fence (e.g. a ```text block) and a `__main__` guard inside a Python block are penalized. The "no forbidden prose" discipline point is credited only when at least one Python block defines a required function. The 11-point behavioral tests are credited only when the harness ran to completion (`harness_ok`); an early exit (`sys.exit` / `SystemExit`) before the completion sentinel scores 0 for them, leaving a maximum of 9.0 for an otherwise-correct response.
