# Multi-Step Instructions Plugin

| Property | Value |
|---|---|
| ID | `multi-step` |
| Name | Multi-Step Instructions |
| Version | `0.1.0` |
| Max Score | 20 |
| Streaming | Yes |

## Task

The model must follow a set of multi-step instructions exactly and produce three Python functions plus a formatted summary line.

Required artifacts:

1. `greet_user(name: str) -> str` returning `'Hello, <name>! Welcome.'`
2. `validate_name(name: str) -> bool` checking non-empty, alphabetic/spaces only, and length ≤ 50
3. `format_greeting(greeting: str, times: int) -> str` returning the greeting repeated `times` times, each on its own line
4. A summary line exactly matching: `[SUMMARY: <total_lines> lines, <total_functions> functions, completed all steps].`

Constraints:

- No `if __name__ == "__main__"` block
- No explanatory text outside code blocks and summary line
- Each function in its own fenced Python code block

## Scoring Rubric

| Criterion | Max | Description |
|---|---|---|
| `greet_user` function | 5 | Correct signature and return string |
| `validate_name` function | 5 | Length check, alphabetic check, boolean return |
| `format_greeting` function | 5 | Handles `times < 1`, repeats greeting on separate lines |
| Summary line format | 3 | Exact required format |
| No extra prose/main block | 2 | No main block, no extra prose, three code fences |

## Temperature

Default temperature can be set with:

```json
"multi-step_temperature": 0.2
```

## Tips for Models

- Use exact function signatures.
- Place each function in its own ` ```python ` block.
- End with the summary line matching the format exactly, including the trailing period.
