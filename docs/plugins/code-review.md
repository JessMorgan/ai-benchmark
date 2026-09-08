# Code Review Plugin

| Property | Value |
|---|---|
| ID | `code-review` |
| Name | Code Review |
| Version | `1.2.0` |
| Max Score | 15 |
| Streaming | Yes |

## Task

The model is given a Python function and asked to identify bugs, anti-patterns, security issues, and maintainability problems. The response should be a JSON object with an `issues` array of issue objects (each with a `description`, `finding`, or `issue` field), a top-level JSON array of such issue objects, or bullet findings (including `•` bullets). A recognized JSON structure that yields no recognized findings is a dead end for the JSON path: the scorer falls back to bullet extraction and records a negative finding naming the format contract; the score is 0 only if no bullet findings are found either.

Example input function:

```python
import os
import time

def process_user_data(user_ids, db_path="/tmp/data.txt"):
    results = []
    f = open(db_path, "w")
    for i in range(len(user_ids)):
        user_id = user_ids[i]
        if user_id == None:
            continue
        data = fetch_data(user_id)
        if data:
            results.append(data)
    f.write(str(results))
    return results
```

## Scoring Rubric

| Criterion | Max | Description |
|---|---|---|
| File handle not closed / resource leak | 3 | A finding must assert both the defect (`f =`/`open(`) and the remediation (`close`/context manager/leak); denials ("closed properly", "no leak") do not count |
| `== None` instead of `is None` | 2 | A finding must assert both the identity-comparison defect and the remediation |
| Hardcoded `/tmp/data.txt` | 2 | A finding must assert the hardcoded path and a remediation (parameterize/inject) |
| Missing error handling / `fetch_data` may fail | 3 | A finding must assert the unguarded call and a remediation (try/except) |
| Unused imports | 2 | A finding must assert the unused imports; short keywords are word-bounded |
| Actionable / concrete fixes | 2 | Up to 2 points for remediation terms across the matched findings |
| Source citations | 1 | At least three distinct findings must cite a variable, call, or literal from the source |

Each finding satisfies at most one defect (one-to-one matching), and a finding matching a defect's denial patterns is a negation, not an assertion, and does not count.

## Temperature

Default temperature can be set with:

```json
"code_review_temperature": 0.3
```

## Tips for Models

- Return valid JSON.
- Be specific and cite the relevant code in each issue description.
- Suggest concrete fixes.
