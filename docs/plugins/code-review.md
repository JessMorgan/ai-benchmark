# Code Review Plugin

| Property | Value |
|---|---|
| ID | `code-review` |
| Name | Code Review |
| Version | `0.2.0` |
| Max Score | 15 |
| Streaming | No |

## Task

The model is given a Python function and asked to identify bugs, anti-patterns, security issues, and maintainability problems. The response must be a JSON object with a single key `issues`, where each issue has a `description` field.

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
| File handle not closed | 3 | Resource leak / context manager |
| `== None` instead of `is None` | 2 | Identity comparison |
| Hardcoded `/tmp/data.txt` | 2 | Path handling |
| Missing error handling | 3 | `fetch_data` may fail |
| Unused imports | 2 | `os` and `time` not used |
| Actionable fix | 3 | Specific recommendation |

## Temperature

Default temperature can be set with:

```json
"code-review_temperature": 0.3
```

## Tips for Models

- Return valid JSON.
- Be specific and cite the relevant code in each issue description.
- Suggest concrete fixes.
