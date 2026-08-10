"""Tests for deterministic negative and contradiction checks."""
from plugins import discover_plugins


def plugin(plugin_id):
    return next(item for item in discover_plugins() if item.id == plugin_id)


def test_debug_traversal_penalizes_invented_identifier():
    text = """
## Root Cause
The bug is a user_ido typo.
## Analysis
Trace the execution and line by line logic.
## Fix
```python
return []
```
## Test
```python
def test_fix():
    assert True
```
## Side Effects
Consider performance and callers.
"""
    result = plugin("debug-traversal").evaluate(text)
    finding = result.rubric[1]["negative_findings"] + result.rubric[0]["negative_findings"]
    assert finding
    assert result.score < plugin("debug-traversal").max_score


def test_rate_limiter_penalizes_placeholder_lines():
    result = plugin("rate-limiter").evaluate("class TokenBucket:\n    pass\n")
    assert any(item["negative_findings"] for item in result.rubric)


def test_error_recovery_penalizes_ellipsis():
    result = plugin("error-recovery").evaluate("class AllProvidersFailedError(Exception):\n    ...\n")
    assert any(item["negative_findings"] for item in result.rubric)


def test_duplicate_conversation_versions_are_penalized():
    version = "Hello, thank you for the interview."
    text = f"## Version 1\n{version}\n## Version 2\n{version}\n## Version 3\nA warm future message.\n## Summary of Changes\nChanged tone."
    result = plugin("multi-turn-conversation").evaluate(text)
    assert any(item["negative_findings"] for item in result.rubric)
