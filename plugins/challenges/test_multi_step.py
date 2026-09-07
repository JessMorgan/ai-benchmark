"""Tests for the executable multi-step challenge."""
from plugins.challenges.multi_step import MultiStepPlugin


def full_response():
    return """```python
def greet_user(name: str) -> str:
    return f"Hello, {name}! Welcome."
```
```python
def validate_name(name: str) -> bool:
    return bool(name.strip()) and name.replace(' ', '').isalpha() and len(name) <= 50
```
```python
def format_greeting(greeting: str, times: int) -> str:
    return '' if times < 1 else '\\n'.join([greeting] * times)
```
[SUMMARY: 3 functions, 3 code blocks, completed all steps]."""


def liar_response():
    """Response whose string functions return an ``__eq__``-liar object.

    The liar passes every ``==`` comparison against the expected values, so
    value-only behavioral asserts credit it as a full implementation.
    """
    return """```python
class _Liar:
    def __eq__(self, other):
        return True
def greet_user(name: str) -> str:
    return _Liar()
```
```python
def validate_name(name: str) -> bool:
    return bool(name.strip()) and name.replace(' ', '').isalpha() and len(name) <= 50
```
```python
def format_greeting(greeting: str, times: int) -> str:
    return _Liar()
```
[SUMMARY: 3 functions, 3 code blocks, completed all steps]."""


def test_empty_response_scores_zero():
    assert MultiStepPlugin().score("") == 0.0


def test_eq_liar_loses_behavioral_points():
    score = MultiStepPlugin().score(liar_response())
    assert score < 20.0


def test_complete_response_scores_full():
    assert MultiStepPlugin().score(full_response()) == 20.0


def test_wrong_behavior_loses_behavioral_points():
    response = full_response().replace("return f\"Hello, {name}! Welcome.\"", "return name")
    assert MultiStepPlugin().score(response) < 20.0


def test_missing_fences_loses_contract_points():
    assert MultiStepPlugin().score("def greet_user(name):\n    return name") < 10.0
