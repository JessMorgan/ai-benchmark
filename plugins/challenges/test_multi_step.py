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


def unfenced_response():
    """Correct code plus the mandated summary line, but no Python fences.

    The trailing summary line is not Python; before the fix it corrupted the
    raw-ast.parse fallback and collapsed this correct response to ~1 point.
    """
    return (
        "def greet_user(name: str) -> str:\n"
        '    return f"Hello, {name}! Welcome."\n'
        "def validate_name(name: str) -> bool:\n"
        "    return bool(name.strip()) and name.replace(' ', '').isalpha() and len(name) <= 50\n"
        "def format_greeting(greeting: str, times: int) -> str:\n"
        "    return '' if times < 1 else '\\n'.join([greeting] * times)\n"
        "[SUMMARY: 3 functions, 3 code blocks, completed all steps]."
    )


def test_unfenced_correct_response_not_zeroed():
    score = MultiStepPlugin().score(unfenced_response())
    assert score >= 15.0


def nested_response():
    """The three functions wrapped inside a class, exposing no module-level API.

    Before the fix, ast.walk name matching credited the nested methods for the
    contract even though no module-level API exists (measured 5/20).
    """
    return """```python
class _Impl:
    def greet_user(self, name: str) -> str:
        return f"Hello, {name}! Welcome."
    def validate_name(self, name: str) -> bool:
        return bool(name.strip()) and name.replace(' ', '').isalpha() and len(name) <= 50
    def format_greeting(self, greeting: str, times: int) -> str:
        return '' if times < 1 else '\\n'.join([greeting] * times)
```
[SUMMARY: 3 functions, 3 code blocks, completed all steps]."""


def test_nested_functions_no_module_api_score_low():
    score = MultiStepPlugin().score(nested_response())
    assert score < 5.0


def text_fence_prose_response():
    """Three correct Python blocks plus a ```text fence holding prose.

    Before the fix the prose scan removed all fences, so the prose in the
    text fence was invisible and the response scored 20/20.
    """
    return full_response().replace(
        "[SUMMARY:",
        "```text\nHere is the explanation of the code above.\n```\n[SUMMARY:",
    )


def test_prose_in_text_fence_is_penalized():
    score = MultiStepPlugin().score(text_fence_prose_response())
    assert score < 20.0


def main_guard_response():
    """A __main__ guard inside the first Python block.

    Before the fix the __main__ scan only looked outside fences, so an
    in-block guard was invisible and the response scored 20/20.
    """
    return full_response().replace(
        '    return f"Hello, {name}! Welcome."\n```',
        '    return f"Hello, {name}! Welcome."\nif __name__ == "__main__":\n    print(greet_user("Ada"))\n```',
    )


def test_main_guard_inside_block_is_penalized():
    score = MultiStepPlugin().score(main_guard_response())
    assert score < 20.0


def codeless_response():
    """A response with no Python blocks and no forbidden content.

    Before the fix the 'no forbidden prose' discipline point was credited
    for free (the response has no forbidden content), so a codeless
    response earned 1.0 for it. After the fix the discipline point is
    gated on >=1 Python block defining a required function, so a
    codeless response earns 0.0.
    """
    return "[SUMMARY: 3 functions, 3 code blocks, completed all steps]."


def test_codeless_response_no_discipline_point():
    result = MultiStepPlugin().evaluate(codeless_response())
    discipline = next(
        c for c in result.rubric if c["name"] == "No forbidden prose or main block"
    )
    assert discipline["earned"] == 0.0
