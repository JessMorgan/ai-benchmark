"""Regression tests for the error-recovery challenge evaluator.

The evaluator historically crashed with ``AttributeError: 'NoneType' object
has no attribute '_fields'`` whenever a response failed Python parsing: the
``ast.walk(tree)`` call in the function-collection comprehension was evaluated
before the ``tree is not None`` guard. Any model emitting malformed code was
therefore recorded as a model-level failure instead of a low score. These tests
pin the contract that invalid Python must produce a normal, bounded evaluation.
"""
from plugins.challenges.error_recovery import ErrorRecoveryPlugin


def test_invalid_python_evaluates_to_zero_instead_of_crashing():
    # Non-Python text fails parsing (SyntaxError) but must not raise.
    result = ErrorRecoveryPlugin().evaluate("def (")
    assert result.score == 0.0
    assert result.diagnostics["errors"]


def test_empty_source_reports_no_python_found():
    result = ErrorRecoveryPlugin().evaluate("```python\n```")
    assert result.score == 0.0
    assert any("no Python source found" in str(e) for e in result.diagnostics["errors"])


def test_syntactically_invalid_python_returns_bounded_failure():
    # A fenced Python block that fails ``ast.parse`` must not walk a None tree.
    response = "```python\nclass WeatherClient(:\\n    async def fetch(self):\\n```"
    result = ErrorRecoveryPlugin().evaluate(response)
    assert result.score < 20.0
    assert any("SyntaxError" in str(e) for e in result.diagnostics["errors"])


def test_valid_but_incomplete_python_does_not_crash():
    # Valid Python that is still not a full implementation must score below max.
    response = "```python\nclass AllProvidersFailedError(Exception):\n    pass\n```"
    result = ErrorRecoveryPlugin().evaluate(response)
    assert 0.0 <= result.score < 20.0


def test_empty_response_evaluates_to_zero():
    result = ErrorRecoveryPlugin().evaluate("")
    assert result.score == 0.0
