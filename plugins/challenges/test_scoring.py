"""Focused scoring tests for executable and structured challenges."""
import json

from plugins import discover_plugins
from plugins.challenges.structured_output import (
    STRUCTURED_OUTPUT_EXPECTED_RECORD,
    STRUCTURED_OUTPUT_RESPONSE_SCHEMA,
)


def plugin(plugin_id):
    return next(item for item in discover_plugins() if item.id == plugin_id)


def test_rate_limiter_requires_all_three_strategies():
    result = plugin("rate-limiter").evaluate("class TokenBucket: pass\nclass SlidingWindowLog: pass")
    assert next(item for item in result.rubric if item["name"] == "FixedWindow")["earned"] == 0.0


def test_rate_limiter_stub_is_not_implementation_credit():
    assert plugin("rate-limiter").score("class TokenBucket:\n    pass") < 5.0


def test_moe_keyword_listing_without_sections_is_limited():
    score = plugin("moe-dense").score("MoE gating, load balancing, training, inference, benchmarks, and papers.")
    assert score < 10.0


def test_tool_calling_requires_exact_call_set():
    response = "<plan>get_weather</plan>\n" + "\n".join(
        '<tool_call>{"name":"get_weather","args":{"location":"Tokyo","unit":"celsius"}}</tool_call>'
        for _ in range(6)
    )
    assert plugin("tool-calling").score(response) < 15.0


def test_structured_output_requests_strict_json_schema():
    request_params = plugin("structured-output").get_request_params({})
    response_format = request_params["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "structured_employee_record"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == STRUCTURED_OUTPUT_RESPONSE_SCHEMA
    schema_text = json.dumps(STRUCTURED_OUTPUT_RESPONSE_SCHEMA)
    assert "minLength" not in schema_text
    assert "maxLength" not in schema_text
    prompt = plugin("structured-output").get_prompt()
    assert "non-empty strings" in prompt
    assert "name non-empty string" in prompt


def test_other_plugins_do_not_opt_into_response_schemas():
    assert plugin("code-review").get_request_params({}) == {}
    assert plugin("reasoning").get_request_params({}) == {}


def test_structured_output_current_profile_scores_full():
    assert plugin("structured-output").score(json.dumps(STRUCTURED_OUTPUT_EXPECTED_RECORD)) == 22.0


def test_structured_output_archived_decoy_does_not_score_as_current_profile():
    payload = json.loads(json.dumps(STRUCTURED_OUTPUT_EXPECTED_RECORD))
    payload["name"] = "Casey Rivera"
    payload["email"] = "casey.rivera@old-example.net"
    payload["department"] = "Sales"
    payload["age"] = 29
    assert plugin("structured-output").score(json.dumps(payload)) < 22.0


def test_structured_output_requires_normalization_and_derived_values():
    payload = json.loads(json.dumps(STRUCTURED_OUTPUT_EXPECTED_RECORD))
    payload["name"] = "Rivera, Jordan"
    payload["age"] = 35
    payload["roles"] = ["auditor", "admin"]
    payload["metadata"]["score"] = 85
    result = plugin("structured-output").evaluate(json.dumps(payload))
    normalization = next(
        item for item in result.rubric if item["name"] == "Normalization and derived values"
    )
    assert normalization["earned"] < normalization["max"]
    assert result.score < 22.0


def test_multi_step_requires_behavior_not_just_definitions():
    response = """```python
def greet_user(name: str) -> str:
    return f'Hello, {name}! Welcome.'
```
```python
def validate_name(name: str) -> bool:
    return bool(name) and name.replace(' ', '').isalpha() and len(name) <= 50
```
```python
def format_greeting(greeting: str, times: int) -> str:
    return '' if times < 1 else '\\n'.join([greeting] * times)
```
[SUMMARY: 3 functions, 3 code blocks, completed all steps]."""
    assert plugin("multi-step").score(response) >= 18.0
