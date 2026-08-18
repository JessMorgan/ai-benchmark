"""Focused scoring tests for executable and structured challenges."""
import json

from plugins import discover_plugins
from plugins.challenges.structured_output import STRUCTURED_OUTPUT_RESPONSE_SCHEMA


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


def test_structured_output_valid_payload_scores_full():
    payload = {
        "id": "550e8400-e29b-41d4-a716-446655440000", "name": "Alice", "age": 30,
        "email": "alice@example.com", "department": "Engineering", "roles": ["admin"],
        "address": {"street": "123 Main", "city": "Springfield", "state": "IL", "zip": "62701"},
        "settings": {"theme": "dark", "notifications": {"email": True, "sms": False, "push": True}, "language": "en"},
        "tags": [{"name": "remote", "priority": 1}],
        "metadata": {"created_at": "2024-01-15T09:30:00Z", "active": True, "score": 0.95},
    }
    assert plugin("structured-output").score(json.dumps(payload)) == 22.0


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
