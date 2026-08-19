"""Focused scoring tests for executable and structured challenges."""
import json

from plugins import discover_plugins
from plugins.challenges.data_transformation import (
    DATA_TRANSFORMATION_EXPECTED_OUTPUT,
    DATA_TRANSFORMATION_RESPONSE_SCHEMA,
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


def test_data_transformation_requests_strict_json_schema():
    request_params = plugin("data-transformation").get_request_params({})
    response_format = request_params["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "data_transformation_result"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == DATA_TRANSFORMATION_RESPONSE_SCHEMA
    schema_text = json.dumps(DATA_TRANSFORMATION_RESPONSE_SCHEMA)
    assert "minLength" not in schema_text
    assert "maxLength" not in schema_text
    assert "minItems" not in schema_text
    assert "maxItems" not in schema_text
    prompt = plugin("data-transformation").get_prompt()
    assert "filtered or superseded records" in prompt
    assert "Sort retained records" in prompt


def test_other_plugins_do_not_opt_into_response_schemas():
    assert plugin("code-review").get_request_params({}) == {}
    assert plugin("reasoning").get_request_params({}) == {}


def test_data_transformation_current_records_score_full():
    assert plugin("data-transformation").score(json.dumps(DATA_TRANSFORMATION_EXPECTED_OUTPUT)) == 22.0


def test_data_transformation_cardinality_is_checked_locally():
    plugin_instance = plugin("data-transformation")
    too_few = json.loads(json.dumps(DATA_TRANSFORMATION_EXPECTED_OUTPUT))
    too_few["records"] = []
    too_many = json.loads(json.dumps(DATA_TRANSFORMATION_EXPECTED_OUTPUT))
    too_many["records"].append({
        "order_id": "O-210", "customer": "Test Person", "total": 55.0, "rank": 6,
    })

    for payload in (too_few, too_many):
        result = plugin_instance.evaluate(json.dumps(payload))
        assert result.diagnostics["response_schema_valid"] is False
        assert "records must contain between 1 and 5 items" in result.diagnostics["response_schema_errors"]


def test_data_transformation_filtered_or_superseded_record_loses_credit():
    payload = json.loads(json.dumps(DATA_TRANSFORMATION_EXPECTED_OUTPUT))
    payload["records"].append({
        "order_id": "O-205", "customer": "Eve Stone", "total": 49.99, "rank": 6,
    })
    result = plugin("data-transformation").evaluate(json.dumps(payload))
    assert result.score < 22.0


def test_data_transformation_requires_latest_version_sorting_and_summary():
    payload = json.loads(json.dumps(DATA_TRANSFORMATION_EXPECTED_OUTPUT))
    payload["records"][0]["total"] = 120.0
    payload["records"].reverse()
    payload["summary"]["total"] = 540.0
    result = plugin("data-transformation").evaluate(json.dumps(payload))
    assert next(
        item for item in result.rubric if item["name"] == "Deduplication and latest versions"
    )["earned"] < 4.0
    assert next(
        item for item in result.rubric if item["name"] == "Sorting and ranking"
    )["earned"] < 3.0
    assert next(
        item for item in result.rubric if item["name"] == "Derived summary"
    )["earned"] < 3.0


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
