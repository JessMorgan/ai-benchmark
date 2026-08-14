"""Cross-plugin scoring contracts and adversarial regressions."""
import json

import pytest

from plugins import discover_plugins

EXPECTED_MAX_SCORES = {
    "code-review": 15.0,
    "debug-consistency": 20.0,
    "debug-traversal": 20.0,
    "error-recovery": 20.0,
    "event-processor": 20.0,
    "instruction-following": 20.0,
    "long-context": 20.0,
    "moe-dense": 17.0,
    "multi-step": 20.0,
    "multi-turn-conversation": 20.0,
    "orchestration": 16.0,
    "prd-creation": 22.0,
    "rate-limiter": 20.0,
    "reasoning": 20.0,
    "software-architecture": 20.0,
    "structured-output": 22.0,
    "tool-calling": 25.0,
    "wireframes": 20.0,
}


def by_id():
    return {plugin.id: plugin for plugin in discover_plugins()}


def test_inventory_and_native_scales_are_explicit():
    assert {plugin.id: plugin.max_score for plugin in discover_plugins()} == EXPECTED_MAX_SCORES


@pytest.mark.parametrize("plugin_id", sorted(EXPECTED_MAX_SCORES))
def test_empty_response_is_zero(plugin_id):
    assert by_id()[plugin_id].score("") == 0.0


def test_debug_traversal_rejects_the_old_invented_identifier_answer():
    result = by_id()["debug-traversal"].evaluate(
        "## Root Cause\nThe user_ido key is wrong.\n## Analysis\nTrace the code.\n"
    )
    assert result.score < 10.0


def test_debug_consistency_requires_challenging_the_report():
    result = by_id()["debug-consistency"].evaluate(
        "## Reproduction\nIt returns empty.\n## Consistency Check\nThere is a bug.\n## Diagnosis\nFix it.\n## Evidence Needed\nLogs.\n## Recommendation\nPatch it."
    )
    assert result.score < 10.0


def test_prd_does_not_count_three_stories_from_one_greedy_match():
    result = by_id()["prd-creation"].evaluate(
        "## User Stories\nAs a developer, I want focus, so that I can work.\n"
    )
    item = next(item for item in result.rubric if item["name"] == "User Stories")
    assert item["earned"] <= 1.0


def test_rate_limiter_requires_fixed_window():
    response = "class TokenBucket: pass\nclass SlidingWindowLog: pass"
    result = by_id()["rate-limiter"].evaluate(response)
    fixed = next(item for item in result.rubric if item["name"] == "FixedWindow")
    assert fixed["earned"] == 0.0


def test_structured_output_rejects_wrong_nested_types_and_explanatory_text():
    payload = {
        "id": "550e8400-e29b-41d4-a716-446655440000", "name": "Alice", "age": 30,
        "email": "alice@example.com", "department": "Engineering", "roles": ["admin"],
        "address": {"street": "123 Main", "city": "Springfield", "state": "IL", "zip": "62701"},
        "settings": {"theme": "dark", "notifications": {"email": "yes", "sms": False, "push": True}, "language": "en"},
        "tags": [{"name": "remote", "priority": 1}],
        "metadata": {"created_at": "2024-01-15T09:30:00Z", "active": True, "score": 0.9},
    }
    result = by_id()["structured-output"].evaluate("```json\n" + json.dumps(payload) + "\n```\nExtra prose")
    types = next(item for item in result.rubric if item["name"] == "Basic types and constraints")
    strict = next(item for item in result.rubric if item["name"] == "Strict format (no extra keys)")
    assert types["earned"] < types["max"]
    assert any("explanatory text" in finding["finding"] for finding in strict["negative_findings"])
    assert sum(item["earned"] for item in result.rubric) == result.score


def test_tool_calling_rejects_duplicate_calls_as_exact_contract():
    response = "<plan>get_weather</plan>\n" + "\n".join(
        '<tool_call>{"name":"get_weather","args":{"location":"Tokyo","unit":"celsius"}}</tool_call>'
        for _ in range(6)
    )
    result = by_id()["tool-calling"].evaluate(response)
    required = next(item for item in result.rubric if item["name"] == "Required tools present")
    assert required["earned"] < required["max"]


def test_rubric_is_reconstructable_for_nonempty_evaluations():
    probes = {
        "code-review": '{"issues":[{"description":"file handle is not closed; use a context manager"}]}',
        "instruction-following": "ORDER T-02 | CUSTOMER JULES | TOTAL 120.00",
        "reasoning": "FAILED_SERVICE: Search\nOWNER: Ben\nPRIORITY: P4\nTIME: 09:30",
        "long-context": "INCIDENT: I-17\nOWNER: Omar\nESCALATION CHANNEL: PagerDuty\nEVIDENCE: F02 F05 F09\nREASONING: EU 14:30 P1 I-17 PagerDuty",
    }
    for plugin_id, response in probes.items():
        result = by_id()[plugin_id].evaluate(response)
        assert sum(item["earned"] for item in result.rubric) == result.score
        assert not any("unknown criterion" in error for error in result.diagnostics.get("errors", []))
