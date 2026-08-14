"""Regression tests for cross-plugin scoring invariants.

These tests deliberately assert contracts that broad "greater than zero" tests
missed: declared scale, empty responses, rubric bookkeeping, penalty ordering,
and penalty evidence.
"""
import json

import pytest

from plugins import discover_plugins

EXPECTED_MAX_SCORES = {
    "code-review": 15.0,
    "debug-traversal": 20.0,
    "error-recovery": 20.0,
    "instruction-following": 20.0,
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


def plugins_by_id():
    return {plugin.id: plugin for plugin in discover_plugins()}


def structured_payload():
    return {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "name": "Alice",
        "age": 30,
        "email": "alice@example.com",
        "department": "Engineering",
        "roles": ["admin"],
        "address": {
            "street": "123 Main St",
            "city": "Springfield",
            "state": "IL",
            "zip": "62701",
        },
        "settings": {
            "theme": "dark",
            "notifications": {"email": True, "sms": False, "push": True},
            "language": "en",
        },
        "tags": [{"name": "full-time", "priority": 1}],
        "metadata": {
            "created_at": "2024-01-15T09:30:00Z",
            "active": True,
            "score": 0.95,
        },
    }


def test_every_plugin_declares_the_sum_of_its_rubric_scale():
    assert {plugin.id: plugin.max_score for plugin in discover_plugins()} == EXPECTED_MAX_SCORES


@pytest.mark.parametrize("plugin_id", sorted(EXPECTED_MAX_SCORES))
def test_every_plugin_scores_an_empty_response_zero(plugin_id):
    plugin = plugins_by_id()[plugin_id]
    assert plugin.score("") == 0.0


@pytest.mark.parametrize(
    ("plugin_id", "response", "criterion"),
    [
        (
            "code-review",
            '{"issues": [{"description": "There is a problem."}]}',
            "Actionable / concrete fix",
        ),
        (
            "debug-traversal",
            "## Root Cause\nThe bug is a user_ido typo.\n## Analysis\nTrace the code.\n",
            "Depth of analysis",
        ),
        (
            "error-recovery",
            "class AllProvidersFailedError(Exception):\n    ...\n",
            "Structure / completeness",
        ),
        (
            "instruction-following",
            "- forbidden extra prose",
            "Exact response discipline",
        ),
        (
            "moe-dense",
            "MoE gating and load balancing are important.",
            "Gating/routing mechanism",
        ),
        (
            "multi-step",
            (
                "```python\ndef greet_user(name):\n    return name\n```\n"
                "if __name__ == '__main__':\n    print('bad')"
            ),
            "No extra prose/main block",
        ),
        (
            "multi-turn-conversation",
            (
                "## Version 1\nHello.\n## Version 2\nHello.\n## Version 3\nWarm future note.\n"
                "## Summary of Changes\nChanged the tone."
            ),
            "Evidence of iteration across versions",
        ),
        (
            "orchestration",
            "Task 1 [PARALLEL] [SEQUENTIAL]",
            "Parallel vs sequential logic",
        ),
        (
            "prd-creation",
            "# Executive Summary\nSame body.\n# Executive Summary\nSame body.",
            "Functional Requirements",
        ),
        (
            "rate-limiter",
            "class TokenBucket:\n    pass\n",
            "Token Bucket",
        ),
        (
            "reasoning",
            "FAILED_SERVICE: Search\nOWNER: Ben\nPRIORITY: P4\nTIME: 09:30",
            "Time-chain deductions",
        ),
        (
            "software-architecture",
            "Microservices are always the answer.",
            "Architecture & Patterns",
        ),
        (
            "structured-output",
            "```json\n" + json.dumps(structured_payload()) + "\n```\nExtra prose",
            "Strict format (no extra keys)",
        ),
        (
            "tool-calling",
            '<tool_call>{"name": "get_weather", "args": {}}</tool_call>',
            "Output format compliance",
        ),
        (
            "wireframes",
            "## Dashboard\nA button.",
            "Navigation flows",
        ),
    ],
)
def test_each_available_penalty_is_attached_to_a_rubric_category(
    plugin_id, response, criterion
):
    result = plugins_by_id()[plugin_id].evaluate(response)
    assert not any(
        "cannot penalize unknown criterion" in error
        for error in result.diagnostics.get("errors", [])
    )
    item = next(item for item in result.rubric if item["name"] == criterion)
    assert item["negative_findings"], (plugin_id, criterion, result)


def test_structured_output_type_checks_fit_the_declared_criterion_weight():
    result = plugins_by_id()["structured-output"].evaluate(
        json.dumps(structured_payload())
    )
    types = next(item for item in result.rubric if item["name"] == "Basic types and constraints")
    assert types["earned"] == 6.0
    assert types["max"] == 6.0


def test_structured_output_invalid_data_keeps_a_complete_rubric():
    result = plugins_by_id()["structured-output"].evaluate("not structured data")
    names = [item["name"] for item in result.rubric]
    assert names == [
        "Valid JSON/YAML syntax",
        "Required top-level fields",
        "Basic types and constraints",
        "Non-empty values / completeness",
        "Strict format (no extra keys)",
        "No placeholder values",
    ]
    assert sum(item["max"] for item in result.rubric) == 22.0
    assert result.score == 0.0


def test_multi_turn_missing_summary_keeps_a_zero_point_rubric_category():
    result = plugins_by_id()["multi-turn-conversation"].evaluate(
        "## Version 1\nAn email.\n## Version 2\nA warmer email.\n## Version 3\nA warmest email."
    )
    summary = next(item for item in result.rubric if item["name"] == "Summary of changes")
    assert summary["earned"] == 0.0
    assert summary["max"] == 2.0


def test_structured_output_penalty_is_reconstructable_from_rubric():
    result = plugins_by_id()["structured-output"].evaluate(
        "```json\n" + json.dumps(structured_payload()) + "\n```\nExtra prose"
    )
    assert sum(item["earned"] for item in result.rubric) == result.score
    assert any(
        finding.get("finding") == "response contains explanatory text outside structured data"
        for item in result.rubric
        for finding in item["negative_findings"]
    )


def test_code_review_keeps_correctness_credit_for_understandable_non_json_findings():
    result = plugins_by_id()["code-review"].evaluate(
        "- The file handle is never closed; use a context manager.\n"
        "- The code uses == None; compare with is None."
    )
    assert result.score > 0.0
    assert result.diagnostics["validations"][0]["valid"] is False
