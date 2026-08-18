"""Tests for the non-scoring schema compatibility sentinel."""
import json
from unittest import mock

from benchmark.core import (
    SCHEMA_SENTINEL_SCHEMA,
    run_schema_sentinel,
    summarize_schema_compatibility,
)
from benchmark.http import NonStreamResult
from plugins.challenges.data_transformation import (
    DATA_TRANSFORMATION_EXPECTED_OUTPUT,
    DataTransformationPlugin,
)

SOURCE_CONFIG = {
    "Local": {
        "api_url": "http://localhost:11434/chat/completions",
        "headers": {},
    }
}


def test_sentinel_reports_likely_enforced_schema_without_score():
    response = NonStreamResult(
        '{"sentinel":"schema-enforced"}', "", {}, 0.1, None, "stop"
    )
    with mock.patch("benchmark.core.nonstream_request", return_value=response) as request:
        result = run_schema_sentinel(SOURCE_CONFIG, "Local", "model")

    assert result["status"] == "schema_likely_enforced"
    assert result["response_schema_valid"] is True
    assert result["schema_enforcement_verified"] is True
    assert request.call_args.kwargs["request_params"]["response_format"]["json_schema"]["schema"] == SCHEMA_SENTINEL_SCHEMA


def test_sentinel_detects_provider_that_returns_the_prompted_forbidden_value():
    response = NonStreamResult(
        '{"sentinel":"schema-not-enforced"}', "", {}, 0.1, None, "stop"
    )
    with mock.patch("benchmark.core.nonstream_request", return_value=response):
        result = run_schema_sentinel(SOURCE_CONFIG, "Local", "model")

    assert result["status"] == "schema_accepted_invalid"
    assert result["response_schema_valid"] is False
    assert result["schema_enforcement_verified"] is False


def test_sentinel_distinguishes_schema_rejection_from_transport_failure():
    rejected = NonStreamResult("", "", {}, 0.1, "HTTP 400: failed to parse response_format schema", None)
    with mock.patch("benchmark.core.nonstream_request", return_value=rejected):
        result = run_schema_sentinel(SOURCE_CONFIG, "Local", "model")
    assert result["status"] == "schema_rejected"

    transport = NonStreamResult("", "", {}, 0.1, "Read timed out", None)
    with mock.patch("benchmark.core.nonstream_request", return_value=transport):
        result = run_schema_sentinel(SOURCE_CONFIG, "Local", "model")
    assert result["status"] == "schema_transport_error"


def test_sentinel_marks_native_non_openai_protocols_as_not_supported():
    source = {"1min": {"api_protocol": "1min"}}
    result = run_schema_sentinel(source, "1min", "model")
    assert result["status"] == "schema_not_supported_by_source"


def test_data_transformation_schema_is_only_one_scored_point():
    plugin = DataTransformationPlugin()
    result = plugin.evaluate(json.dumps(DATA_TRANSFORMATION_EXPECTED_OUTPUT))
    schema = next(item for item in result.rubric if item["name"] == "Structured schema contract")
    assert schema["max"] == 1.0
    assert result.diagnostics["schema_requested"] is True
    assert result.diagnostics["response_schema_valid"] is True
    assert result.diagnostics["schema_enforcement_verified"] is False


def test_schema_summary_is_separate_from_task_score():
    plugin = DataTransformationPlugin()
    results = [{
        "data-transformation_score": 55,
        "data-transformation_schema_requested": True,
        "data-transformation_schema_request_status": "schema_accepted_valid",
        "data-transformation_response_schema_valid": True,
        "data-transformation_schema_enforcement_verified": False,
    }, {
        "data-transformation_score": "fail",
        "data-transformation_schema_requested": True,
        "data-transformation_schema_request_status": "schema_rejected",
        "data-transformation_response_schema_valid": False,
        "data-transformation_schema_enforcement_verified": False,
    }]
    summary = summarize_schema_compatibility(results, [plugin])
    assert summary["requested_cells"] == 2
    assert summary["response_valid_cells"] == 1
    assert summary["response_schema_valid_rate"] == 0.5
    assert summary["by_plugin"]["data-transformation"]["statuses"] == {
        "schema_accepted_valid": 1,
        "schema_rejected": 1,
    }


def test_data_transformation_valid_shape_with_wrong_values_keeps_schema_metadata():
    payload = json.loads(json.dumps(DATA_TRANSFORMATION_EXPECTED_OUTPUT))
    payload["records"][0]["customer"] = "Wrong Person"
    result = DataTransformationPlugin().evaluate(json.dumps(payload))
    assert result.diagnostics["response_schema_valid"] is True
    assert result.score < 22.0
