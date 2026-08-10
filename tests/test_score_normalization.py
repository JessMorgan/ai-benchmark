"""Tests for percentage normalization at the benchmark boundary."""
import json
import math

import pytest

from benchmark.plugin import normalize_score, sanitize_diagnostics, serialize_rubric
from benchmark.state import BenchmarkState
from plugins.challenges.rate_limiter import RateLimiterPlugin


@pytest.mark.parametrize(
    ("raw", "maximum", "expected"),
    [
        (0, 20, 0),
        (20, 20, 100),
        (12.5, 20, 63),
        (12.49, 20, 62),
        (99.5, 100, 100),
        (-1, 20, 0),
        (25, 20, 100),
    ],
)
def test_normalize_score_boundaries(raw, maximum, expected):
    result = normalize_score(raw, maximum)
    assert result == expected
    assert type(result) is int


@pytest.mark.parametrize("raw", [math.nan, math.inf, -math.inf, "not-a-number", True])
def test_normalize_score_rejects_invalid_values(raw):
    with pytest.raises(ValueError):
        normalize_score(raw, 20)


def test_normalize_score_rejects_invalid_maximum():
    with pytest.raises(ValueError):
        normalize_score(1, 0)


def test_serialize_rubric_preserves_native_points_and_total():
    rubric = serialize_rubric([
        {
            "name": "Syntax",
            "max": 5,
            "earned": 4,
            "missed": 1,
            "matched": True,
            "evidence": [{"kind": "regex", "points": 2}],
        }
    ])
    assert rubric == [{
        "name": "Syntax",
        "points": 4,
        "total": 5,
        "matched": True,
        "evidence": [{"kind": "regex", "points": 2}],
    }]


def test_sanitize_diagnostics_preserves_native_debug_fields():
    diagnostics = sanitize_diagnostics({
        "validations": [{"points": 2, "nested": {"earned": 1}}],
        "score": 12.5,
    })
    assert diagnostics == {
        "validations": [{"points": 2, "nested": {"earned": 1}}],
        "score": 12.5,
    }


def test_state_persists_and_validates_percentage_schema(tmp_path):
    state = BenchmarkState({"model": "Local"}, ["rate-limiter"])
    state.update("model", status="completed", **{"rate-limiter_score": 75})
    path = tmp_path / "benchmark_state.json"
    state.save_state(str(path))

    saved = json.loads(path.read_text())
    assert saved["score_schema"] == "percentage-v1"
    loaded = BenchmarkState.load_state(
        str(path), {"model": "Local"}, ["rate-limiter"]
    )
    assert loaded.score_schema == "percentage-v1"

    saved["score_schema"] = "legacy-v0"
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="percentage-v1"):
        BenchmarkState.load_state(
            str(path), {"model": "Local"}, ["rate-limiter"]
        )


def test_builtin_score_remains_native_and_core_normalizes_it():
    plugin = RateLimiterPlugin()
    score = plugin.score("not a solution")
    assert isinstance(score, float)
    assert 0 <= score <= plugin.max_score
    assert normalize_score(score, plugin.max_score) == 0


def test_legacy_score_plugin_remains_native():
    from benchmark.plugin import BenchmarkTaskPlugin

    class Legacy(BenchmarkTaskPlugin):
        id = "legacy"
        version = "1.0.0"
        name = "Legacy"
        max_score = 20

        def get_prompt(self):
            return "prompt"

        def get_temperature(self, global_config):
            return None

        def score(self, response_text):
            return 12.5

    assert Legacy().score("answer") == 12.5
    assert Legacy().evaluate("answer").score == 12.5
