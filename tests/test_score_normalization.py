"""Tests for the public percentage-v1 score contract."""
import json
import math

import pytest

from benchmark.plugin import normalize_rubric, normalize_score, sanitize_diagnostics
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


def test_normalize_rubric_contains_only_percentage_fields():
    rubric = normalize_rubric([
        {
            "name": "Syntax",
            "max": 5,
            "earned": 4,
            "missed": 1,
            "matched": True,
            "evidence": [{"kind": "regex", "points": 2}],
        }
    ], 20)
    assert rubric == [{
        "name": "Syntax",
        "score_percent": 80,
        "weight_percent": 25,
        "matched": True,
        "evidence": [{"kind": "regex", "points_percent": 40}],
    }]
    assert not {"max", "earned", "missed"}.intersection(rubric[0])


def test_sanitize_diagnostics_removes_native_point_fields_recursively():
    diagnostics = sanitize_diagnostics({
        "validations": [{"points": 2, "points_percent": 40, "nested": {"earned": 1}}],
        "score": 80,
    })
    assert diagnostics == {
        "validations": [{"points_percent": 40, "nested": {}}],
        "score": 80,
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


def test_builtin_public_score_is_integer_percentage():
    plugin = RateLimiterPlugin()
    score = plugin.score("not a solution")
    assert type(score) is int
    assert 0 <= score <= 100


def test_legacy_score_plugin_uses_normalized_public_api():
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

    assert Legacy().score("answer") == 63
    assert Legacy().evaluate("answer").score == 12.5
